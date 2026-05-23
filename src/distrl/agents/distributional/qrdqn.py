"""Quantile Regression DQN agent.

The default configuration (quantile_mode='midpoint', risk_type='mean') is
exactly vanilla QR-DQN. The agent additionally supports non-uniform quantile
positioning (Gauss-Legendre, trapezoidal-with-fixed-endpoints) and a
"truncated" CVaR variant that only learns the bottom-k quantiles. See
`quantile_modes.py` for details.
"""

import os
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from src.distrl.agents.base import BaseAgent
from src.distrl.agents.distributional.quantile_modes import (
    QuantileScheme,
    build_scheme,
)
from src.distrl.agents.networks import (
    QuantileHead,
    UnifiedQNet,
    build_trunk,
)


class QRDQNAgent(BaseAgent):
    """Quantile Regression DQN with pluggable quantile-positioning schemes."""

    def __init__(
        self,
        config: dict[str, Any],
        observation_space: Any,
        action_space: Any,
        device: str = "cpu",
    ) -> None:
        super().__init__(config, observation_space, action_space, device)

        action_dim = action_space.n

        self.num_quantiles = int(config.get("num_quantiles", 50))
        self.kappa = float(config.get("kappa", 1.0))
        self.risk_type = config.get("risk_type", "mean")
        self.risk_fraction = float(config.get("risk_fraction", 0.1))
        self.quantile_mode = config.get("quantile_mode", "midpoint")
        self.truncate_upper = bool(
            config.get("truncate_upper_quantiles", False)
        )
        self.q_min = float(config.get("q_min", 0.0))
        self.q_max = float(config.get("q_max", 50.0))
        self.beta_alpha = float(config.get("beta_alpha", 2.0))
        self.beta_beta = float(config.get("beta_beta", 2.0))

        self.scheme: QuantileScheme = build_scheme(
            mode=self.quantile_mode,
            num_quantiles=self.num_quantiles,
            device=self.device,
            q_min=self.q_min,
            q_max=self.q_max,
            risk_type=self.risk_type,
            risk_fraction=self.risk_fraction,
            truncate_upper=self.truncate_upper,
            beta_alpha=self.beta_alpha,
            beta_beta=self.beta_beta,
        )

        def make_qnet() -> UnifiedQNet:
            trunk = build_trunk(config, observation_space)
            head = QuantileHead(
                trunk.output_dim, action_dim, self.scheme.num_predicted,
            )
            return UnifiedQNet(trunk, head).to(self.device)

        # Build q_net and target_net as independent module instances so the
        # soft-update copy is real (not a shared-reference no-op).
        self.q_net = make_qnet()
        self.target_net = make_qnet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.adam_eps = float(config.get("adam_eps", 1e-8))
        gc = config.get("grad_clip", None)
        self.grad_clip = float(gc) if gc is not None else None

        # Fused Adam runs the entire update in a single kernel per param
        # group rather than the per-tensor Python loop. Big win on
        # accelerators (CUDA/XPU); CPU's foreach path is already
        # competitive so we leave it off there. Verified to work on
        # PyTorch 2.11 + XPU.
        use_fused = self.device.type != "cpu"
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=float(config.get("lr", 1e-4)),
            eps=self.adam_eps,
            fused=use_fused,
        )

    def _action_values(self, predicted: torch.Tensor) -> torch.Tensor:
        """Aggregate predicted quantiles into per-action Q-values.

        Args:
            predicted: shape [..., A, num_predicted].
        Returns:
            Shape [..., A].
        """
        if self.risk_type == "cvar":
            return self.scheme.cvar(predicted)
        return self.scheme.expectation(predicted)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.rand() < epsilon:
            return int(self.action_space.sample())

        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            predicted = self.q_net(state_t)  # [1, A, num_predicted]
            q_values = self._action_values(predicted)  # [1, A]
            return int(q_values.argmax(dim=1).item())

    def train_step(
        self, batch: tuple[torch.Tensor, ...]
    ) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        batch_size = states.size(0)

        # Per-quantile slot count of the network head.
        num_predicted = self.scheme.num_predicted

        with torch.no_grad():
            next_predicted = self.target_net(next_states)
            next_q = self._action_values(next_predicted)
            best_next_actions = next_q.argmax(dim=1)  # [B]
            # torch.gather along the action axis. Equivalent to
            # next_predicted[range(B), best_next_actions] but stays
            # device-native — the previous form built a Python range,
            # converted to a tensor, and shipped it to the agent device
            # on every train step.
            best_next_predicted = next_predicted.gather(
                1,
                best_next_actions.view(-1, 1, 1).expand(
                    -1, 1, num_predicted,
                ),
            ).squeeze(1)  # [B, num_predicted]
            # Assemble the full target distribution. For trapezoidal this
            # prepends q_min and appends q_max; for other schemes it is a
            # no-op.
            best_next_assembled = self.scheme.assemble_full(
                best_next_predicted
            )  # [B, num_total]
            target_quantiles = (
                rewards
                + (1 - dones) * self.gamma * best_next_assembled
            )  # [B, num_total]

        current_all = self.q_net(states)
        # Same gather pattern for the action the agent actually took.
        # actions has shape [B, 1]; expand to [B, 1, num_predicted].
        current_quantiles = current_all.gather(
            1, actions.unsqueeze(-1).expand(-1, -1, num_predicted),
        ).squeeze(1)  # [B, num_predicted]

        # Pinball matrix over (predictor i, target j).
        # diff: [B, num_predicted, num_total]
        diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        abs_diff = diff.abs()
        huber_loss = torch.where(
            abs_diff <= self.kappa,
            0.5 * diff.pow(2),
            self.kappa * (abs_diff - 0.5 * self.kappa),
        )
        tau = self.scheme.tau.view(1, -1, 1)
        asymmetry = torch.abs(tau - (diff.detach() < 0).float())
        pinball_loss = asymmetry * huber_loss  # [B, num_predicted, num_total]

        # Weighted aggregation. For non-uniform schemes (Gauss-Legendre,
        # trapezoidal) a plain .mean() silently reverts to a uniform
        # midpoint aggregator and discards the quadrature.
        #   - target axis (dim=2) uses mean_weights, the probability mass
        #     of each atom in the target distribution.
        #   - predictor axis (dim=1) uses predictor_weights, the integration
        #     weights for the loss across tau values.
        # For midpoint these both reduce to uniform 1/N and the result is
        # identical to .mean() — backward compatible.
        target_w = self.scheme.mean_weights.view(1, 1, -1)
        per_predictor_loss = (pinball_loss * target_w).sum(dim=2)  # [B, P]
        pred_w = self.scheme.predictor_weights.view(1, -1)
        per_sample_loss = (per_predictor_loss * pred_w).sum(dim=1)  # [B]
        loss = per_sample_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.grad_clip,
            )
        self.optimizer.step()

        self.update_counter += 1
        self._update_target(self.q_net, self.target_net)

        # Return loss as a detached 0-d tensor instead of forcing a D2H
        # sync via .item(). The caller is expected to accumulate these and
        # materialise once per logging interval — every .item() during
        # training is an XPU pipeline stall.
        return {"loss": loss.detach()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # Stash the scheme settings so checkpoints are self-describing.
                "scheme": {
                    "mode": self.scheme.mode,
                    "num_predicted": self.scheme.num_predicted,
                    "quantile_mode": self.quantile_mode,
                    "num_quantiles": self.num_quantiles,
                    "risk_type": self.risk_type,
                    "risk_fraction": self.risk_fraction,
                    "truncate_upper": self.truncate_upper,
                    "q_min": self.q_min,
                    "q_max": self.q_max,
                    "beta_alpha": self.beta_alpha,
                    "beta_beta": self.beta_beta,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
