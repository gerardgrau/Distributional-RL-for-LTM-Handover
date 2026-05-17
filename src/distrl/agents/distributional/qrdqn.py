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
    CNNTrunk,
    MLPTrunk,
    QuantileHead,
    UnifiedQNet,
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

        self.scheme: QuantileScheme = build_scheme(
            mode=self.quantile_mode,
            num_quantiles=self.num_quantiles,
            device=self.device,
            q_min=self.q_min,
            q_max=self.q_max,
            risk_type=self.risk_type,
            risk_fraction=self.risk_fraction,
            truncate_upper=self.truncate_upper,
        )

        trunk_type = config.get("trunk_type", "mlp")
        if trunk_type == "cnn":
            in_channels = int(observation_space.shape[0])
            trunk = CNNTrunk(
                in_channels=in_channels,
                output_dim=int(config.get("cnn_feature_dim", 512)),
            )
        else:
            input_dim = int(observation_space.shape[0])
            trunk = MLPTrunk(
                input_dim, config.get("hidden_dims", [128, 128])
            )

        head = QuantileHead(
            trunk.output_dim, action_dim, self.scheme.num_predicted,
        )

        self.q_net = UnifiedQNet(trunk, head).to(self.device)
        # Build a fresh target net with a brand-new trunk+head so its
        # parameters are independent of self.q_net (avoid shared modules).
        if trunk_type == "cnn":
            tgt_trunk = CNNTrunk(
                in_channels=int(observation_space.shape[0]),
                output_dim=int(config.get("cnn_feature_dim", 512)),
            )
        else:
            tgt_trunk = MLPTrunk(
                int(observation_space.shape[0]),
                config.get("hidden_dims", [128, 128]),
            )
        tgt_head = QuantileHead(
            tgt_trunk.output_dim, action_dim, self.scheme.num_predicted,
        )
        self.target_net = UnifiedQNet(tgt_trunk, tgt_head).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=float(config.get("lr", 1e-4))
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

        with torch.no_grad():
            next_predicted = self.target_net(next_states)
            next_q = self._action_values(next_predicted)
            best_next_actions = next_q.argmax(dim=1)  # [B]
            best_next_predicted = next_predicted[
                range(batch_size), best_next_actions
            ]  # [B, num_predicted]
            target_quantiles = (
                rewards
                + (1 - dones) * self.gamma * best_next_predicted
            )

        current_all = self.q_net(states)
        current_quantiles = current_all[
            range(batch_size), actions.squeeze(1)
        ]  # [B, num_predicted]

        # Quantile Huber loss across all (current x target) pairs.
        # diff: [B, num_predicted_current, num_predicted_target]
        diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        abs_diff = diff.abs()
        huber_loss = torch.where(
            abs_diff <= self.kappa,
            0.5 * diff.pow(2),
            self.kappa * (abs_diff - 0.5 * self.kappa),
        )

        # tau is over the PREDICTOR axis (dim=1), shape [num_predicted].
        tau = self.scheme.tau.view(1, -1, 1)
        weight = torch.abs(tau - (diff.detach() < 0).float())
        loss = (weight * huber_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        self._update_target(self.q_net, self.target_net)

        return {"loss": float(loss.item())}

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
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
