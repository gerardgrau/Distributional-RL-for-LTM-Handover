import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.distrl.agents.base import BaseAgent
from src.distrl.agents.networks import QHead, UnifiedQNet, build_trunk


class DQNAgent(BaseAgent):
    """Standard Deep Q-Network Agent."""

    def __init__(
        self,
        config: dict[str, Any],
        observation_space: Any,
        action_space: Any,
        device: str = "cpu",
    ) -> None:
        super().__init__(config, observation_space, action_space, device)

        action_dim = action_space.n

        def make_qnet() -> UnifiedQNet:
            trunk = build_trunk(config, observation_space)
            head = QHead(trunk.output_dim, action_dim)
            return UnifiedQNet(trunk, head).to(self.device)

        # Build q_net and target_net as independent module instances so the
        # soft-update copy is real (not a shared-reference no-op).
        self.q_net = make_qnet()
        self.target_net = make_qnet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Fused Adam: single-kernel update on accelerators. See QR-DQN
        # agent for the rationale; CPU stays on the default foreach path.
        use_fused = self.device.type != "cpu"
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=float(config.get("lr", 1e-4)),
            fused=use_fused,
        )

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.rand() < epsilon:
            return int(self.action_space.sample())
        
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        
        # Squeeze tensors for standard Q-learning
        rewards = rewards.squeeze(1)
        dones = dones.squeeze(1)

        # Compute Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute Current Q-values
        current_q_values = self.q_net(states).gather(1, actions).squeeze(1)

        # Loss and Optimization
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        self._update_target(self.q_net, self.target_net)

        # Return loss as a detached 0-d tensor; the caller materialises
        # (e.g. via torch.stack(losses).mean().item()) at episode end so
        # the training loop is not paying a D2H sync per train step.
        return {"loss": loss.detach()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
