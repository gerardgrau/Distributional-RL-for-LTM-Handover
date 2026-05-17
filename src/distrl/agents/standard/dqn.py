import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Any
import os

from src.distrl.agents.base import BaseAgent
from src.distrl.agents.networks import CNNTrunk, MLPTrunk, QHead, UnifiedQNet


def _make_trunk(config: dict[str, Any], observation_space: Any) -> Any:
    if config.get("trunk_type", "mlp") == "cnn":
        return CNNTrunk(
            in_channels=int(observation_space.shape[0]),
            output_dim=int(config.get("cnn_feature_dim", 512)),
        )
    return MLPTrunk(
        int(observation_space.shape[0]),
        config.get("hidden_dims", [128, 128]),
    )


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

        # Build q_net and target_net with their own independent modules so
        # parameter updates don't accidentally propagate via shared refs.
        trunk = _make_trunk(config, observation_space)
        head = QHead(trunk.output_dim, action_dim)
        self.q_net = UnifiedQNet(trunk, head).to(self.device)

        tgt_trunk = _make_trunk(config, observation_space)
        tgt_head = QHead(tgt_trunk.output_dim, action_dim)
        self.target_net = UnifiedQNet(tgt_trunk, tgt_head).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=float(config.get("lr", 1e-4))
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

        return {"loss": float(loss.item())}

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
