import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Any
import os

from src.distrl.agents.base import BaseAgent
from src.distrl.models.networks import MLPTrunk, QuantileHead, UnifiedQNet

class QRDQNAgent(BaseAgent):
    """
    Quantile Regression DQN (QR-DQN) Agent.
    Models the return distribution using a fixed set of quantiles.
    """

    def __init__(
        self, 
        config: dict[str, Any], 
        observation_space: Any, 
        action_space: Any, 
        device: str = "cpu"
    ) -> None:
        super().__init__(config, observation_space, action_space, device)

        input_dim = observation_space.shape[0]
        action_dim = action_space.n
        
        self.num_quantiles = int(config.get("num_quantiles", 50))
        self.kappa = float(config.get("kappa", 1.0)) # Huber loss threshold
        
        trunk = MLPTrunk(input_dim, config.get("hidden_dims", [128, 128]))
        head = QuantileHead(trunk.output_dim, action_dim, self.num_quantiles)
        
        self.q_net = torch.compile(UnifiedQNet(trunk, head).to(self.device))
        self.target_net = UnifiedQNet(trunk, head).to(self.device)
        
        # Correctly load state_dict even if q_net is compiled
        orig_net = self.q_net._orig_mod if hasattr(self.q_net, "_orig_mod") else self.q_net
        self.target_net.load_state_dict(orig_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=float(config.get("lr", 1e-4)))
        
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.target_update_freq = int(config.get("target_update_freq", 1000))
        self.update_counter = 0
        
        self.cumulative_probabilities = torch.arange(
            0.5, self.num_quantiles, 1.0, device=self.device
        ) / self.num_quantiles

    def select_action(self, state: np.ndarray | torch.Tensor, epsilon: float = 0.0) -> int:
        # TODO: per ara fem epsilon-greedy, canviar?
        if np.random.random() < epsilon:
            return int(self.action_space.sample())
        
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            quantiles = self.q_net(state)
            q_values = quantiles.mean(dim=2)
            # TODO: provar amb la mitja dels X quantils més petits
        
        return int(q_values.argmax(dim=1).item())

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        batch_size = states.size(0)

        # 1. Current Quantiles
        curr_quantiles = self.q_net(states)
        curr_quantiles = curr_quantiles.gather(
            1, actions.long().unsqueeze(-1).expand(batch_size, 1, self.num_quantiles)
        ).squeeze(1) 

        # 2. Target Quantiles
        with torch.no_grad():
            next_quantiles = self.target_net(next_states) 
            next_actions = next_quantiles.mean(dim=2).argmax(dim=1, keepdim=True) 
            next_quantiles = next_quantiles.gather(
                1, next_actions.unsqueeze(-1).expand(batch_size, 1, self.num_quantiles)
            ).squeeze(1) 
            target_quantiles = rewards + (1 - dones) * self.gamma * next_quantiles 

        # 3. Quantile Huber Loss
        diff = target_quantiles.unsqueeze(2) - curr_quantiles.unsqueeze(1)
        abs_diff = diff.abs()
        huber_loss = torch.where(abs_diff <= self.kappa, 0.5 * diff.pow(2), self.kappa * (abs_diff - 0.5 * self.kappa))
        tau = self.cumulative_probabilities.view(1, 1, -1)
        weight = torch.abs(tau - (diff.detach() < 0).float())
        loss = (weight * huber_loss).mean()

        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # 5. Update Target
        self.update_counter += 1
        self._update_target()

        return {"loss": loss.item(), "mean_q": curr_quantiles.mean().item()}

    def _update_target(self) -> None:
        if self.tau < 1.0:
            for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)
        elif self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        orig_net = self.q_net._orig_mod if hasattr(self.q_net, "_orig_mod") else self.q_net
        torch.save({
            'q_net_state_dict': orig_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        orig_net = self.q_net._orig_mod if hasattr(self.q_net, "_orig_mod") else self.q_net
        orig_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
