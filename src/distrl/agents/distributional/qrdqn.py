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
    Quantile Regression DQN Agent.
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
        
        self.q_net = UnifiedQNet(trunk, head).to(self.device)
        self.target_net = UnifiedQNet(trunk, head).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=float(config.get("lr", 1e-4)))
        
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.target_update_freq = int(config.get("target_update_freq", 1000))
        self.update_counter = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.rand() < epsilon:
            return int(self.action_space.sample())
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # [1, action_dim, num_quantiles]
            quantiles = self.q_net(state_t)
            # q_values = mean across quantiles: [1, action_dim]
            q_values = quantiles.mean(dim=2)
            return int(q_values.argmax(dim=1).item())

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        batch_size = states.size(0)

        # 1. Compute target quantiles
        with torch.no_grad():
            # [batch_size, action_dim, num_quantiles]
            next_quantiles = self.target_net(next_states)
            # Select best action using mean: [batch_size, action_dim]
            next_q_values = next_quantiles.mean(dim=2)
            best_next_actions = next_q_values.argmax(dim=1) # [batch_size]
            
            # Extract quantiles for best actions: [batch_size, num_quantiles]
            best_next_quantiles = next_quantiles[range(batch_size), best_next_actions]
            
            # T_theta = r + gamma * theta_next
            # rewards/dones are [batch_size, 1], broadcast over [batch_size, num_quantiles]
            target_quantiles = rewards + (1 - dones) * self.gamma * best_next_quantiles

        # 2. Compute current quantiles
        # [batch_size, action_dim, num_quantiles]
        current_all_quantiles = self.q_net(states)
        # [batch_size, num_quantiles]
        # actions is (batch_size, 1), squeeze to (batch_size,) for indexing
        current_quantiles = current_all_quantiles[range(batch_size), actions.squeeze(1)]

        # 3. Compute Quantile Huber Loss
        # Pairwise differences: [batch_size, num_quantiles (current), num_quantiles (target)]
        diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        
        # Huber loss
        abs_diff = diff.abs()
        huber_loss = torch.where(abs_diff <= self.kappa, 
                                 0.5 * diff.pow(2), 
                                 self.kappa * (abs_diff - 0.5 * self.kappa))
        
        # Quantile weights
        tau = torch.arange(0.5 / self.num_quantiles, 1, 1 / self.num_quantiles, device=self.device)
        # weight = |tau - I(diff < 0)|
        weight = torch.abs(tau.unsqueeze(1) - (diff.detach() < 0).float())
        
        loss = (weight * huber_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        self._update_target()

        return {"loss": float(loss.item())}

    def _update_target(self) -> None:
        if self.tau < 1.0:
            for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)
        elif self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

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
