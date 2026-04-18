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

        # Get dimensions
        input_dim = observation_space.shape[0]
        action_dim = action_space.n
        
        # QRDQN specific parameters
        self.num_quantiles = config.get("num_quantiles", 50)
        self.kappa = config.get("kappa", 1.0) # Huber loss threshold
        
        # Initialize Networks (Trunk + QuantileHead)
        hidden_dims = config.get("hidden_dims", [128, 128])
        trunk = MLPTrunk(input_dim, hidden_dims)
        head = QuantileHead(trunk.output_dim, action_dim, self.num_quantiles)
        
        self.q_net = UnifiedQNet(trunk, head).to(self.device)
        self.target_net = UnifiedQNet(trunk, head).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        lr = config.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.target_update_freq = config.get("target_update_freq", 1000)
        self.update_counter = 0
        
        # Cumulative probabilities for the midpoints of the quantiles
        self.cumulative_probabilities = torch.arange(
            0.5, self.num_quantiles, 1.0, device=self.device
        ) / self.num_quantiles

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Selects an action based on the mean of the quantile distribution.
        """
        if np.random.random() < epsilon:
            return int(self.action_space.sample())
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # quantiles shape: [1, action_dim, num_quantiles]
            quantiles = self.q_net(state_tensor)
            # mean Q-values shape: [1, action_dim]
            q_values = quantiles.mean(dim=2)
        
        return int(q_values.argmax().item())

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        """
        Performs a single QRDQN optimization step using Quantile Huber Loss.
        """
        states, actions, rewards, next_states, dones = batch
        batch_size = states.size(0)

        # 1. Current Quantiles
        # Output: [batch_size, action_dim, num_quantiles]
        # Gather by action: [batch_size, 1, num_quantiles]
        curr_quantiles = self.q_net(states)
        curr_quantiles = curr_quantiles.gather(
            1, actions.long().unsqueeze(-1).expand(batch_size, 1, self.num_quantiles)
        ).squeeze(1) # [batch_size, num_quantiles]

        # 2. Target Quantiles
        with torch.no_grad():
            # Get next quantiles from target net
            next_quantiles = self.target_net(next_states) # [batch_size, action_dim, num_quantiles]
            
            # Select best action based on the mean of the target quantiles
            next_actions = next_quantiles.mean(dim=2).argmax(dim=1, keepdim=True) # [batch_size, 1]
            
            # Gather quantiles for the best actions
            next_quantiles = next_quantiles.gather(
                1, next_actions.unsqueeze(-1).expand(batch_size, 1, self.num_quantiles)
            ).squeeze(1) # [batch_size, num_quantiles]
            
            # Compute target distribution (Bellman operator)
            target_quantiles = rewards + (1 - dones) * self.gamma * next_quantiles # [batch_size, num_quantiles]

        # 3. Quantile Huber Loss
        # pairwise_diff shape: [batch_size, num_quantiles (target), num_quantiles (curr)]
        # We broadcast: target[b, i, 1] - curr[b, 1, j]
        diff = target_quantiles.unsqueeze(2) - curr_quantiles.unsqueeze(1)
        
        # Huber loss component
        abs_diff = diff.abs()
        huber_loss = torch.where(abs_diff <= self.kappa, 0.5 * diff.pow(2), self.kappa * (abs_diff - 0.5 * self.kappa))
        
        # Quantile penalty component
        # loss = |tau - I(diff < 0)| * huber_loss
        tau = self.cumulative_probabilities.view(1, 1, -1)
        weight = torch.abs(tau - (diff.detach() < 0).float())
        loss = (weight * huber_loss).mean()

        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()
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
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
