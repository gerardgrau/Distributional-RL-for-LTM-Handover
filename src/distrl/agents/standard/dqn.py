import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Any
import os

from src.distrl.agents.base import BaseAgent
from src.distrl.models.networks import MLPTrunk, QHead, UnifiedQNet

class DQNAgent(BaseAgent):
    """
    Standard Deep Q-Network (DQN) Agent.
    """

    def __init__(
        self, 
        config: dict[str, Any], 
        observation_space: Any, 
        action_space: Any, 
        device: str = "cpu"
    ) -> None:
        super().__init__(config, observation_space, action_space, device)

        # Get dimensions from spaces
        # Assuming observation_space has a .shape and action_space has a .n (Discrete)
        input_dim = observation_space.shape[0]
        action_dim = action_space.n

        # Initialize Networks (Trunk + QHead)
        hidden_dims = config.get("hidden_dims", [128, 128])
        trunk = MLPTrunk(input_dim, hidden_dims)
        head = QHead(trunk.output_dim, action_dim)
        
        self.q_net = UnifiedQNet(trunk, head).to(self.device)
        self.target_net = UnifiedQNet(trunk, head).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        lr = config.get("lr", 1e-4)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005) # For soft update
        self.target_update_freq = config.get("target_update_freq", 1000)
        self.update_counter = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Selects an action using epsilon-greedy policy.
        """
        if np.random.random() < epsilon:
            return int(self.action_space.sample())
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        
        return int(q_values.argmax().item())

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        """
        Performs a single DQN optimization step.
        Batch: (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = batch
        
        # Get current Q values for selected actions
        curr_q = self.q_net(states).gather(1, actions.long())
        
        # Get target Q values (Double DQN style can be added later, using vanilla DQN for now)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Compute Loss (MSE)
        loss = F.mse_loss(curr_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update Target Network
        self.update_counter += 1
        self._update_target()
        
        return {"loss": loss.item(), "mean_q": curr_q.mean().item()}

    def _update_target(self) -> None:
        """
        Updates the target network using soft or hard updates.
        """
        if self.tau < 1.0:
            # Soft Update
            for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)
        elif self.update_counter % self.target_update_freq == 0:
            # Hard Update
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        """
        Saves the agent's model state.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """
        Loads the agent's model state.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
