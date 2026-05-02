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

        input_dim = observation_space.shape[0]
        action_dim = action_space.n

        trunk = MLPTrunk(input_dim, config.get("hidden_dims", [128, 128]))
        head = QHead(trunk.output_dim, action_dim)
        
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

    def select_action(self, state: np.ndarray | torch.Tensor, epsilon: float = 0.0) -> int:
        # TODO: per ara fem epsilon-greedy, canviar?
        if np.random.random() < epsilon:
            return int(self.action_space.sample())
        
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            q_values = self.q_net(state)
        
        return int(q_values.argmax(dim=1).item())

    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        
        curr_q = self.q_net(states).gather(1, actions.long())
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        loss = F.mse_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_counter += 1
        self._update_target()
        
        return {"loss": loss.item(), "mean_q": curr_q.mean().item()}

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
