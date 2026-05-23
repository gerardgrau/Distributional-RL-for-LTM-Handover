from abc import ABC, abstractmethod
from typing import Any
import torch
import numpy as np

class BaseAgent(ABC):
    """
    Abstract Base Class for all RL Agents in the framework.
    """

    def __init__(
        self, 
        config: dict[str, Any], 
        observation_space: Any, 
        action_space: Any, 
        device: str = "cpu"
    ) -> None:
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        
        # Common RL Parameters
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.target_update_freq = int(config.get("target_update_freq", 1000))
        self.update_counter = 0

    def _update_target(self, q_net: torch.nn.Module, target_net: torch.nn.Module) -> None:
        """Soft (tau<1) or hard (tau==1, every `target_update_freq`) update."""
        if self.tau < 1.0:
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(target_net.parameters()),
                    list(q_net.parameters()),
                    self.tau,
                )
        elif self.update_counter % self.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

    @abstractmethod
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Selects an action based on the current state and exploration strategy.
        
        Args:
            state: The current environment state.
            epsilon: The exploration rate (for epsilon-greedy policies).
        """
        pass

    @abstractmethod
    def train_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        """
        Performs a single optimization/training step using a batch of experiences.
        
        Args:
            batch: A batch of transitions (s, a, r, s', done).
        
        Returns:
            dict: A dictionary containing training metrics (e.g., loss).
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the agent's model checkpoints and internal state to the specified path.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the agent's model checkpoints and internal state from the specified path.
        """
        pass

    def reset(self) -> None:
        """
        Resets the agent's internal state between episodes (if any).
        """
        pass
