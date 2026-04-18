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
