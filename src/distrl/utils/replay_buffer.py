import numpy as np
import torch
from typing import Any

class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    """
    def __init__(
        self, 
        max_size: int,
        state_shape: int | tuple[int, ...], 
        action_dim: int | tuple[int, ...] = (1,), 
        device: str = "cpu"
    ) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)

        # Handle both vector and image observations
        if isinstance(state_shape, int):
            state_shape = (state_shape,)
            
        self.state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.action = np.zeros((max_size, 1), dtype=np.int64)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """
        Adds a transition to the buffer (aliased to 'push' or 'add').
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add(self, *args: Any, **kwargs: Any) -> None:
        """Alias for push."""
        self.push(*args, **kwargs)

    def sample(self, batch_size: int, device: str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch of transitions.
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        target_device = torch.device(device) if device else self.device

        return (
            torch.FloatTensor(self.state[ind]).to(target_device),
            torch.LongTensor(self.action[ind]).to(target_device),
            torch.FloatTensor(self.reward[ind]).to(target_device),
            torch.FloatTensor(self.next_state[ind]).to(target_device),
            torch.FloatTensor(self.done[ind]).to(target_device)
        )

    def __len__(self) -> int:
        return self.size
