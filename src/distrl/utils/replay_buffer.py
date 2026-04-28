import numpy as np
import torch
from typing import Any

class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions using PyTorch Tensors.
    """
    def __init__(
        self, 
        max_size: int,
        state_shape: int | tuple[int, ...], 
        action_shape: int | tuple[int, ...] = (1,), 
        device: str = "cpu"
    ) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)

        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)
            
        self.state = torch.zeros((max_size, *state_shape), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((max_size, *action_shape), dtype=torch.long, device=self.device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)
        self.next_state = torch.zeros((max_size, *state_shape), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device)

    def push(
        self, 
        state: np.ndarray | torch.Tensor, 
        action: int | torch.Tensor, 
        reward: float | torch.Tensor, 
        next_state: np.ndarray | torch.Tensor, 
        done: bool | torch.Tensor
    ) -> None:
        """
        Adds a transition to the buffer.
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
        
        if device is not None and torch.device(device) != self.device:
            target_device = torch.device(device)
            return (
                self.state[ind].to(target_device),
                self.action[ind].to(target_device),
                self.reward[ind].to(target_device),
                self.next_state[ind].to(target_device),
                self.done[ind].to(target_device)
            )
        
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )

    def __len__(self) -> int:
        return self.size
