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
        self.device = torch.device("cpu") # Always keep buffer on CPU for fast insertion

        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)
            
        # Use pinned memory for faster host-to-device transfers
        self.state = torch.zeros((max_size, *state_shape), dtype=torch.float32, device=self.device).pin_memory()
        self.action = torch.zeros((max_size, *action_shape), dtype=torch.long, device=self.device).pin_memory()
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device).pin_memory()
        self.next_state = torch.zeros((max_size, *state_shape), dtype=torch.float32, device=self.device).pin_memory()
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=self.device).pin_memory()

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
        # as_tensor without device copy is very fast on CPU
        self.state[self.ptr] = torch.as_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = torch.as_tensor(action, dtype=self.action.dtype)
        self.reward[self.ptr] = torch.as_tensor(reward, dtype=self.reward.dtype)
        self.next_state[self.ptr] = torch.as_tensor(next_state, dtype=self.next_state.dtype)
        self.done[self.ptr] = torch.as_tensor(float(done), dtype=self.done.dtype)

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
        
        target_device = torch.device(device) if device is not None else None
        
        if target_device is not None:
            # Pinned memory + non_blocking=True is the fastest way to move to XPU/CUDA
            return (
                self.state[ind].to(target_device, non_blocking=True),
                self.action[ind].to(target_device, non_blocking=True),
                self.reward[ind].to(target_device, non_blocking=True),
                self.next_state[ind].to(target_device, non_blocking=True),
                self.done[ind].to(target_device, non_blocking=True)
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
