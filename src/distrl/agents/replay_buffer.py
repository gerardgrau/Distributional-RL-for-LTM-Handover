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
        device: str = "cpu",
        state_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Experience Replay Buffer for storing transitions on CPU.
        Uses pinned memory for faster transfers to GPU/XPU during sampling.

        state_dtype lets callers opt into uint8 storage for image observations
        (~4x memory saving for Atari). The agent's trunk is responsible for
        casting/normalising uint8 inputs to float internally.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.storage_device = torch.device("cpu")
        self.target_device = torch.device(device)

        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)

        # Allocate on CPU and pin
        self.state = torch.zeros((max_size, *state_shape), dtype=state_dtype, device=self.storage_device).pin_memory()
        self.action = torch.zeros((max_size, *action_shape), dtype=torch.long, device=self.storage_device).pin_memory()
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=self.storage_device).pin_memory()
        self.next_state = torch.zeros((max_size, *state_shape), dtype=state_dtype, device=self.storage_device).pin_memory()
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=self.storage_device).pin_memory()

    def push(
        self, 
        state: np.ndarray | torch.Tensor, 
        action: int | torch.Tensor, 
        reward: float | torch.Tensor, 
        next_state: np.ndarray | torch.Tensor, 
        done: bool | torch.Tensor
    ) -> None:
        """
        Adds a transition to the buffer. Inputs are moved to CPU if necessary.
        """
        def to_cpu_tensor(x, dtype):
            if isinstance(x, torch.Tensor):
                return x.detach().to(self.storage_device, non_blocking=True).to(dtype)
            return torch.as_tensor(x, dtype=dtype, device=self.storage_device)

        self.state[self.ptr] = to_cpu_tensor(state, self.state.dtype)
        self.action[self.ptr] = to_cpu_tensor(action, self.action.dtype)
        self.reward[self.ptr] = to_cpu_tensor(reward, self.reward.dtype)
        self.next_state[self.ptr] = to_cpu_tensor(next_state, self.next_state.dtype)
        self.done[self.ptr] = to_cpu_tensor(float(done), self.done.dtype)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add(self, *args: Any, **kwargs: Any) -> None:
        """Alias for push."""
        self.push(*args, **kwargs)

    def sample(self, batch_size: int, device: str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch of transitions and moves them to the target device.
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # Use provided device or the one from __init__
        out_device = torch.device(device) if device is not None else self.target_device
        
        if out_device.type != "cpu":
            # Pinned memory + non_blocking=True is the fastest way to move to XPU/CUDA
            return (
                self.state[ind].to(out_device, non_blocking=True),
                self.action[ind].to(out_device, non_blocking=True),
                self.reward[ind].to(out_device, non_blocking=True),
                self.next_state[ind].to(out_device, non_blocking=True),
                self.done[ind].to(out_device, non_blocking=True)
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
