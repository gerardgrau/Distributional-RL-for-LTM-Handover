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

        # Pinning only buys speed when the target device is a non-CPU
        # accelerator (cuda/xpu) AND the total pinned tensors stay under the
        # OS memlock limit. Skip it for CPU targets — the copy is in-process
        # so pinning is wasted, and large uint8 image buffers can segfault
        # the process if they exceed RLIMIT_MEMLOCK.
        pin = self.target_device.type != "cpu"

        def alloc(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            t = torch.zeros(shape, dtype=dtype, device=self.storage_device)
            return t.pin_memory() if pin else t

        self.state = alloc((max_size, *state_shape), state_dtype)
        self.action = alloc((max_size, *action_shape), torch.long)
        self.reward = alloc((max_size, 1), torch.float32)
        self.next_state = alloc((max_size, *state_shape), state_dtype)
        self.done = alloc((max_size, 1), torch.float32)

    def push(
        self,
        state: np.ndarray | torch.Tensor,
        action: int | torch.Tensor,
        reward: float | torch.Tensor,
        next_state: np.ndarray | torch.Tensor,
        done: bool | torch.Tensor,
    ) -> None:
        """Adds a transition to the buffer.

        Inlined fast path: numpy state/next_state get assigned directly to
        a preallocated tensor row (PyTorch handles the dtype-aware copy via
        the buffer protocol), and Python scalars get written into the
        single-element action/reward/done slots. The previous version
        wrapped each field in a `to_cpu_tensor(...)` helper that allocated
        a fresh torch.Tensor per push — measurable per-step overhead.
        """
        p = self.ptr
        if isinstance(state, torch.Tensor):
            # Tensor input — may be on a non-CPU device.
            self.state[p] = state.detach().to(self.storage_device, non_blocking=True)
            self.next_state[p] = next_state.detach().to(self.storage_device, non_blocking=True)
        else:
            # numpy input: from_numpy shares memory (zero-copy wrap), and
            # the assignment then copies into the preallocated buffer row.
            # PyTorch will not accept a raw ndarray here, so the wrap is
            # required even though it looks redundant.
            self.state[p] = torch.from_numpy(state)
            self.next_state[p] = torch.from_numpy(next_state)

        if isinstance(action, torch.Tensor):
            self.action[p, 0] = int(action.item())
        else:
            self.action[p, 0] = action

        if isinstance(reward, torch.Tensor):
            self.reward[p, 0] = float(reward.item())
        else:
            self.reward[p, 0] = reward

        self.done[p, 0] = float(done)

        self.ptr = (p + 1) % self.max_size
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
