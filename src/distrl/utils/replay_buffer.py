import numpy as np
import torch

class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    """
    def __init__(
        self, 
        state_shape: int | tuple[int, ...], 
        action_dim: int, 
        max_size: int = 100000, 
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

    def add(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
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

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch of transitions.
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def __len__(self) -> int:
        return self.size
