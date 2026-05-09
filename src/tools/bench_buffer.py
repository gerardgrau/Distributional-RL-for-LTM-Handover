import numpy as np
import torch
import time
from src.distrl.utils.replay_buffer import ReplayBuffer

def benchmark_buffer_insertion(device_str, size=10000):
    device = torch.device(device_str)
    buffer = ReplayBuffer(size, (88,), (1,), device=device_str)
    
    # Force storage on device for the comparison test
    if device_str != "cpu":
        buffer.state = buffer.state.to(device)
        buffer.action = buffer.action.to(device)
        buffer.reward = buffer.reward.to(device)
        buffer.next_state = buffer.next_state.to(device)
        buffer.done = buffer.done.to(device)
        buffer.storage_device = device
    
    state = np.random.randn(88).astype(np.float32)
    action = 1
    reward = 1.0
    next_state = np.random.randn(88).astype(np.float32)
    done = False
    
    start = time.time()
    for _ in range(size):
        buffer.push(state, action, reward, next_state, done)
    end = time.time()
    
    print(f"Device: {device_str} | Insertion Time for {size} steps: {end-start:.4f}s")

if __name__ == "__main__":
    benchmark_buffer_insertion("cpu")
    if torch.cuda.is_available():
        benchmark_buffer_insertion("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        benchmark_buffer_insertion("xpu")
