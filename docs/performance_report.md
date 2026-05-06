# Performance Report - Distributional RL for LTM Handover

## 1. Environment Optimization Summary
Through a series of advanced performance optimizations, the `LTMEnv` was accelerated from ~220 steps/s to over 5,000 steps/s (raw environment speed).

### Key Optimizations:
1.  **Vectorized Oracle Physics:** Replaced 21x scalar loops with NumPy matrix math.
2.  **Circular Buffers:** Removed `np.array(list)` casting overhead in history management.
3.  **Episode Pre-calculation:** Shifted deterministic radio physics from `step()` to a single 2D matrix pass in `reset()`.
4.  **Global UE Caching:** Implemented in-RAM caching of pre-calculated matrices to make subsequent episode resets instantaneous (~0.1ms).
5.  **O(1) Observations:** Converted moving average calculations to running sums, removing `np.mean` from the observation loop.
6.  **Vectorized HOF:** Pre-calculated Handover Failure probabilities for all potential actions and time steps.
7.  **Expanded Global Cache:** Increased cache limit to **1,000 users** (~10.9 GB total) to fully utilize 32GB RAM environments.

## 2. Hardware Benchmarking (CPU vs. XPU)
Benchmarking was performed using the faithful production training workload (QRDQN agent, training every step) for 60 episodes per device.

| Device | Total Time (60 eps) | Steps / Second |
| :--- | :--- | :--- |
| **CPU** | 1,930.43s | **93.24** |
| **XPU (iGPU)** | 2,186.20s | 82.33 |

### Analysis:
*   **CPU dominance:** The CPU is **13% faster** than the Intel iGPU for the standard QRDQN training task.
*   **Small Model Overhead:** The neural networks are relatively small (128x128 MLPs). The overhead of copying tensors to the iGPU memory space every step (Observation -> GPU, Action -> CPU) outweighs any parallel processing gains.
*   **Vectorization Sympathy:** NumPy is highly optimized for CPU SIMD (AVX-512/AVX2). Since the environment is now purely NumPy-based, it stays in the CPU cache, avoiding the PCIE/Bus bottleneck.

## 3. torch.compile Analysis
We performed a comparison between standard Eager Execution and `torch.compile` (Inductor backend) on CPU.

| Mode | Total Execution Time | Impact |
| :--- | :--- | :--- |
| **Standard (Eager)** | 197s | Baseline |
| **torch.compile** | 218s | **~10% Slower** |

### Findings:
1.  **Compilation Overhead**: For small MLP models (128x128), the initial JIT compilation time (~2-4 mins) far exceeds any execution speedup.
2.  **Incompatibility**: `torch.compile` triggered multiple `Backend compiler exceptions` and graph breaks in the distributional (QRDQN) head due to symbolic inference limits on CPU.
3.  **Stability**: Eager mode is 100% stable and provides immediate startup.

**Conclusion**: `torch.compile` has been disabled in the production agents.
