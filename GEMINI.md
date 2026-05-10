# Project Context: Distributional RL for LTM Handover

> **IMPORTANT**: This file must be ALWAYS up to date. It must be edited and optimized whenever a change occurs or something relevant is added to the project.

This document serves as the ground truth for the **Distributional RL for LTM Handover** project. It outlines the modular architecture, physical constraints, reward structures, and engineering standards to ensure research consistency.

---

## 1. Project Overview
The goal of this project is to optimize **5G Lower Layer Triggered Mobility (LTM)** handovers using standard and distributional Reinforcement Learning (DQN vs. QRDQN). The environment simulates a realistic 5G multi-sector deployment using channel gain data from 1,000 unique UE trajectories.

---

## 2. Core Environment Constants
- **Simulation Duration**: 300s episodes (sampled at 10ms resolution). RL steps at 100ms.
- **HO Parity**: 50ms total delay from decision to new cell (20ms on old cell + 30ms interruption).
- **Deployment**: 7 Base Stations (BS), 21 Sectors (NBS=21).
- **Tx Power**: 25 dBm. **Bandwidth**: 200 MHz.
- **State Space (88 Dims)**: 
  - [Speed(1), Tenure(1), One-Hot Serving(21), RSRP(21), Moving-Average MCS(21), Moving-Average SNIR(21), X(1), Y(1)].
- **Reward Formula**: Multiplicative Ainna-Reward.
  - $R = R_{thr} \cdot (\alpha_{HO}^{ind_{HO}} \cdot \alpha_{PP}^{ind_{PP}} \cdot \alpha_{HOF}^{ind_{HOF}}) \cdot \text{reliability\_factor}$
  - Alphas: $\alpha_{HOF}=0.1, \alpha_{HO}=0.8, \alpha_{PP}=0.9$.
  - Reliability Factor: Soft reverse-sigmoid based on Out-Of-Sync counter ($N_{OOS}$).
- **Research Protocol**: **Full Dataset Protocol**. Agents are trained on all 1,000 trajectories and definitively evaluated on the same 1,000 trajectories with frozen weights and $\epsilon=0$.
- **Simulation Parity**: Environment calibrated to match paper metrics (Capacity ~3.3-3.7, Prep Rate ~780, HO Rate ~11). Reliability (~98.6%) is higher than paper (~95%) due to lack of explicit blockage models in current channel files.

---

## 3. Engineering Standards
- **Architecture**: Modular structure in `src/distrl/` isolating `envs`, `agents`, `models`, and `utils`.
- **Frameworks**: PyTorch 2.0+, Gymnasium, NumPy, SciPy.
- **Performance**: Environment is highly optimized for **CPU-bound NumPy vectorization** (~5,250 steps/s, 23x speedup).
  - **Dataset Pre-computation**: 1,000-user trajectory physics pre-calculated and stored as optimized `.npz` binaries (~0GB permanent RAM footprint).
  - **O(1) Complexity**: Observation moving averages use running sums.
- **Hardware Selection**: Use **XPU** (Intel iGPU) or **CPU**. With the latest **Dataset Pre-computation** logic, the **XPU (~319 steps/s) is ~41% faster than the CPU (~227 steps/s)** as the CPU is no longer burdened by radio physics math. Select via `--device xpu` (recommended) or `--device cpu` CLI flags.
- **Type Safety**: Fully modernized with Python 3.10+ type hints (`|` union, `dict[str, Any]`, etc.).

---

## 4. Evaluation & Reporting
- **8 Metrics Suite**: Automated calculation of Capacity, RLF, HO Rate, Ping-Pong, Reliability, Cell Preparation, Resource Reservation, and HOF Rate.
- **Dual Reporting**: Every evaluation generates a **CSV Summary** (Metric, Mean, Std) and a **Raw CSV** (per-episode data).
- **Master Comparisons**: Final benchmarks are consolidated in `results/final_metrics/` using `src/tools/generate_final_plots.py`, comparing ours vs. paper results (LTM, LMMSE, CMAB).
- **Visualization (MP4)**: Agent behavior dashboard generates MP4 videos (preferred over GIF for playback control) located in `results/benchmarks/bmk_.../animations/`.
  - **Standard**: 7-color hexagonal layout, segmented RSRP opacity (100% when active), and 'x' sector change markers.
- **Benchmark Naming**: Follow the convention `bmk_YYYY-MM-DD_num_description`.

---

## 5. Workflow & Planning
- **Agentic Task Planning**: ALL agentic task planning and track management must go through the **Conductor** extension.
- **Reproducibility**: `src/main.py` copies the active `config.yaml` into every benchmark folder.
- **Definitive Benchmark**: A large-scale 12-hour benchmark (2,000 episodes) for DQN and QR-DQN has been completed.
- **Ablation Study**: A 4-point ablation study on quantile counts (N=[10, 50, 100, 200]) is currently on standby.

---

## 6. Project Conventions & Rules
- **Configuration**: All YAML files must reside in the `configs/` directory.
- **Python Path**: Always export the `src` directory to ensure correct package resolution:
  `export PYTHONPATH=$PYTHONPATH:$(pwd)/src`
- **Maintenance Policy**: This file (`GEMINI.md`) is the ground truth context and must be updated with any change to the state space, reward formula, or architecture.
- **Physics Engine**: All radio physics and ICIC math are consolidated in `src/distrl/envs/physics.py` to ensure consistency between the Gymnasium environment and post-processing metrics.
