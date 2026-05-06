# Project Context: Distributional RL for LTM Handover

> **IMPORTANT**: This file must be ALWAYS up to date. It must be edited and optimized whenever a change occurs or something relevant is added to the project.

This document serves as the ground truth for the **Distributional RL for LTM Handover** project. It outlines the modular architecture, physical constraints, reward structures, and engineering standards to ensure research consistency.

---

## 1. Project Overview
The goal of this project is to optimize **5G Lower Layer Triggered Mobility (LTM)** handovers using standard and distributional Reinforcement Learning (DQN vs. QRDQN). The environment simulates a realistic 5G multi-sector deployment using channel gain data from 1,000 unique UE trajectories.

---

## 2. Core Environment Constants
- **Simulation Duration**: 300s episodes (sampled at 10ms/100ms resolution).
- **Deployment**: 7 Base Stations (BS), 21 Sectors (NBS=21).
- **Tx Power**: 25 dBm (Aligned with CMAB for LTM HO paper).
- **State Space (88 Dims)**: 
  - [Speed(1), Tenure(1), One-Hot Serving(21), RSRP(21), Moving-Average MCS(21), Moving-Average SNIR(21), X(1), Y(1)].
- **Reward Formula**: Multiplicative Ainna-Reward.
  - $R = R_{thr} \cdot (\alpha_{HO}^{ind_{HO}} \cdot \alpha_{PP}^{ind_{PP}} \cdot \alpha_{HOF}^{ind_{HOF}}) \cdot \text{reliability\_factor}$
  - Alphas: $\alpha_{HOF}=0.1, \alpha_{HO}=0.8, \alpha_{PP}=0.9$.

---

## 3. Engineering Standards
- **Architecture**: Modular structure in `src/distrl/` isolating `envs`, `agents`, `models`, and `utils`.
- **Frameworks**: PyTorch 2.0+ (utilizing `torch.compile`), Gymnasium, NumPy, SciPy.
- **Hardware Sympathy**: Environment is heavily optimized for **CPU-bound NumPy vectorization**. CPU mode is 2.5x faster than XPU for this workload. Default execution device is set via the `--device` CLI flag (defaults to `cpu`).
- **Type Safety**: Fully modernized with Python 3.10+ type hints (`|` union, `dict[str, Any]`, etc.).

---

## 4. Benchmark & Visualization Results
- **Success Case**: Both agents successfully navigate the 300s journey with the 88-dim state.
- **DQN Observation**: Standard DQN shows high variance and instability (e.g., higher HOF/PP frequency) in noisy environments.
- **QRDQN Observation**: Superior stability by modeling the return distribution. Risk analysis via quantile plots shows narrower distributions in stable regions and wider, multi-modal spreads during handover transitions.
- **Risk-Averse Policy**: Supports **Conditional Value-at-Risk (CVaR)** selection. By choosing actions based on the mean of the bottom $k$ quantiles, the agent can prioritize reliability in high-interference zones.
- **Evaluation Protocol**: Implements a formal **Train/Test Split** (80/20). Agents are trained on 800 trajectories and definitively evaluated on 200 unseen trajectories with frozen weights and $\epsilon=0$.
- **8 Metrics Suite**: Automated calculation of Capacity, RLF, HO Rate, Ping-Pong, Reliability, Cell Preparation, Resource Reservation, and HOF Rate.
- **Hardcore Benchmark (2026-04-23)**: 500-episode multi-seed run completed. Artifacts in `results/benchmarks/benchmark_2026-04-23_20-30-46/`.
- **Optimization Artifacts**: The environment is now optimized to ~5,200 steps/s utilizing trajectory caching and vectorized HOF logic.

---

## 5. Workflow & Planning
- **Agentic Task Planning**: ALL agentic task planning and track management must go through the **Conductor** extension. Refer to `conductor/GEMINI.md` for specific planning protocols, track management rules, and the universal file resolution protocol.
- **Track Management**: No major changes should be implemented without an active Conductor track and an approved implementation plan (`plan.md`).

---

## 6. Project Conventions & Rules
- **Distributed `.gitignore`**: To keep the repository clean while tracking folder structures, use local `.gitignore` files in subdirectories (e.g., `results/models/`, `results/benchmarks/`) with local rules (prefix `/`, e.g., `/*.pth`).
- **Maintenance Policy**: This file (`GEMINI.md`) is the ground truth context and must be updated with any change to the state space, reward formula, or architecture.
- **Python Path**: Always export the `src` directory to ensure correct package resolution:
  `export PYTHONPATH=$PYTHONPATH:$(pwd)/src`
