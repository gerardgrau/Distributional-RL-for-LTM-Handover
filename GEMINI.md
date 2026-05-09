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
- **Research Protocol**: **Full Dataset Protocol**. Agents are trained on all 1,000 trajectories and definitively evaluated on the same 1,000 trajectories with frozen weights and $\epsilon=0$.

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
- **Branch Strategy**: Use feature branches (e.g., `feature/codebase-refactor`) and Pull Requests for all modifications.
- **Benchmark Naming**: Follow the convention `bmk_YYYY-MM-DD_num_description`.

---

## 5. Workflow & Planning
- **Agentic Task Planning**: ALL agentic task planning and track management must go through the **Conductor** extension. Refer to `conductor/GEMINI.md` for specific planning protocols and track management rules.
- **Reproducibility**: `src/main.py` (previously `experiment.py`) copies the active `config.yaml` into every benchmark folder.

---

## 6. Project Conventions & Rules
- **Configuration**: All YAML files must reside in the `configs/` directory.
- **Python Path**: Always export the `src` directory to ensure correct package resolution:
  `export PYTHONPATH=$PYTHONPATH:$(pwd)/src`
- **Maintenance Policy**: This file (`GEMINI.md`) is the ground truth context and must be updated with any change to the state space, reward formula, or architecture.
