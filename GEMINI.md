# Project Context: Distributional RL for LTM Handover

> **IMPORTANT**: This file must be ALWAYS up to date. It must be edited and optimized whenever a change occurs or something relevant is added to the project.

This document serves as the persistent context for the research project, detailing the architecture, scientific formulations, and findings discovered during development.

---

## 1. Scientific Objectives
*   **Goal**: Compare **Standard RL (DQN)** vs. **Distributional RL (QR-DQN)** for optimizing 5G/6G Handover (HO) decisions.
*   **Target Environment**: 5G Lower Layer Triggered Mobility (LTM), modeled on a 7-Base Station hexagonal urban layout.
*   **Hardware**: Optimized for **Intel XPU (ARC GPU)** acceleration using specialized PyTorch builds.

---

## 2. Environment Design (`src/distrl/envs/`)

### **A. Markovian State Space (88 Dimensions)**
We transitioned from a simple RSRP-only vector to a rich, Markovian context to enable the agent to see the "signal velocity" and mobility trends:
1.  **UE Mobility**: Normalized Speed (1)
2.  **HO Dynamics**: Serving Tenure (1 step = 100ms) and One-Hot Serving Sector ID (21).
3.  **Signal Landscape**: Current L3-filtered RSRP for all 21 sectors.
4.  **Oracle Network Stability**: Moving $p$-sample average of **MCS** (21) and **SNIR** (21) for *all* sectors.
5.  **Spatial Context**: Normalized UE X/Y coordinates (2).

### **B. Multiplicative Reward**
Aligned with the research paper, we use a multiplicative formula to balance throughput against stability:
$$r = \bar{r}_{\mathrm{thr}} \cdot \frac{\alpha_{\mathrm{HO}}^{\mathbb{I}_{\mathrm{HO}}} \cdot \alpha_{\mathrm{PP}}^{\mathbb{I}_{\mathrm{PP}}} \cdot \alpha_{\mathrm{HOF}}^{\mathbb{I}_{\mathrm{HOF}}}}{1 + \exp(2(N_{OOS} - 2))}$$
*   **Coefficients**: $\alpha_{\mathrm{HOF}}=0.1$, $\alpha_{\mathrm{HO}}=0.8$, $\alpha_{\mathrm{PP}}=0.9$.
*   **Reliability**: A reverse sigmoid based on $N_{OOS}$ (Out-of-Sync) ситуаtions provides a soft transition before failure.

### **C. Hybrid Decision Strategy**
*   **When Connected**: RL Agent chooses the action (Stay or HO).
*   **When Disconnected (Recovery)**: Environment uses a **Greedy Baseline** (Standard Cell Search) to reconnect to the strongest tower, isolating the learning task to handover optimization.

---

## 3. Core Architecture (`src/distrl/`)
- **Modular Packages**: De-coupled Learning Logic (`agents/`) from Neural Network Architecture (`models/`).
- **Unified Net**: All agents share a common `MLPTrunk` feature extractor for a 100% fair comparison.
- **Type Safety**: Fully modernized with Python 3.10+ type hints (`|` unions, built-in collections).

---

## 4. Benchmark & Visualization Results
- **Success Case**: Both agents have proven they can successfully navigate the full 300s journey using the new 88-dim state.
- **DQN Observation**: Standard DQN shows high variance and instability in noisy 5G environments.
- **QRDQN Observation**: Demonstrates superior stability by explicitly modeling the return distribution (risk).
- **Automation**: `src/benchmark.py` automatically generates learning curves, efficiency plots, and Quantile Distribution Insights for every run.

---

## 5. Directory Convention
All artifacts are saved independently into structured subfolders under `results/benchmarks/benchmark_YYYY-MM-DD_HH-MM-SS/`:
- `output/`: Raw CSV logs.
- `models/`: Trained weights (`_best.pth`).
- `figures/`: Learning curves and Mobility animations.
