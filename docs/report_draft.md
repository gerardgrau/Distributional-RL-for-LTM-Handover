# Project Report: Distributional RL for 5G LTM Handover Optimization

## 1. Introduction
Modern 5G and future 6G networks face significant challenges in managing mobility for high-speed users. **Lower Layer Triggered Mobility (LTM)** has emerged as a key technology to reduce handover latency and improve reliability. This project explores the application of **Quantile Regression DQN (QR-DQN)** to optimize the LTM handover process, providing risk-aware decision-making that conventional RL methods lack.

## 2. System Model
The environment simulates a multi-sector 5G deployment with 7 Base Stations and 21 sectors. We utilize a realistic dataset of 1,000 unique UE trajectories with pre-calculated channel gains.
- **State Space (88 dimensions):** Includes UE speed, tenure, serving sector (one-hot), RSRP, moving-average MCS, and SNIR.
- **Action Space:** 21 discrete actions, each representing a target sector for handover.
- **Reward Function:** A multiplicative formula considering throughput (MCS), handover frequency, ping-pong effects, and handover failures, weighted by a reliability factor.

## 3. Implementation and Optimization
The framework is built on PyTorch and Gymnasium, featuring a heavily optimized environment (~5,200 steps/s) and a high-throughput training loop (~400 steps/s).
- **Vectorization:** Entire radio physics engine implemented in NumPy matrix operations.
- **Memory Pipeline:** Replay Buffer utilizing pinned CPU memory with non-blocking device transfers.
- **Accuracy:** Fixed dimensional math in ICIC calculations to ensure 100% parity with standard 3GPP models.

[Work in Progress: Methodology and Results sections will be updated after the definitive benchmark.]
