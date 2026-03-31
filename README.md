# Distributional Reinforcement Learning for LTM Handover

This project explores the application of **Distributional Reinforcement Learning (DistRL)** to optimize the **Lower Layer Triggered Mobility (LTM)** handover process in 5G/6G wireless networks. Unlike traditional RL, which focuses on the expected value of future rewards, Distributional RL models the entire probability distribution of returns, enabling more robust and risk-aware decision-making in highly dynamic environments like mobile communications.

---

## 🎯 Project Objectives

### 1. Robust Handover Decision-Making
Implement and evaluate state-of-the-art Distributional RL algorithms (C51, IQN, QRDQN, and FQF) to handle the uncertainty and high variability of the 5G radio environment.

### 2. LTM-HO Simulation Framework
Develop a high-fidelity simulation environment based on the **LTM (Lower Layer Triggered Mobility)** protocol, incorporating:
*   **Realistic Physical Layer:** 5G signal-to-interference-plus-noise ratio (SINR) modeling.
*   **Mobility Patterns:** Real-world vehicle trajectories (via SUMO integration).
*   **Protocol Dynamics:** Multi-stage handover processes including preparation, execution, and resource reservation.

### 3. Multi-Objective Optimization
Optimize the handover process across conflicting metrics:
*   **Maximize Throughput:** Maintain high Modulation and Coding Schemes (MCS).
*   **Minimize Failures:** Reduce Radio Link Failures (RLF) and Handover Failures (HOF).
*   **Cost Efficiency:** Minimize unnecessary resource reservations and ping-pong effects.

### 4. Risk-Aware Strategies
Leverage the "Distributional" aspect of the agents to implement risk-aware policies (e.g., using CVaR or CPW) that prioritize network stability during critical mobility events.

---

## 🏗️ System Architecture

To ensure modularity and scalability, the reinforcement learning implementation is divided into two core components:

### 1. Agents (`agents/`)
The **Agent** handles the high-level **learning logic** and interaction with the environment. It is responsible for:
*   Experience Replay management and sampling.
*   Loss calculation (e.g., Quantile Huber Loss).
*   Hyperparameter management (learning rate, discount factor, epsilon-greedy).
*   Action selection and training loops.

### 2. Models (`models/`)
The **Model** defines the **neural network architecture** (the "brain"). It maps input observations (e.g., SINR vectors or Atari images) to the appropriate output format (e.g., Q-value distributions). 

**Benefit:** This modularity allows us to use the same Agent logic for both complex 5G handover simulations and standard Atari baselines simply by swapping the Model architecture.

---

## 🚀 Roadmap

- [ ] **Phase 1: Baseline Verification (Atari)**
    - [ ] Modify and test original algorithms in standard **Atari (ALE)** environments.
    - [ ] Establish performance baselines to verify algorithm correctness.
- [ ] **Phase 2: LTM-HO Gymnasium Wrapper**
    - [ ] Wrap the `ltm_ho.py` simulation into a standard `Gymnasium` environment.
- [ ] **Phase 3: Integration & Training**
    - [ ] Adapt DistRL agents to the 5G handover observation and action spaces.
    - [ ] Benchmark agents against classical rule-based handover strategies.
- [ ] **Phase 4: Risk-Aware Policy Evaluation**
    - [ ] Implement and evaluate risk-aware decision-making policies (e.g., CVaR) for network stability.

---

## 📋 Requirements

*   Python 3.8+
*   PyTorch
*   Gymnasium
*   NumPy / SciPy
*   SUMO (for trajectory generation)
*   Pandas / NetworkX

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
