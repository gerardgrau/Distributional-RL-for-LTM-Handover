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

## 🏗️ Repository Structure

The project is organized as a modular Python package to ensure scalability and clear separation of concerns:

- `src/`: Source code directory.
    - `main.py`: Main entry point for training and simulations.
    - `distrl/`: Core framework package.
        - `agents/`: Learning logic and algorithm implementations (C51, IQN, etc.).
        - `models/`: Neural network architectures (decoupled from agent logic).
        - `envs/`: Gymnasium environment wrappers and LTM-HO simulation logic.
        - `utils/`: Shared utilities, configuration management, and data loading.
- `data/`: Data storage for simulation artifacts.
    - `ChannelGains/`: Storage for generated channel gain datasets.
- `conductor/`: Local project management and planning (untracked).
- `config.yaml`: Global configuration file for hyperparameters and simulation settings.

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.8+ and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running a Single Experiment
The project uses a centralized entry point in `src/main.py`. To run it, ensure the `src/` directory is in your `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/main.py
```

### 3. Benchmarking & Visualization
For scientific evaluation across multiple seeds, use the benchmarking suite:

#### Run Experiment
Runs automated trials for both DQN and QRDQN across multiple random seeds.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 src/experiment.py
```

#### Generate Performance Plots
Creates learning curves and efficiency graphs (Reward/Loss vs Time).
```bash
python3 src/distrl/utils/plot.py
```

#### Distributional Visualizer
Visualizes the learned return distributions (quantiles) for a specific state.
```bash
python3 src/gym_test/test_quantile_vis.py
```

### 4. Verification
To ensure everything is working correctly, follow the [Environment Verification Guide](docs/verification.md).

---

## 📋 Requirements

*   Python 3.8+
*   PyTorch
*   Gymnasium
*   NumPy / SciPy
*   PyYAML
*   ale-py (for Atari baselines)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
