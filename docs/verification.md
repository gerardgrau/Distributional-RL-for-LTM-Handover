# Environment Verification Guide

This document provides instructions on how to manually verify that the RL environments (Atari and LTM-HO) are correctly configured and functional.

## Prerequisites
Ensure the virtual environment is set up and dependencies are installed:
```bash
# Install dependencies if you haven't already
./venv-RL/bin/python3 -m pip install -r requirements.txt
```

---

## 1. Atari Environment Verification
This test verifies that `Gymnasium` and `ale-py` (Atari) are correctly registered and can execute actions.

### Running the Test
Execute the standalone test script:
```bash
./venv-RL/bin/python3 src/gym_test/test_env.py
```

### What to Expect
- **If you have a GUI/Display**: A window will open showing the "ALE/Breakout-v5" game. You will see a paddle moving randomly and a ball bouncing.
- **If you are in a headless environment (no display)**: The script will crash with a "No display" error. 
    - *To fix this for headless testing, edit `src/gym_test/test_env.py` and change `render_mode="human"` to `render_mode=None`.*
- The terminal will print: `Starting environment verification...`

---

## 2. LTM-HO Simulation Verification
This test verifies that the custom Lower Layer Triggered Mobility (LTM) handover simulation can load channel data and execute the protocol logic.

### Running the Test
Ensure the project root is in your `PYTHONPATH` so the script can find internal modules, then run the environment file:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./venv-RL/bin/python3 src/distrl/envs/ltm_env.py
```

### What to Expect
1.  **Detection**: The script should print: `Detected 1000 UE channel files. Simulating 5 UEs.`
2.  **Simulation**: You will see progress updates: `Simulando UE 1/5...`, `Simulando UE 2/5...`, etc.
3.  **Completion**: It will print `Simulación completada.` followed by a **PERFORMANCE SUMMARY** table for each UE (Capacity, HO count, HOF, etc.).
4.  **Artifacts**: A folder named `ltm_decision/` will be created containing `Performance_all.pkl` and `Metrics.pkl`.

---

## Troubleshooting
- **ModuleNotFoundError**: Ensure you are using the python binary from the virtual environment: `./venv-RL/bin/python3`.
- **FileNotFoundError**: Ensure you are running the commands from the **project root** directory so that relative paths like `data/ChannelGains` resolve correctly.
