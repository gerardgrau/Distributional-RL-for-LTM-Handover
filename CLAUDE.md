# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project applying **Distributional Reinforcement Learning (DQN vs QR-DQN)** to optimize **5G Lower Layer Triggered Mobility (LTM)** handover decisions. The environment simulates a multi-sector 5G deployment (7 BS × 3 sectors = 21 sectors, NBS=21) over 1,000 pre-computed UE trajectories from `.mat`/`.npz` channel-gain datasets.

This file is the single source of truth for architecture, commands, and parity decisions. When state space, reward, physics, or simulation constants change, update it here.

## Environment & Common Commands

The project ships a virtualenv at `venv-RL/`; `requirements.txt` lists deps (PyTorch, Gymnasium, NumPy/SciPy, PyYAML, matplotlib, pandas, tqdm, sumolib, ale-py extras). Always export `PYTHONPATH` before invoking any script — every entry point uses absolute `src.distrl.*` imports.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./venv-RL/bin/python3 <script>            # use the venv binary explicitly
```

### Training / benchmarks
```bash
# Full benchmark (DQN + QR-DQN, multi-seed, auto eval + plots)
./venv-RL/bin/python3 src/main.py --config configs/config.yaml --device cpu --description my-run

# Profiling run with no artifacts
./venv-RL/bin/python3 src/main.py --config configs/config.yaml --no_save

# Pick agents to run (comma-separated): dqn, qrdqn, ltm_baseline
./venv-RL/bin/python3 src/main.py --agents qrdqn --device cpu
```

`--device` accepts `cpu`, `cuda`, or `xpu`. **Use `cpu`.** Because `main.py`
self-parallelizes seeds across CPU cores and the networks are small MLPs,
multi-core CPU is substantially faster in wall-clock than the Intel iGPU
(`xpu`), which cannot parallelize seeds and is dominated by per-op overhead on
this workload. (Empirically: a 1000-ep × 3-seed `[256,256]` QR-DQN run is
~50 min on CPU vs **>85 min for a single seed** on XPU; every recent run,
including all the finals, uses CPU.) The older note that XPU was ~41% faster
predates the precomputed dataset + seed-parallelism and no longer holds.

### Single-agent evaluation (frozen weights, ε=0, all 1,000 users)
```bash
./venv-RL/bin/python3 src/evaluate_model.py \
    --agent qrdqn --model results/benchmarks/<run>/models/qrdqn_best.pth \
    --config configs/config.yaml --output results/final_metrics/qrdqn
```

### Data preprocessing (one-time, generates `data/Precomputed/{no_ris,with_ris}/*.npz`)
```bash
./venv-RL/bin/python3 src/tools/preprocess_dataset.py
```

### Smoke / verification tests (no pytest harness — each is a runnable script)
```bash
./venv-RL/bin/python3 src/scripts/test_env.py                       # Atari smoke (Breakout via ale-py)
./venv-RL/bin/python3 src/scripts/test_dqn_ltm.py                   # DQN end-to-end
./venv-RL/bin/python3 src/scripts/test_qrdqn_ltm.py                 # QR-DQN end-to-end
./venv-RL/bin/python3 src/scripts/test_metrics_calc.py              # 8-metric correctness
./venv-RL/bin/python3 src/scripts/test_quantile_modes.py            # quantile-scheme math (midpoint/GL/trapezoidal/truncated)
./venv-RL/bin/python3 src/scripts/verify_simulation_parity.py       # LTM baseline in gym vs paper numbers
```

### Atari benchmark (QR-DQN on standard Nature-DQN preprocessing)
```bash
./venv-RL/bin/python3 src/atari_main.py \
    --config configs/atari/qrdqn_midpoint.yaml \
    --game Breakout --frames 500000 --device cpu --threads 3
```
`--threads` caps the PyTorch intra-op pool (use 3-4 when running in parallel with an LTM job).

### Visualization & analysis
```bash
./venv-RL/bin/python3 src/tools/generate_final_plots.py         # master bar/radial plots from results/final_metrics/*.csv
./venv-RL/bin/python3 src/tools/plot_quantile_mode_study.py     # LTM 5-variant comparison (auto-discovers bmk_*_qmode_*)
./venv-RL/bin/python3 src/tools/plot_atari_study.py --game Breakout  # Atari 5-variant comparison
./venv-RL/bin/python3 src/scripts/run_dashboard.py              # MP4 agent-behavior animation (uses src/distrl/viz/dashboard.py)
./venv-RL/bin/python3 src/scripts/test_quantile_vis.py          # QR-DQN learned return-distribution viewer
```

### Multi-run orchestrators (sweep configs under `configs/`)
```bash
./venv-RL/bin/python3 src/tools/run_ablation.py                 # quantile-count ablation (N=10/50/100/200)
./venv-RL/bin/python3 src/tools/run_cvar_study.py               # CVaR risk-fraction sweep
./venv-RL/bin/python3 src/tools/run_quantile_mode_study.py      # midpoint vs GL vs trapezoidal vs CVaR-full vs CVaR-truncated
./venv-RL/bin/python3 src/tools/run_hp_refresh.py               # 6-variant HP refresh on top of the champion (lr, tau, hard-update)
./venv-RL/bin/python3 src/tools/run_atari_study.py --game Breakout  # same five quantile modes on Atari
```

## Architecture

### Two parallel simulators (must stay in physical parity)
- `src/distrl/envs/ltm_gym.py` — Gymnasium `LTMEnv` used by the RL agents. RL steps at 100 ms; the underlying simulator advances at 10 ms.
- `src/distrl/envs/legacy_simulation.py` — Standalone reference simulator (baseline LTM hardcoded algorithm) used for paper-parity verification.
- `src/distrl/envs/physics.py` — **Single source of truth** for SINR/MCS/HOF math and all `System`/`Time`/`HO` constants. Both simulators import from here. Do not duplicate physics code in the env files.

### Agent framework (`src/distrl/agents/`)
- `base.py` — `BaseAgent` ABC with shared soft/hard target-update logic.
- `networks.py` — `MLPTrunk` (LTM vector obs), `CNNTrunk` (Nature-DQN conv stack for 84x84x4 Atari uint8), and interchangeable heads (`QHead`, `QuantileHead`) wired via `UnifiedQNet`. Trunk/head split is intentional: agents pick a head, not a network. Both DQN and QR-DQN select the trunk via `agent.trunk_type ∈ {"mlp", "cnn"}`.
- `standard/dqn.py` — Vanilla DQN. Target net has independent params (soft / hard update both real).
- `standard/ltm_baseline.py` — Hardcoded LTM heuristic (no learning), used as a comparison baseline.
- `distributional/quantile_modes.py` — `QuantileScheme` dataclass + `build_scheme(...)` factory. Defines four quantile-positioning modes (midpoint / gauss_legendre / trapezoidal / midpoint+truncate_upper) and their `tau`, `mean_weights`, `cvar_weights`, `predictor_weights`. The agent consumes the scheme to drive both action selection (`scheme.expectation` / `scheme.cvar`) and the QR Huber loss (target-axis weighted by `mean_weights`, predictor-axis by `predictor_weights`).
- `distributional/qrdqn.py` — QR-DQN. Configured by `agent` block keys: `quantile_mode ∈ {midpoint, gauss_legendre, trapezoidal}`, `risk_type ∈ {mean, cvar}`, `risk_fraction`, `truncate_upper_quantiles`, `q_min`/`q_max` (trapezoidal only). Target net has independent params; loss weights both axes by quadrature.

### Observation & reward
- **State (88 dims)**: `[speed(1), tenure(1), serving_one_hot(21), RSRP(21), MA_MCS(21), MA_SNIR(21), x(1), y(1)]`. Moving averages use O(1) running sums.
- **Reward**: Multiplicative reward form — `R = R_thr · α_HO^ind_HO · α_PP^ind_PP · α_HOF^ind_HOF · reliability_factor`. Alphas live in the `ho_reward:` block of the active YAML (paper defaults: `α_HOF=0.1, α_HO=0.8, α_PP=0.9`); reliability factor is a soft reverse-sigmoid of the Out-Of-Sync counter.

### Config system
- All YAML configs live in `configs/`. `src/distrl/utils/config.py` exposes a `Config` singleton — call `Config.set_config_path(path)` once at startup, then `Config.get()` anywhere.
- `main.py` copies the active YAML into every benchmark folder (`config.yaml` next to `metadata.json`) for full reproducibility.

### Benchmark output layout
Every `main.py` invocation creates `results/benchmarks/bmk_YYYY-MM-DD_<num>_<description>/` with subfolders:
- `train/` — per-episode CSV logs (reward, loss, 8 metrics)
- `eval/` — post-training frozen evaluation on all 1,000 users (summary + raw CSVs)
- `models/` — `<agent>_seed<N>.pth` checkpoints plus `<agent>_best.pth`
- `figures/` — learning curves, efficiency plots, quantile distribution (QR-DQN)
- `config.yaml`, `metadata.json`

Stick to the `bmk_YYYY-MM-DD_num_description` naming when creating folders manually; `main.py` auto-increments `num` to avoid collisions.

### Evaluation protocol
Training and final evaluation both use **all 1,000 trajectories** (no train/test split). Final evaluation is invoked automatically at the end of each seed in `src/distrl/utils/evaluation.py` with ε=0 and frozen weights. The 8 metrics (Capacity, RLF, HO, PP, Reliability, Prep, ResReservation, HOF) are computed by `src/distrl/utils/metrics.py`; the prep/resource-reservation math is non-trivial and shared between live tracking and retroactive computation.

### Master comparison
`results/final_metrics/` aggregates the final summaries from this project (`dqn_summary.csv`, `qrdqn_summary.csv`, `baseline_summary.csv`, `legacy_baseline_summary.csv`) alongside the paper reference numbers (`paper_ltm.csv`, `paper_lmmse.csv`, `paper_ltm_cmab.csv`, `paper_lmmse_cmab.csv`). `src/tools/generate_final_plots.py` consumes that folder to produce the master bar/radial plots.

## Conventions

- **Code style**: Google Python Style Guide. 80-char lines, 4-space indent, snake_case modules/functions, PascalCase classes, ALL_CAPS constants. Type hints required on all function signatures; use modern Python 3.10+ syntax (`int | None`, `list[str]`, `dict[str, Any]`). Docstrings on public functions/classes use the Args/Returns/Raises format. Never use mutable defaults (`= []`, `= {}`) or bare `except:`. Use a leading underscore for module-private symbols.
- **Imports inside `src/`**: always import via the full `src.distrl....` path (mirrors what `main.py` sets up via `sys.path`).
- **`.gitignore`**: prefer distributed `.gitignore` files inside data/result subfolders with relative patterns (`/pattern`) over global rules — these double as documentation of expected directory contents.
- **Commits**: atomic, one logical change each; task summaries belong in the commit body.

## Things to double-check before editing

- Changes to `physics.py`, `System["TxPower"]`, `ExecPowerOffset`, or the SINR table affect **paper parity**. The current calibration follows the paper's Table I / II: `TxPower=25 dBm`, `NoiseLevel=-101 dBm` (reference code: -174 dBm/Hz over 20 MHz — the paper's prose implies 200 MHz/-91, but the reference wins, as `physics.py` confirms), `ExecPowerOffset=3.0 dB`, `MaxNumberPreparedBS=5` (reference code; paper Table II says 4 — the reference wins), 26-step SINR table with Outage < -3 dB, `ChannelBS2UE_noRIS` channels. The canonical reference implementation is `docs/reference/ltm_ho_reference.py` — when in doubt about a physics constant, match that file rather than the paper's prose. Re-run `src/scripts/verify_simulation_parity.py` after touching any of these.
- Anything that changes the 88-dim state vector breaks all saved `.pth` checkpoints — bump and document.
- The active calibration target is the published paper LTM numbers; `src/scripts/verify_simulation_parity.py` (LTMBaselineAgent in LTMEnv, 10 ms high-res, 1000 UEs) is the regression net. Gym ↔ legacy_simulation bit-exactness was achieved during the parity audit and is no longer continuously tested — if you change either side, re-run that script and eyeball the numbers against the paper LTM reference printed at the bottom of the run. By default the script is **read-only** (prints only); it overwrites the committed `results/final_metrics/baseline_summary.csv` only when run with `--canonical`, and that refresh must use the full 1000-UE set to stay comparable to `legacy_baseline_summary.csv`.
