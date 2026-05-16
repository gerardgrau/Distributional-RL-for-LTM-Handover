# Distributional Reinforcement Learning for LTM Handover

Research project applying **Distributional Reinforcement Learning** (DQN vs
QR-DQN) to optimize **Lower Layer Triggered Mobility (LTM)** handover decisions
in 5G/6G wireless networks. Unlike standard RL, which optimizes the expected
return, distributional RL models the full distribution of returns — enabling
risk-aware policies (CVaR) that are well suited to highly variable radio
environments.

## Project goals

1. Benchmark DQN, QR-DQN, and a hardcoded LTM baseline on a high-fidelity 5G
   sectored deployment (7 BS × 3 sectors, 1000 pre-computed UE trajectories).
2. Track 8 scientific metrics: capacity, RLF rate, HO rate, ping-pong rate,
   reliability, cell-preparation rate, resource reservation, HOF rate.
3. Compare results against the published Ainna / LTM-CMAB paper numbers under
   matched physics (TxPower=25 dBm, 26-step SINR table, etc.).
4. Explore risk-aware variants (CVaR-QR-DQN) for stability-critical mobility.

## Repository layout

```
src/distrl/      Core framework
  agents/        BaseAgent + DQN, QR-DQN, LTM heuristic baseline
  envs/          Gymnasium env, legacy reference simulator, shared physics
  utils/         Config, replay buffer, metrics, plot, evaluation, dashboard
src/tools/       Standalone CLI utilities (see src/tools/README.md)
src/scripts/     Smoke / verification scripts (see CLAUDE.md)
src/main.py      Training entrypoint
src/evaluate_model.py
                 Frozen-weight evaluation on all 1000 UEs
configs/         YAML configs (see configs/README.md)
data/            Channel-gain dataset (raw .mat + precomputed .npz cache)
docs/            Reference docs, parity audit notes, future improvements
notes/           Working notes — task list, meeting notes (see notes/README.md)
results/         All training / eval / animation outputs (see results/README.md)
```

For deeper guidance, see:
- **`CLAUDE.md`** — full architecture overview + canonical commands + parity decisions
- **`notes/tasks.md`** — live to-do list and calibration history

## Getting started

```bash
# 1. One-time setup (creates venv-RL, installs deps)
python3 -m venv venv-RL
./venv-RL/bin/pip install -r requirements.txt

# 2. Set PYTHONPATH (every shell)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 3. Generate the precomputed dataset (one-time)
./venv-RL/bin/python3 src/tools/preprocess_dataset.py

# 4. Run a benchmark (DQN + QR-DQN, multi-seed, auto eval + plots)
./venv-RL/bin/python3 src/main.py \
    --config configs/config.yaml \
    --device xpu \
    --description my-run
```

Outputs land in `results/benchmarks/bmk_<date>_<num>_<desc>/`. Frozen-weight
post-training evaluation runs automatically at the end of each seed.

## Performance

The env is heavily optimized for research-scale iteration:
- Precomputed channel-gain `.npz` cache (no RAM resets, hashed against the
  active physics constants)
- Vectorized SINR / MCS / HOF math in `src/distrl/envs/physics.py`
- O(1) moving-average state computations
- ~1.5 s/episode on Intel XPU (≈ 41 % faster than CPU since the precomputed
  cache removes the radio-physics bottleneck)

## License

MIT — see the `LICENSE` file.
