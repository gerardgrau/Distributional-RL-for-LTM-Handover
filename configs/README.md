# configs/

All training / evaluation YAMLs. Pick one with `--config <path>` on `main.py`.

## Canonical
- **`config.yaml`** — Default. 500-episode multi-seed dev config, balanced
  hyperparameters. Use this for normal development.
- **`definitive-10h.yaml`** — Long benchmark (2000 episodes × 2 seeds). The
  intended config for the paper-parity definitive run. Run on `--device cpu`
  (`main.py` parallelizes seeds across cores — see `CLAUDE.md` on why CPU beats
  the XPU path here).

## Ablations / studies
- **`definitive-comparison.yaml`** — DQN vs QR-DQN comparison at moderate
  scale.
- **`definitive-5000ep.yaml`** — 5000-episode definitive (deeper, slower).
- **`test-quantiles.yaml`** — Quantile-count ablation base config (consumed
  by `tools/run_ablation.py`).

## Tests / smoke / perf
- **`tiny-test.yaml`** — Smallest possible config; finishes in seconds.
  Use for "does the pipeline run at all?" sanity checks.
- **`smoke-test-all.yaml`** — Smoke all three agents (DQN, QR-DQN, baseline)
  on a short episode budget.
- **`feature-test.yaml` / `feature-test-long.yaml`** — Iteration-time configs
  for trying out new features without running a real benchmark.
- **`hardware-test.yaml`** — CPU vs XPU device benchmark config.
- **`perf-baseline.yaml`** — Profiling baseline for performance regressions.
- **`profile-config.yaml`** — cProfile-friendly minimal config.
- **`test-metadata-config.yaml`** — Tiny config that exercises the
  `metadata.json` writing path end-to-end.

## Study subfolders

Beyond the top-level configs above, the subdirectories (`masked/`, `headline/`,
`alpha_v3/`, `cvar_alpha/`, `n_sweep/`, `quantile_study/`, `quantile_bridge/`,
`hp_search/`, `hp_refresh/`, `atari/`, …) hold the per-variant YAMLs for the
sweeps and studies. Most are driven by the orchestrators in `src/tools/`
(`run_ablation.py`, `run_cvar_study.py`, `run_quantile_mode_study.py`,
`run_hp_refresh.py`, `run_atari_study.py`); a few are run directly with
`--config`. `masked/no_gate/finals_n25/` holds the canonical no-gate final
agents reported in the paper.

The active YAML is copied into every benchmark output folder (`<bmk>/config.yaml`)
for reproducibility. The full list of consumable keys is documented inline at
the top of `config.yaml`.
