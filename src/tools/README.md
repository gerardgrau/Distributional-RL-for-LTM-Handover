# src/tools/

Standalone utility scripts. None of these are imported by the main `src/distrl/`
package — they're command-line tools you run by hand. Group them by purpose:

## Data pipeline
- **`preprocess_dataset.py`** — One-time: convert raw `data/ChannelGains/*.mat`
  into `data/Precomputed/{no_ris,with_ris}/User*_precomputed.npz`. Stamps the
  current `physics_hash()` into each file; the env refuses to load a cache
  whose hash doesn't match the live constants.

## Final-comparison plots (master results)
- **`generate_final_plots.py`** — Read every `*_summary.csv` in
  `results/final_metrics/` and produce `master_bar_plots.png` +
  `master_radial_plot.png` in `results/final_metrics/plots/`. The canonical
  comparison against the paper numbers (`paper_ltm.csv` etc.).
- **`generate_legacy_summary.py`** — Run `legacy_simulation.py` over all
  1000 UEs and write `results/final_metrics/legacy_baseline_summary.csv`.

## Multi-config orchestrators
- **`run_ablation.py`** — Sweep QR-DQN over `num_quantiles ∈ {10, 50, 100, 200}`.
- **`run_cvar_study.py`** — Sweep QR-DQN CVaR `risk_fraction` over `{0.05, 0.1, 0.25, 0.5}`.

## Profiling
- **`benchmark_device.py`** — CPU vs CUDA vs XPU iteration-rate benchmark for
  the env-only and env+agent loops.
