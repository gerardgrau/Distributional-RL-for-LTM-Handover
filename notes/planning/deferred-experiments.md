# Deferred experiments (post-tutor-meeting backlog)

Held off intentionally on 2026-05-18. The Simpson code path and the
sweep orchestrators are in place; the runs themselves are pending.

## Already implemented, just not run

These have configs + orchestrators ready. Each is a single command away.

### 1. Simpson's rule on LTM
- Config: `configs/quantile_study/qmode_simpson.yaml` (N=51, q_min=0, q_max=15)
- Run via: `./venv-RL/bin/python3 src/main.py --config configs/quantile_study/qmode_simpson.yaml --agents qrdqn --device xpu --description qmode_simpson`
- Or via the orchestrator (which now includes it):
  `./venv-RL/bin/python3 src/tools/run_quantile_mode_study.py --only qmode_simpson --device xpu`
- Cost: ~3h on XPU at 2000 ep × 1 seed.
- Hypothesis: lands between GL (3.497) and midpoint (3.515). Simpson is order-4 accurate vs midpoint/trap's order-2, but uniform tau placement avoids GL's boundary under-training. Probably uninformative for the headline, but completes the "spectrum of quadratures" axis.

### 2. Quantile-count ablation (N ∈ {10, 25, 50, 100, 200})
- Orchestrator: `src/tools/run_ablation.py`
- Most informative on top of `cvar_truncated` (the winning variant): does the truncated grid still benefit from more atoms?
- Cost: ~12-15h on XPU for 5 variants × 2000 ep × 1 seed.

### 3. Risk-fraction (CVaR α) sweep
- Orchestrator: `src/tools/run_cvar_study.py`
- α ∈ {0.05, 0.10, 0.20, 0.30, 0.50} on top of `cvar_truncated`.
- Most important follow-up: directly probes the winning axis. Cost: ~15h on XPU.

### 4. Hyperparameter searches that we never touched

  - **Huber `kappa`** (held at 1.0) — relevant after our predictor-collapse analysis. Try κ ∈ {0.25, 0.5, 1.0, 2.0}. Smaller κ → less Huber bias on truncated CVaR.
  - **`hidden_dims`** (held at [128, 128]) — capacity. Truncated CVaR has 5 output heads vs 50, so the trunk might be over-provisioned. Try [64, 64].
  - **`batch_size`** (likely 256 in current configs) — affects target-distribution diversity per gradient step. Worth a quick check at 128 / 512.
  - **Gradient clipping** (off) — Dabney clips to 10 in the original QR-DQN. Single ablation.
  - **Adam `eps`** (1e-8 default) — Rainbow uses 1.5e-4. Sometimes matters.

## Locked-for-paper-parity (do NOT touch)

- Reward alphas (`α_HOF=0.1, α_HO=0.8, α_PP=0.9`) — paper defaults.
- Physics constants (TxPower, noise floor, SINR table, ICIC) — see
  `docs/reference/ltm_ho_codi_ainna.py` and `notes/tasks.md` parity
  config.

## Priority order if compute frees up

If I had ~25h of XPU after the meeting, in priority order:
1. risk_fraction sweep (most likely to move the headline number)
2. num_quantiles ablation on top of cvar_truncated
3. kappa sweep on top of cvar_truncated
4. Simpson's rule (cheap, completes the quadrature axis for the paper)
5. trapezoidal q_max retuning (currently a guess at 15.0)
