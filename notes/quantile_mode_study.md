# QR-DQN quantile-positioning study

Working document for the comparison of quantile-placement schemes in
QR-DQN. Update results here as runs finish.

## What's being compared

All variants share the HP-search champion settings (`lr=1e-4`, all-else
baseline) and differ only on the QR-DQN representation:

| # | ID                  | Quantile mode  | Risk policy        | Notes |
|---|---------------------|----------------|--------------------|-------|
| 1 | qmode_midpoint      | midpoint (uniform tau, uniform weights) | mean | vanilla QR-DQN baseline |
| 2 | qmode_gauss_legendre| Gauss-Legendre nodes on [0,1] + GL weights | mean | better quadrature, same network output count |
| 3 | qmode_trapezoidal   | uniform-with-endpoints; q_min=0, q_max=15 | mean | network only predicts N-2 interior quantiles; endpoints are fixed integration anchors |
| 4 | qmode_cvar_full     | midpoint           | CVaR(0.1)          | bottom-k average over the full learned distribution |
| 5 | qmode_cvar_truncated| midpoint, only bottom-k | CVaR(0.1) (=mean of those k) | top 90% never represented; tests whether the unused upper lobe drags learning |

## Selection criterion

Same as the HP search:

- Primary: `capacity_avg` (higher is better).
- Tie-break (cap within ~5%): lower `hof_rate`, then `rlf_rate`.
- `ho_rate` / `pp_rate` documented for context, not optimised.

## How to run

Sequentially via the bundled orchestrator:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./venv-RL/bin/python3 src/tools/run_quantile_mode_study.py --device xpu
```

Or run a single variant directly:

```bash
./venv-RL/bin/python3 src/main.py \
    --config configs/quantile_study/qmode_gauss_legendre.yaml \
    --agents qrdqn --device xpu --description qmode_gauss_legendre
```

## Results

| # | ID                  | Status   | capacity_avg | hof_rate | rlf_rate | ho_rate | pp_rate | Notes |
|---|---------------------|----------|--------------|----------|----------|---------|---------|-------|
| 1 | qmode_midpoint      | done     | 3.515        | 1.989    | 0.258    | 16.363  | 2.093   | bmk_2026-05-17_3; within noise of HP champion (3.510 / 2.154 / 0.257) — corrected code does not regress |
| 2 | qmode_gauss_legendre| done     | 3.497        | 2.417    | 0.275    | 16.121  | 2.058   | bmk_2026-05-17_4; ~0.5% cap regression, **HOF +21% worse** — non-uniform quantile placement does NOT help on LTM |
| 3 | qmode_trapezoidal   | done     | **3.528**    | **1.885**| **0.235**| 16.704  | 2.156   | bmk_2026-05-17_5; **leads all three primary metrics**. Fixed q_min=0/q_max=15 endpoints appear to regularise the extreme quantile estimates. |
| 4 | qmode_cvar_full     | running  |              |          |          |         |         |       |
| 5 | qmode_cvar_truncated| pending  |              |          |          |         |         |       |

## Things to revisit if results are tight

- `q_max` for trapezoidal is a guess (15.0). Measure the empirical Q-value
  range from the midpoint baseline's learned distribution and re-run if
  it sits notably below/above.
- 1 seed × 2000 ep matches the HP-search extras budget; consider a
  3-seed re-run for the top two if their gap is under HOF/RLF noise.
- `num_quantiles=50` is fixed for parity. The existing
  `tools/run_ablation.py` can sweep N independently after the mode is
  locked.

## Implementation note (target-net fix)

The QRDQN refactor that introduced the schemes also gave the target
network real parameters of its own. Previously the target net's trunk
and head were the same module instances as the online net's, so the
soft-update became a no-op. HP-search absolute numbers were obtained
under that regime; numbers in this study run on top of the fixed
target net and are not directly comparable to the HP search.

## Implementation note (QR-loss quadrature fix)

A second bug landed mid-flight (Gemini-caught, commit `79126be`): the
QR Huber loss aggregator collapsed with `.mean()` over batch /
predictor / target, which uniformly averages each axis. That is
correct for midpoint (uniform weights) but discards the non-uniform
quadrature of gauss_legendre and trapezoidal — effectively turning
both into a uniform midpoint loss while keeping the predictor at the
non-uniform tau nodes.

The fix weights the target axis by `scheme.mean_weights` (probability
mass at each target atom) and the predictor axis by
`scheme.predictor_weights` (integration weights for the loss across
tau). For midpoint these are uniform 1/N and the new loss is
bit-equal to the old `.mean()` (locked in by
`test_loss_backward_compat`). For trapezoidal the Bellman target now
also includes the fixed q_min and q_max atoms in the target axis.

Per-variant impact:

  - `qmode_midpoint`: identical loss before and after the fix.
    Variant 1 was running with old code at the moment of the fix;
    results are valid either way.
  - `qmode_gauss_legendre`: spawned after the fix → uses the corrected
    GL-weighted loss.
  - `qmode_trapezoidal`: spawned after the fix → target includes the
    fixed endpoints + uses trapezoidal weights.
  - `qmode_cvar_full` / `qmode_cvar_truncated`: built on midpoint, so
    target distribution stays uniform → loss identical before/after.
