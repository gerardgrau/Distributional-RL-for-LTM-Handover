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
| 1 | qmode_midpoint      | pending  |              |          |          |         |         |       |
| 2 | qmode_gauss_legendre| pending  |              |          |          |         |         |       |
| 3 | qmode_trapezoidal   | pending  |              |          |          |         |         |       |
| 4 | qmode_cvar_full     | pending  |              |          |          |         |         |       |
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
