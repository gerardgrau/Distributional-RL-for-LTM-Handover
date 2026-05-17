# Autonomous experiments log

Master tracker for the work done while the user is away. Updated as
each batch lands. Read top-down: most recent at the bottom.

## Branch + commit context

Working branch: `feature/distributional-improvements` (rebased onto
`master` after the hp-search merge).

Key code commits this log relies on:
  - `4797207` (rebased to `667c69b`) — non-uniform quantile modes +
    Atari scaffolding + target-net fix
  - `db684b0` (rebased to `ddd90c2`) — replay buffer skips pinning on
    CPU target
  - `79126be` — QR-loss quadrature fix (target axis weighted by
    `mean_weights`, predictor axis by `predictor_weights`)
  - `14df72f` — Atari runner simplifications
  - `bf02c33` — terminal_on_life_loss=True for Atari training

## Running studies

| ID | What | Device | ETA | Output dirs |
|----|------|--------|-----|-------------|
| `bumx7je6v` | LTM quantile-mode study (5 variants × 2000 ep × 1 seed) | XPU | ~12h total | `results/benchmarks/bmk_2026-05-17_3_qmode_*` |
| (TBD) | Atari Breakout study (5 QR-DQN variants × 500k frames × 1 seed), relaunched after the terminal_on_life_loss fix | CPU | ~5h total | `results/atari/Breakout_qrdqn_*` |

## Planned follow-ups (autonomous queue)

After LTM and Atari studies finish, in priority order:

1. **HP delta check** — read `qmode_midpoint` eval (LTM). If
   `capacity_avg` is within ~5% of HP-search champion (3.510), the
   target-net + quadrature fixes did not shift the HP landscape and we
   keep the champion config. If meaningfully worse, kick off a
   focused HP re-search on the axes most affected by the
   target-net fix: `lr` (1e-4, 3e-4, 1e-3) and `tau` (0.005, 0.01,
   0.05), plus a hard-update variant (`tau=1.0` + finite
   `target_update_freq`). Budget: 6 runs × 500 ep × 1 seed ≈ 5h on
   XPU.

2. **Atari generalisation** — re-run the 5-variant study on Pong and
   SpaceInvaders to verify the LTM finding generalises. ~5h per game
   on CPU.

3. **Trapezoidal q_max tuning** — if trapezoidal lands competitive on
   LTM, measure the empirical Q-value range from the `qmode_midpoint`
   trained model and re-run trapezoidal with a tighter `q_max`. One
   variant, 2000 ep, ~3h on XPU.

4. **Top-2 multi-seed re-validation** — pick the two best variants by
   `capacity_avg` (HOF/RLF tie-break) and run them 3 seeds × 2000 ep.
   ~5h on XPU per variant.

5. **Quantile-distribution plots** — for each trained model, sample
   states and dump `plot_quantiles` figures. Cheap (minutes).

Document all of these as they complete.

## Results

(Filled in as each variant lands. See also
`notes/quantile_mode_study.md` for the per-variant LTM table.)

### LTM quantile-mode study (in-flight)

| Variant | Status | cap_avg | hof_rate | rlf_rate | ho_rate | pp_rate | reliability | Notes |
|---------|--------|---------|----------|----------|---------|---------|-------------|-------|
| qmode_midpoint | running |  |  |  |  |  |  | |
| qmode_gauss_legendre | pending |  |  |  |  |  |  | uses corrected QR-quadrature loss |
| qmode_trapezoidal | pending |  |  |  |  |  |  | uses corrected loss + assembled target |
| qmode_cvar_full | pending |  |  |  |  |  |  | midpoint base, CVaR(0.1) action selection |
| qmode_cvar_truncated | pending |  |  |  |  |  |  | k=5 quantiles in [0, 0.1] |

### Atari Breakout (in-flight, relaunched with `terminal_on_life_loss=True`)

| Variant | Status | final_eval | wall-clock | Notes |
|---------|--------|------------|------------|-------|
| qrdqn_midpoint | pending |  |  | baseline |
| qrdqn_gauss_legendre | pending |  |  | |
| qrdqn_trapezoidal | pending |  |  | q_max=50, q_min=-50 |
| qrdqn_cvar_full | pending |  |  | |
| qrdqn_cvar_truncated | pending |  |  | |

## Open questions / known caveats

- Trapezoidal `q_max` is a guess (LTM 15.0, Atari 50.0). With loose
  endpoints, the predicted top quantile is biased toward the fixed
  value. For paper-grade trapezoidal numbers, tighten after the
  midpoint baseline finishes (planned follow-up #3).
- Atari frame budget (500k) is enough for ordering but not absolute
  paper-quality numbers (literature uses 10M+). Treat as a
  relative-comparison study, not a Nature-DQN reproduction.
- CVaR + Gauss-Legendre combinations are not in the active study —
  the CVaR cutoff doesn't align exactly with GL nodes, so the CVaR
  estimate is slightly approximate (documented in
  quantile_modes.py). Acceptable for the comparison.
