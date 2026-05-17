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
| `bumx7je6v` | LTM quantile-mode study (5 variants × 2000 ep × 1 seed) | XPU + CPU | ~11h remaining | `results/benchmarks/bmk_2026-05-17_3_qmode_*` |
| `btmoh6awg` | Atari Breakout study (5 QR-DQN variants × 500k frames × 1 seed), `terminal_on_life_loss=True` for train, capped at 3 PyTorch threads to avoid starving LTM | CPU | ~4-5h | `results/atari/Breakout_qrdqn_*` |

System utilisation under parallel run: load avg ~15 on 22 cores
(LTM ≈7 cores XPU+CPU, Atari ≈3 cores capped). LTM steps at ~6 s/ep
vs the 4.7 s/ep solo baseline — a 30% slowdown that the parallel
throughput more than recovers vs running them sequentially.

## Planned follow-ups (autonomous queue)

After LTM and Atari studies finish, in priority order:

1. **HP refresh** (configs + orchestrator already in `configs/hp_refresh/`
   and `src/tools/run_hp_refresh.py`) — re-tests the champion plus
   five target-net-sensitive variants (`lr` ∈ {1e-3, 3e-4}, `tau` ∈
   {0.005, 0.05}, hard update with `tau=1.0`). 500 ep × 1 seed each,
   ~70 min on XPU. Done unconditionally, not just on regression — the
   `tau` and hard-update axes are genuinely new since the target-net
   fix.

2. **Atari generalisation** — re-run the 5-variant study on Pong and
   SpaceInvaders. ~5h per game on CPU (with `--threads 3` if LTM
   follow-ups are running on XPU).

3. **Top-2 multi-seed re-validation** — pick the two best LTM
   variants by `capacity_avg` (HOF/RLF tie-break) and run them 3
   seeds × 2000 ep. ~5h on XPU per variant.

4. **Trapezoidal q_max tuning** — if trapezoidal lands competitive on
   LTM, measure the empirical Q-value range from the `qmode_midpoint`
   trained model and re-run trapezoidal with a tighter `q_max`. One
   variant, 2000 ep, ~3h on XPU.

5. **Quantile-distribution plots** — for each trained model, sample
   states and dump `plot_quantiles` figures. Cheap (minutes).
   Aggregator plots already ready:
   `src/tools/plot_quantile_mode_study.py`,
   `src/tools/plot_atari_study.py`.

Document all of these as they complete.

## Results

(Filled in as each variant lands. See also
`notes/quantile_mode_study.md` for the per-variant LTM table.)

### LTM quantile-mode study (in-flight)

| Variant | Status | cap_avg | hof_rate | rlf_rate | ho_rate | pp_rate | reliability | Notes |
|---------|--------|---------|----------|----------|---------|---------|-------------|-------|
| qmode_midpoint | **done** | **3.515** | **1.989** | **0.258** | 16.363 | 2.093 | 95.528 | Essentially equal to HP champion (3.510 / 2.154 / 0.257) — corrected code does not regress the baseline. Even slightly better on HOF and reliability. |
| qmode_gauss_legendre | done | 3.497 | 2.417 | 0.275 | 16.121 | 2.058 | 95.422 | Slight cap regression (-0.5%), HOF +21% worse. Non-uniform quantile placement does NOT improve LTM. Tentative interpretation: midpoint is already sufficiently accurate for the Q-value range in this task, and GL's quadrature advantage doesn't surface. |
| qmode_trapezoidal | done | **3.528** | **1.885** | **0.235** | 16.704 | 2.156 | 95.576 | **Best so far on all three primary metrics**. Capacity +0.4% vs midpoint, HOF −5%, RLF −9%. The fixed q_min=0 / q_max=15 endpoints appear to anchor the extreme predicted quantiles and reduce variance there — a mild regulariser effect on top of the QR loss. Trades against a small bump in HO/PP rate (more aggressive handover triggering). |
| qmode_cvar_full | done | 3.492 | 2.502 | 0.237 | 18.511 | 2.083 | 95.242 | Worst on capacity and HOF, but RLF ties trapezoidal. HO rate +13% vs midpoint — CVaR(0.1) selects more conservative (defensive) actions that trigger handovers more often, but the extra handovers themselves fail more (HOF up). Tentative reading: LTM's reward is already dense and well-shaped, so risk-aware action selection doesn't add signal; it adds noise via the small bottom-k tail estimate. |
| qmode_trapezoidal | pending |  |  |  |  |  |  | uses corrected loss + assembled target |
| qmode_cvar_full | pending |  |  |  |  |  |  | midpoint base, CVaR(0.1) action selection |
| qmode_cvar_truncated | pending |  |  |  |  |  |  | k=5 quantiles in [0, 0.1] |

**Key finding from variant 1**: the target-net fix + QR-quadrature
fix didn't shift the LTM dynamics under midpoint. The HP champion
config is still valid. We can interpret the remaining four
variants' absolute numbers in HP-search terms.

### Atari Breakout (aborted — 500k frames is too sparse-reward)

| Variant | Status | final_eval | wall-clock | Notes |
|---------|--------|------------|------------|-------|
| qrdqn_midpoint | done (uninformative) | 0.00 | 2h48m | Eval = 0 reward over 5 episodes. The agent never explored enough to break any brick (eps was still ≈0.50 at frame 500k since `epsilon_decay_frames=1_000_000` was set for 1M-frame schedules). Literature QR-DQN runs at 10M+ frames for Breakout. Switching to Pong, which gives reward every 1-2 s of play and is learnable inside a 500k-frame budget. |

### Atari Pong (in-flight, denser reward)

| Variant | Status | final_eval | wall-clock | Notes |
|---------|--------|------------|------------|-------|
| qrdqn_midpoint | running |  |  | replaces the Breakout sweep |
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
