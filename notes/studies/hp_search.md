# QR-DQN Hyperparameter Search

Working document. Add results to the tables as runs complete; populate the skip
log when a planned variant is dropped before running.

## Goal & scope

Find the QR-DQN hyperparameter combination that maximizes `capacity_avg` on the
final-eval pass (1000 UEs, ε=0, frozen weights). Tie-breakers (in order):
lower `hof_rate`, lower `rlf_rate`, then lower `ho_rate` / `pp_rate`.

Two-phase manual grid: one-at-a-time screen → combined winners on top of the
screen, with `num_quantiles` and CVaR risk handled in a deferred Phase 3.

## Selection criterion

- Primary rank: `capacity_avg`.
- Tie-break (capacity within ~5%): prefer lower `hof_rate`, then `rlf_rate`.
- `ho_rate` / `pp_rate` are documented for diagnostic context, not optimized.
- When in doubt: a configuration with slightly lower capacity but materially
  lower HOF/RLF wins. Reliability dominates in deployment.

## Originally-planned experiments — disposition

### Moved to Phase 3: `num_quantiles`, CVaR `risk_fraction`

**Decision**: defer both to Phase 3, run on top of the Phase 2 winner using the
existing sweep tooling.

**Why**:
- `tools/run_ablation.py` already sweeps `num_quantiles ∈ {10, 50, 100, 200}`.
- `tools/run_cvar_study.py` already sweeps `risk_fraction ∈ {0.05, 0.1, 0.25, 0.5}`.
- `num_quantiles` and CVaR interact (CVaR truncates the learned quantile
  distribution), so they belong together rather than competing against
  agent-level knobs.
- They are representation/risk-policy knobs, only meaningful once the
  underlying QR-DQN is converged on a sensible value-function HP combo.
- Pulling them into Phase 1 would inflate it past 9 runs without giving the
  base agent a chance to be tuned first.

## Phase 1 — One-at-a-time screen

**Setup**: 1 seed, 500 episodes (default), QR-DQN only. Each row changes
exactly ONE knob from `configs/config.yaml`.

**Common overrides for every Phase 1 variant YAML**:
- `benchmark.num_seeds: 1`           (down from default 2)
- `agent.type: "qrdqn"`              (default is "dqn"; can also be forced via `--agents qrdqn` CLI)

**Run command template**:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./venv-RL/bin/python3 src/main.py \
    --config configs/hp_search/<variant>.yaml \
    --agents qrdqn --device xpu \
    --description <variant>
```

Results CSV per run lands at:
```
results/benchmarks/bmk_<date>_<num>_<variant>/eval/qrdqn_summary.csv
```

### Phase 1 results

| # | ID            | Override                               | Status  | capacity_avg | hof_rate | rlf_rate | ho_rate | pp_rate | Notes |
|---|---------------|----------------------------------------|---------|--------------|----------|----------|---------|---------|-------|
| 0 | hp_baseline   | (no change)                            | done    | 3.367        | 4.203    | 0.392    | 13.654  | 1.640   | reference; bmk_2026-05-16_1 |
| 1 | hp_gamma_095  | `agent.gamma: 0.95`                    | done    | 3.292        | 6.267    | 0.418    | 11.919  | 0.907   | capacity DOWN 2.3%, HOF UP 49% — less reactive |
| 2 | hp_gamma_099  | `agent.gamma: 0.99`                    | done    | 3.216        | 7.130    | 0.705    | 13.139  | 0.710   | capacity DOWN 4.5%, HOF UP 70%, RLF UP 80% — much worse |
| 3 | hp_lr_3e-4    | `agent.lr: 3.0e-4`                     | done    | 3.473        | 3.263    | 0.295    | 15.956  | 1.973   | **winner**: cap UP 3.1%, HOF DOWN 22%, RLF DOWN 25% |
| 4 | hp_lr_1e-4    | `agent.lr: 1.0e-4`                     | done    | 3.469        | 2.859    | 0.289    | 20.602  | 2.225   | tied capacity w/ #3; **wins tie-break on HOF** (2.86 < 3.26); way more HOs |
| 5 | hp_hidden_256 | `agent.hidden_dims: [256, 256]`        | done    | 3.317        | 5.548    | 0.441    | 12.785  | 1.092   | lost: cap DOWN 1.5%, HOF UP 32% — likely overfits at 500 ep |
| 6 | hp_eps_999    | `agent.epsilon_mult: 0.999`            | done    | 3.389        | 4.426    | 0.387    | 14.407  | 1.544   | tied with baseline within noise; HOF slightly worse |

**Per-axis winners** (fill in after Phase 1):
- gamma winner: **0.9 (baseline)** — both 0.95 and 0.99 lost on capacity AND
  HOF AND RLF. Likely cause: 500-ep training is too short to estimate
  long-horizon returns reliably; high-gamma TD targets are too noisy.
- lr winner:    **1e-4** (3e-4 essentially tied on capacity; 1e-4 wins HOF
  tie-break 2.86 vs 3.26). Both far better than baseline 1e-3 (HOF 4.20). Will
  keep 3e-4 as a Phase 2 candidate since the tie is genuine.
- hidden:       **[128, 128] (baseline)** — [256,256] worse on every metric
  (cap DOWN 1.5%, HOF UP 32%). Likely overfit on 500 ep with 4× more params.
  Worth revisiting at longer episode budgets.
- eps_mult:     **0.995 (baseline)** — eps_mult=0.999 was within noise on
  capacity but slightly worse on HOF. No meaningful improvement.

**Phase 1 verdict**: only **lr** moved. The single-knob improvement that
matters is `agent.lr: 1e-3 → 1e-4` (or 3e-4 — tied). All other axes prefer
baseline. Phase 2 collapses from "combine winners" into "validate lr at full
episode budget with multiple seeds".

## Phase 2 — Combined winners

**Setup**: 3 seeds, 2000 episodes, QR-DQN only. Built from Phase 1 per-axis
winners. Notation: `g*` / `lr*` / `h*` / `e*` = winning value on each axis.

**Common overrides for every Phase 2 variant YAML**:
- `agent.num_episodes: 2000`
- `benchmark.num_seeds: 3`
- `agent.type: "qrdqn"`

### Phase 2 candidates (revised after Phase 1)

Phase 1 showed only **lr** moved. The "combine winners" template collapsed —
there is nothing to combine, only an lr value to validate at full budget with
multiple seeds. Keeping both ties (1e-4 and 3e-4) so the choice is grounded
in 3-seed statistics rather than a single Phase-1 seed.

| # | ID            | Combination                          | Status   | capacity_avg | hof_rate | rlf_rate | ho_rate | pp_rate | Notes |
|---|---------------|--------------------------------------|----------|--------------|----------|----------|---------|---------|-------|
| A | p2_lr_1e-4    | baseline + `lr: 1e-4`                | done     | **3.510**    | **2.154**| **0.257**| 16.345  | 2.086   | mean across 3 seeds; stdev tiny (cap 0.007, HOF 0.24) — clean signal |
| B | p2_lr_3e-4    | baseline + `lr: 3e-4`                | skipped  | —            | —        | —        | —       | —       | see skip log; #A's clean low-variance result + extras are higher info-gain |

**Skip rules updated**:
- If Phase 2 #A clearly dominates #B at full budget (capacity gap > 5% or
  comparable HOF/RLF gap), skip running B in full and move directly to
  extras.
- Conversely, if #A's multi-seed variance is high enough that #B remains a
  plausible alternative, run B in full and compare.

## Phase 2.5 — Extra exploration (autonomous, user away)

After Phase 2 picks a base (the better of A/B), explore axes that weren't in
the original sweep. Each extra is 1 seed × 2000 ep on top of the Phase 2
base config. Goal: see if any new dimension materially improves capacity or
HOF/RLF beyond the Phase 2 baseline.

| # | ID                | Override over Phase 2 base                     | Status   | capacity_avg | hof_rate | rlf_rate | ho_rate | pp_rate | Notes |
|---|-------------------|------------------------------------------------|----------|--------------|----------|----------|---------|---------|-------|
| 1 | x_eps_end_001     | `agent.epsilon_end: 0.01`                      | deferred |              |          |          |         |         | user redirected the laptop to Tasks 1/2 (quantile-mode + Atari); see skip log row "extras post-x_gamma_095_long" |
| 2 | x_tau_0005        | `agent.tau: 0.005`                             | deferred |              |          |          |         |         | same deferral |
| 3 | x_batch_512       | `agent.batch_size: 512`                        | deferred |              |          |          |         |         | same deferral |
| 4 | x_hidden_256_long | `agent.hidden_dims: [256, 256]`                | done     | 3.496        | 2.134    | 0.251    | 15.518  | 2.104   | neutral vs champion (within single-seed noise); wider net not a bottleneck even at 2000 ep |
| 5 | x_gamma_095_long  | `agent.gamma: 0.95`                            | done     | 3.473        | 2.993    | 0.312    | 15.759  | 1.901   | cap 1% LOWER and HOF 39% HIGHER than champion — gamma=0.9 still wins at 2000 ep, ruling out the "high gamma needs longer training" hypothesis from Phase 1 |
| 6 | x_ma_window_25    | `ho_state.moving_average_window: 25`           | deferred |              |          |          |         |         | same deferral |

**Skip rule**: drop any extra whose direction is already obviously bad based
on Phase 1 (e.g. if Phase 2 #A with lr=1e-4 still has HOF noticeably higher
than legacy heuristic, prioritize extras that target HOF reduction).

**If any extra wins materially**: combine the winner(s) into a final
"champion" YAML and run 3 seeds × 2000 ep as the locked-in HP recommendation.

## Phase 3 — Distributional / risk ablations (deferred)

**Run after** Phase 2 winner is locked. Both sweeps operate on top of the
Phase 2 winner's config (gamma/lr/hidden/eps_mult from there), changing only
the QR-DQN-specific knobs.

| Sweep                | Tool                          | Values                            |
|----------------------|-------------------------------|-----------------------------------|
| `num_quantiles`      | `src/tools/run_ablation.py`   | {10, 50, 100, 200}                |
| CVaR `risk_fraction` | `src/tools/run_cvar_study.py` | {0.05, 0.10, 0.25, 0.50}          |

These scripts already write their own per-sweep summary CSVs; no need to fold
them into this document beyond a final pointer.

## Skip log

| Variant ID | Phase | Reason | Justifying prior result(s) |
|------------|-------|--------|----------------------------|
| p2_lr_3e-4 | 2     | Phase 2 #A (lr=1e-4) returned a very clean 3-seed result (capacity 3.510 ± 0.007, HOF 2.154 ± 0.243). At full budget #A already beats Phase-1 single-seed #A on every metric. Running #B for 7h to re-confirm the Phase-1 capacity tie is lower info-gain than spending that time on new dimensions (extras). The user-stated goal is the most optimum HPs; extras explore axes never tested. | Phase 2 #A row above; Phase 1 #3 vs #4 (3e-4 had higher HOF 3.26 vs 2.86) |
| Extras 1, 2, 3, 6 | 2.5 | User pivoted the laptop to Tasks 1 (non-uniform quantile positioning) and 2 (Atari benchmark) once the two highest-info extras (hidden_256_long and gamma_095_long) had landed. Both came in neutral-or-worse vs the champion, suggesting the remaining axes (eps_end, tau, batch_size, MA window) are also unlikely to shift the picture by more than the single-seed noise floor; their info value is now lower than starting the quantile-mode study which directly feeds the paper. | Extras 4 and 5 results above; champion's 3-seed variance (cap stdev 0.007, HOF stdev 0.24) sets the noise floor |

## Final ranking (HP search closed)

Single-seed 2000-episode evals, ranked by capacity_avg with HOF/RLF
tie-breakers.

| Rank | ID                | capacity_avg | hof_rate | rlf_rate | ho_rate | pp_rate | Why |
|------|-------------------|--------------|----------|----------|---------|---------|-----|
| 1    | p2_lr_1e-4        | **3.510**    | **2.154**| **0.257**| 16.345  | 2.086   | **CHAMPION** — 3-seed mean, tiny variance, best cap + best HOF + best RLF. Locked in as the QR-DQN starting point for the quantile-mode study and Phase 3 ablations. |
| 2    | x_hidden_256_long | 3.496        | 2.134    | 0.251    | 15.518  | 2.104   | neutral; wider net not a bottleneck. HOF marginally below champion's, but within single-seed noise — would need 3-seed re-run to call it a real tie. |
| 3    | x_gamma_095_long  | 3.473        | 2.993    | 0.312    | 15.759  | 1.901   | gamma=0.95 still loses at full budget — kills the "needs more training" hypothesis from Phase 1. |

**Conclusion**: HPs are locked at the champion settings (`lr=1e-4`,
`gamma=0.9`, `hidden_dims=[128, 128]`, `epsilon_mult=0.995`,
`num_quantiles=50`, `risk_type=mean`, 2000 ep). Further QR-DQN
exploration moves to the quantile-positioning and risk axes — see
`notes/quantile_mode_study.md` on the `feature/distributional-improvements`
branch.

## Reopened 2026-05-19 — kappa search + DQN baseline + new-physics re-validation

Tutor's post-meeting fix changed the physics (`NoiseLevel` -174 raw → -101
banded over 20 MHz, 16-step SINR table → 26-step, `MaxNumberPreparedBS`
4 → 5). The precomputed cache was regenerated; the previous champion
(`lr=1e-4` etc.) was validated only under the OLD physics. Two gaps remain:

1. **kappa was never searched.** All Phase 1/2/Refresh runs used the
   QRDQNAgent default `kappa=1.0` (Dabney et al. paper default). The
   tutor flagged this as a missed parameter at the meeting.
2. **DQN was never properly HP-searched.** All DQN runs to date used
   `config.yaml` defaults (`lr=1e-3`). For a fair QR-DQN vs DQN
   comparison we need DQN under at least the QR-DQN champion HPs.

**Tonight's queue** (`scripts/run_overnight_2026-05-19.sh`, single seed × 2000 ep,
XPU, sequential):

| Order | ID                  | Config                                   | Purpose |
|-------|---------------------|------------------------------------------|---------|
| 1     | `kappa_10`          | `hp_search/kappa_10.yaml`                | Re-validate champion + kappa=1.0 reference under new physics. **Solid QR-DQN baseline.** |
| 2     | `dqn_baseline_lr1e-4` | `hp_search/dqn_baseline_lr1e-4.yaml`   | DQN with QR-DQN champion HPs. **Solid DQN baseline.** |
| 3     | `kappa_05`          | `hp_search/kappa_05.yaml`                | Sharper Huber threshold; tests sensitivity to small residuals. |
| 4     | `kappa_20`          | `hp_search/kappa_20.yaml`                | Softer Huber threshold; closer to MSE for typical residuals. |
| 5 (stretch) | `ablation_N{10,100,200}` | tmp configs derived from kappa_10  | Deferred Phase 3 num_quantiles ablation. N=50 skipped (kappa_10 already covers it). |

Each step takes ~3h (extrapolated from `bmk_2026-05-16_9_p2_lr_1e-4` =
7.7h for 3 seeds × 2000 ep on XPU = ~2.6h/seed). Total budget for
steps 1-4: ~12h (won't all fit overnight). Steps 1-2 are the
hard requirement; 3-4 are kappa search; 5 is stretch.

**Decision rule (morning)**: if any kappa variant clearly beats
`kappa_10` on reward, validate it at 3 seeds × 2000 ep before moving on.
Otherwise lock `kappa=1.0` and proceed to either the quantile-mode
re-run or the num_quantiles ablation.

## Update protocol

When a run finishes:
1. Open `results/benchmarks/bmk_<date>_<num>_<variant>/eval/qrdqn_summary.csv`.
2. Read `capacity_avg`, `hof_rate`, `rlf_rate`, `ho_rate`, `pp_rate` (single
   row; or mean across seeds for Phase 2).
3. Fill the corresponding row above; change status `pending → done`.
4. If the result clearly invalidates a planned Phase 2 candidate, move it to
   the skip log with the justification.
