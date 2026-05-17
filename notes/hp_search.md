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
| 1 | x_eps_end_001     | `agent.epsilon_end: 0.01`                      | planned  |              |          |          |         |         | more committed greedy at end of training |
| 2 | x_tau_0005        | `agent.tau: 0.005`                             | planned  |              |          |          |         |         | slower target-net update; should reduce TD oscillation |
| 3 | x_batch_512       | `agent.batch_size: 512`                        | planned  |              |          |          |         |         | larger batch → smoother gradient updates |
| 4 | x_hidden_256_long | `agent.hidden_dims: [256, 256]`                | done     | 3.496        | 2.134    | 0.251    | 15.518  | 2.104   | neutral vs champion (within single-seed noise); wider net not a bottleneck even at 2000 ep |
| 5 | x_gamma_095_long  | `agent.gamma: 0.95`                            | planned  |              |          |          |         |         | revisit higher gamma at 2000 ep (lost at 500 ep, likely undertrained) |
| 6 | x_ma_window_25    | `ho_state.moving_average_window: 25`           | planned  |              |          |          |         |         | tighter state smoothing → faster reaction to channel changes |

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

## Final ranking

Filled at end of Phase 2.

| Rank | ID | capacity_avg | hof_rate | rlf_rate | ho_rate | pp_rate | Why |
|------|----|--------------|----------|----------|---------|---------|-----|
|      |    |              |          |          |         |         |     |

## Update protocol

When a run finishes:
1. Open `results/benchmarks/bmk_<date>_<num>_<variant>/eval/qrdqn_summary.csv`.
2. Read `capacity_avg`, `hof_rate`, `rlf_rate`, `ho_rate`, `pp_rate` (single
   row; or mean across seeds for Phase 2).
3. Fill the corresponding row above; change status `pending → done`.
4. If the result clearly invalidates a planned Phase 2 candidate, move it to
   the skip log with the justification.
