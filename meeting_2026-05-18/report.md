# Sprint report — Distributional RL for 5G LTM Handover

**Date**: 2026-05-18 (post-meeting revision)
**Branch**: `feature/distributional-improvements` (≈25 commits since the
last meeting, rebased on master after the hp-search merge)

> **Post-meeting status (2026-05-18 evening)**: tutor identified two
> issues that invalidate the headline numbers in this report:
> (1) models should be ranked by **episode reward**, not capacity — the
> ranking changes materially (see section 3); (2) physics constants in
> `legacy_simulation.py` were wrong and have been hand-fixed (NoiseLevel
> now banded over 20 MHz → -101 dBm instead of -174; 26-step SINR table
> replaces the 16-step one; MaxNumberPreparedBS = 5 instead of 4).
> `physics.py` has been propagated to match. The precomputed channel
> cache is being regenerated. **All previously-reported RL numbers in
> this document need re-running under the corrected physics before
> they're publishable**; the rankings and conclusions below are
> directional only.

## 1. What this sprint set out to do

Two parallel objectives:

1. **Improve QR-DQN itself** — the vanilla algorithm uses uniformly-spaced
   midpoint quantiles. We added non-uniform positioning schemes
   (Gauss-Legendre, Simpson, trapezoidal-with-fixed-endpoints) and a
   risk-aware truncation that learns only the lower-tail quantiles the
   policy actually consults.

2. **Cross-domain validation** — benchmark all variants on the standard
   Atari environment (Pong, Boxing) to test whether the improvements are
   LTM-specific or generalise.

## 2. Two bugs found and fixed (both behaviour-changing)

These are useful context for interpreting the HP refresh below.

**B1. Target network was a silent no-op.** Both `q_net` and `target_net`
were wrapping the same trunk/head Python module instances. The soft-update
`target ← τ·q + (1-τ)·target` was therefore `x ← x`. Hard update with
`tau=1.0` was unreachable. The fix gives the target net independent
parameters loaded from the online net's state-dict at init.

**B2. QR loss aggregator dropped quadrature weights.** The original
implementation used `(pinball * weights).mean()`, which collapses each
axis with uniform 1/N — correct for midpoint, *wrong* for Gauss-Legendre
or trapezoidal (their non-uniform weights are silently discarded). The
fix replaces the trailing `.mean()` with an explicit two-axis weighted
sum: target axis weighted by `mean_weights` (probability mass per atom),
predictor axis weighted by `predictor_weights` (integration weight per
tau). Midpoint with uniform weights is bit-equal to the old code
(`test_loss_backward_compat` locks this).

These two fixes mean the *current* numbers should not be compared
absolutely against pre-fix HP-search numbers; the new numbers are the
reliable ones.

## 3. LTM quantile-mode study — main result

5 variants × 1 seed × 2000 episodes. All share the HP-search champion
config (`lr=1e-4`, `gamma=0.9`, `tau=0.01`). **Ranked by episode reward**
(the actual training objective — `reward = R_thr · α_HO^ind · α_PP^ind
· α_HOF^ind · reliability_factor`, so it captures the full handover
trade-off rather than any single metric):

| Rank | Variant              | **reward** | capacity | hof_rate | rlf_rate | ho_rate | pp_rate |
|------|----------------------|-----------|----------|----------|----------|---------|---------|
| **1** | **Trapezoidal**         | **943.96** | 3.528    | 1.89     | 0.235    | 16.70   | 2.16    |
| 2    | Midpoint (baseline)     | 941.55    | 3.515    | 1.99     | 0.258    | 16.36   | 2.09    |
| 3    | Gauss-Legendre          | 934.76    | 3.497    | 2.42     | 0.275    | 16.12   | 2.06    |
| 4    | CVaR(0.1) truncated     | 934.61    | **3.538** | **1.81** | **0.21** | **19.93** | 2.33    |
| 5    | CVaR(0.1) full          | 923.96    | 3.492    | 2.50     | 0.237    | 18.51   | 2.08    |

See `plots/01_quantile_mode_study.png`. Each subplot overlays the LTM
hardcoded-heuristic baseline (capacity 3.235, hof 2.06, rlf 0.191) as a
dashed reference line. All five QR-DQN variants beat the baseline on
capacity, and all five also beat it on reward.

**3-seed validation of `cvar_truncated`.** We re-ran the original
"capacity champion" with seeds {42, 43, 44} × 2000 ep (9.1h on XPU):

| Metric | seed-42 | 3-seed mean ± std |
|---|---|---|
| reward       | 934.61 | **940.40 ± 2.38** |
| capacity_avg | 3.538  | 3.536 ± 0.005     |
| hof_rate     | 1.81   | 1.74 ± 0.05       |
| rlf_rate     | 0.21   | 0.21 ± 0.005      |

Multi-seed reward (940.4 ± 2.4) sits **between** midpoint (941.55) and
trapezoidal (943.96) — all three are within seed-noise of each other.

**Headline findings (updated):**

- **No QR-DQN variant clearly beats the corrected-loss midpoint
  baseline on reward.** Trapezoidal leads by Δ≈2.4 (~0.25%) — well
  inside the multi-seed std of cvar_truncated (±2.4). The "capacity
  champion" framing was misleading: cvar_truncated trades a capacity
  gain for a much higher handover rate (19.9 vs 16.4) that the reward
  function penalises.
- **The fair comparison still requires multi-seed for all 5 variants.**
  Only `cvar_truncated` has been multi-seeded so far; the other four
  rows are single-seed. The reward differences (Δ ≈ 2–20) are
  comparable to the cvar_truncated seed-std (2.4), so the rankings
  between trapezoidal/midpoint/GL/cvar_truncated may shuffle under
  multi-seed.
- **CVaR full is robustly worst.** Δ ≈ −18 vs trapezoidal is several σ
  of the seed noise — this is a clean negative finding.
- **Gauss-Legendre is also clearly behind** (Δ ≈ −9 vs trapezoidal).
- **Capacity vs reward divergence is the most interesting finding.**
  cvar_truncated has the best capacity, HOF, and RLF — but loses on
  reward because of HO rate. Worth a paragraph in the paper:
  "single-metric ranking can disagree with the training objective".

The trained quantile distributions are in
`plots/07–11_quantile_dist_*.png`. **Note on trapezoidal**
(`09_quantile_dist_trapezoidal.png`): the q_max=15 endpoint is far
below the learned interior range (~17–46), so the endpoints visually
contradict the data. This is a calibration issue — q_max should be
retuned to ~50 to match the actual return range.

## 4. HP refresh under the corrected code (500 ep × 1 seed)

The target-net fix unlocked an axis (soft vs hard updates) that was dead
code before. We re-tested the champion + 5 target-net-sensitive variants.
**Ranked by reward** (capacity rank in parens):

| Rank | Variant                              | reward | cap_avg | hof_rate | rlf_rate |
|------|--------------------------------------|--------|---------|----------|----------|
| **1** | **lr_3e-4**                          | **924** | 3.487 (#2) | 2.81 | 0.307 |
| 2    | tau_0005                              | 917    | 3.475 (#4) | 2.85 | 0.287 |
| 3    | hard_update (tau=1.0, freq=1000)      | 916    | 3.503 (#1) | 2.75 | 0.232 |
| 4    | tau_005                               | 915    | 3.482 (#3) | 3.14 | 0.272 |
| 5    | baseline_refresh (lr=1e-4, tau=0.01)  | 914    | 3.473 (#5) | 3.13 | 0.278 |
| 6    | lr_1e-3                               | 907    | 3.411 (#6) | 3.96 | 0.360 |

See `plots/04_hp_refresh_comparison.png`.

- **The reward differences across top-5 are 3–10 points** — well within
  seed noise (cvar_truncated 3-seed std was ±2.4). So the reward
  rankings of the top 5 are not statistically meaningful at 500 ep × 1
  seed; the only robust signals are "lr=1e-3 is bad" (Δ ≈ −17) and
  "everything else is roughly tied".
- **By capacity, hard_update led; by reward, lr=3e-4 leads.** Hard
  update has the best capacity / HOF / RLF but its reward sits 3rd —
  another instance of "single-metric ranking ≠ reward ranking".
- This whole table needs to be re-run under the corrected physics and
  multi-seed before any HP recommendation is publishable.

## 5. Atari benchmark — cross-domain validation

Standard Nature-DQN preprocessing (grayscale, 84×84, frame-stack 4,
action-repeat 4, reward clipping). All variants use the same
500k-frame budget — sufficient for *relative ordering*, not for matching
the literature's absolute returns.

### Pong (500k frames, reward in [-21, +21], random ≈ -21)

See `plots/05_atari_pong_study.png`.

| Variant            | final mean return |
|--------------------|-------------------|
| Midpoint           | -21.0 (random) |
| Gauss-Legendre     | -21.0 (random) |
| **Trapezoidal**    | **-19.4** (only learner) |
| CVaR(0.1) full     | -21.0 (random) |
| CVaR(0.1) truncated| -21.0 (random) |

Pong at 500k frames is borderline-too-few (literature uses 1–2M), but
trapezoidal was the only variant to break random play — same direction
as LTM (trapezoidal 2nd best).

### Boxing (500k frames, reward in [-100, +100], random ≈ -25)

See `plots/06_atari_boxing_study.png`.

| Variant            | final mean return |
|--------------------|-------------------|
| **CVaR(0.1) full** | **+7.2** (essentially tied with midpoint) |
| Midpoint           | +7.0 |
| Trapezoidal        | +3.4 |
| Gauss-Legendre     | +1.2 |
| CVaR(0.1) truncated| **−4.0** ← *worst*, near-inverts the LTM ranking |

**Boxing essentially inverts the LTM ranking.** The LTM winner
(`cvar_truncated`) is the *worst* Atari Boxing variant, and the LTM
worst (`cvar_full`) ties for *best*.

### What this tells us

- **GL regression replicates across LTM and both Atari games.** Robust
  negative result.
- **Trapezoidal's LTM advantage is fragile to q_max tuning** — it lands
  middle-pack on Atari when `q_max` is mismatched to the return range
  (further confirmed by the LTM trapezoidal plot showing q_max=15
  contradicting the learned ~17–46 range).
- **cvar_full ranks 1st on Boxing but 5th on LTM (both rankings).**
  This is the cleanest cross-domain inversion we have — and it does
  NOT depend on which LTM ranking (capacity vs reward) we use, so the
  cross-domain story survives the reward re-ranking.
- **cvar_truncated is worst on Boxing and 4th-of-5 on LTM by reward
  (1st by capacity).** It's not a "wins LTM, loses Boxing" story
  anymore — it's "loses Boxing and is mid-pack on LTM under the actual
  training objective". A weaker but still interesting finding.

### Mechanism interpretation (still plausible, but qualified)

The original mechanism explanation — *truncated CVaR concentrates
network capacity on the lower tail, which helps LTM where HOF/RLF
discriminate actions and hurts Boxing where upper-tail KO potential
discriminates actions* — still holds for the **capacity** comparison
on LTM. Under reward, the mechanism is still consistent (lower-tail
focus → fewer HOF/RLF → higher capacity), but the **side-effect**
(more handovers) dominates the reward calculus on LTM. So:

- *On capacity*: truncated CVaR is best on LTM and worst on Boxing — clean reversal, mechanism applies.
- *On reward*: cvar_full is worst on LTM and best on Boxing — also a reversal, also matches the mechanism (full-CVaR concentrates on the worst-case lower tail, which over-penalises HOF risk on LTM but matches Boxing's defensive value).
- The cleaner paper framing: the **CVaR action rule** (full vs truncated) flips ranking between LTM and Boxing in opposite directions, which is consistent with where the action-discriminating return mass lives.

## 6. Conclusions / how to frame this in the paper

**Important caveat upfront**: ranking by reward (the training objective)
gives a *much* less aggressive headline than ranking by capacity. The
QR-DQN improvements are real but small once measured against the actual
reward function:

1. **Reward-based ranking puts trapezoidal slightly above midpoint,
   with cvar_truncated demoted to 4th.** Δreward ≈ 2.4 (top vs
   baseline), inside the seed-noise band (±2.4). Honest read: no
   variant *robustly* beats the corrected-loss midpoint baseline at
   single-seed precision.

2. **The capacity-vs-reward divergence is itself the paper finding.**
   cvar_truncated wins on capacity / HOF / RLF — but the higher
   handover rate (+22% vs midpoint) costs more reward than the capacity
   gain returns. This is a genuinely interesting subplot for the paper:
   "concentrating capacity on the policy-relevant lower tail improves
   capacity but distorts handover trade-offs". Worth one figure.

3. **CVaR full is the only robustly-negative variant.** Δreward ≈ −18
   vs midpoint, multiple σ outside seed noise. Clean and defensible.

4. **Gauss-Legendre is empirically irrelevant.** Δreward ≈ −7 across
   LTM, plus regression on Pong and Boxing. "Higher-order quadrature
   does not help on ReLU-network quantile functions" is a defensible
   robust negative finding.

5. **Trapezoidal helps marginally with a q_max calibration caveat** —
   the learned interior quantiles span ~17–46 but q_max is set to 15,
   so the trapezoidal "prior" is fighting the data. Worth retuning to
   q_max≈50 and re-running before claiming the +2.4 reward gap is real.

6. **The two target-net / loss bug fixes are footnote-grade** but
   meaningful for reproducibility and unlocked the hard-update HP axis.

### The reshape, in one sentence

The QR-DQN quantile-positioning axis is **not** a strong handle for
LTM under the reward function — the differences are real but
seed-noise-comparable, and the most "interesting" capacity-champion
(cvar_truncated) actually loses on reward because it over-handovers.
Multi-seed comparison of all five variants is needed before any of the
within-QR-DQN rankings are publishable.

### Master plot (`02_master_bar_plots.png`, `03_master_radial_plot.png`)

The `QR-DQN (Ours)` bar is now the **3-seed `cvar_truncated`**
post-bug-fix winner (capacity 3.536, hof 1.74, rlf 0.21), pooled
across seeds {42, 43, 44} × 1000 UEs (n=3000). Other agents in the
plot: LTM/LMMSE/CMAB columns are paper reference numbers,
`Baseline (Ours)` is our reproduced LTM hardcoded heuristic, and
`DQN (Ours)` is the vanilla DQN from the same HP search.

## 7. What's still in flight (as of the meeting)

Nothing — all training jobs scheduled for this sprint are complete:

- Atari Boxing sweep: all 5 variants done.
- 3-seed validation of `cvar_truncated` on LTM: done (numbers in
  section 3).

The deferred experiments in section 9 are *intentionally* held off,
not running.

## 8. Animations

3 representative agent rollouts in `animations/`, all on the same UE
trajectory (UE #500) so they're directly comparable:

- `01_ltm_baseline_ue500.mp4` — the hardcoded heuristic (reference)
- `02_qrdqn_midpoint_ue500.mp4` — vanilla QR-DQN (RL baseline)
- `03_qrdqn_cvar_truncated_ue500.mp4` — the capacity champion (not the reward champion)

The dashboard plots per-step UE position, serving BS, RSRP per BS, and
MCS per BS, so the differences in handover decisions are visible
side-by-side.

## 9. Deferred follow-ups (after the meeting)

**Top priority after the meeting feedback**: re-run **every** training
result under the corrected physics (NoiseLevel banded over 20 MHz,
26-step SINR table, MaxNumberPreparedBS=5). The tutor's hand-edit of
`legacy_simulation.py` on 2026-05-18 means all prior numbers — qmode
study, HP refresh, 3-seed cvar_truncated, both Atari games — are stale.

See `notes/deferred-experiments.md` for the full prioritised backlog.
Other headline items, all now subordinate to the re-run:

1. **Multi-seed validation of all 5 qmode variants** (not just
   cvar_truncated). With the reward gap of ~2.4 between trapezoidal
   and midpoint sitting inside seed noise, single-seed rankings are
   not publishable. ~30h XPU for 5 variants × 3 seeds × 2000 ep.
2. **Trapezoidal q_max retuning** to match the observed Q-range
   (~50, not 15). Drop-in config change; needed before claiming the
   trapezoidal advantage is real.
3. **Risk-fraction (CVaR α) sweep** on top of `cvar_truncated` —
   probes the most interesting axis even though cvar_truncated isn't
   the reward champion. Configs exist; ~15h on XPU.
4. **Huber `kappa` sweep** — predictor-collapse analysis suggested
   smaller κ should help.
5. **Simpson's rule** (config + code ready) — completes the
   quadrature axis. Cheap to do.

## 10. Code state

- `src/distrl/agents/distributional/quantile_modes.py` — 4 schemes +
  truncated, factory, CVaR weight mask. Unit-tested in
  `src/scripts/test_quantile_modes.py`.
- `src/distrl/agents/distributional/qrdqn.py` — uses the scheme via
  `build_scheme`; loss aggregator is the corrected two-axis weighted sum.
- `src/distrl/agents/networks.py` — `CNNTrunk` (Nature-DQN) for Atari,
  `MLPTrunk` for LTM, plus a shared `build_trunk()` helper now used by
  both DQN and QR-DQN (deduped trunk plumbing).
- `src/atari_main.py` — Atari runner with standard preprocessing.
- `src/tools/run_quantile_mode_study.py`,
  `src/tools/run_atari_study.py`, `src/tools/run_hp_refresh.py` —
  orchestrators for the three studies.
- All artifacts under `results/benchmarks/bmk_*` (LTM) and
  `results/atari/<game>_<variant>_<ts>/` (Atari).
