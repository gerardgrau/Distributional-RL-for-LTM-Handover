# No-gate ablation findings (2026-06-03)

Env: no-gate (learned_trigger + event_driven), reform+(a), 1000 ep × 3 seeds,
eval on all 1,000 UEs, ε=0. Ranking scalar = **reward per action** = total
episode reward ÷ n_decisions (de-confounds the event-driven cadence). All 8
KPIs reported. Numbers below are best-checkpoint re-evals (seed-mean cross-
checked against per-seed summaries; consistent to ±0.1 on tails).

---

## 0. Why reward-per-action, re-confirmed empirically

Truncated CVaR α=0.10 has the **highest total reward of any run (5331)** yet the
**lowest reward-per-action (3.534)** — it simply triggers more decisions
(n_dec=1509 vs ~1330). Total reward would rank the most risk-averse agent #1;
reward-per-action correctly ranks it last. This is the cadence confound the
metric was chosen to remove.

---

## 1. N-sweep (risk-neutral, midpoint, mean policy) → N PENDING tie-breaker

CORRECTION: an earlier single-best-checkpoint table credited N=25 (HO 4.72,
PP 0.668) and I wrongly wrote "N buys nothing past 10". The **seed-averaged**
view (3 seeds, mean±std) tells a different story:

| metric | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| capacity | 4.066±.003 | 4.068±.001 | 4.069±.001 | 4.069±.002 |
| RLF | 0.047±.002 | 0.048±.003 | 0.046±.001 | 0.049±.002 |
| HO  | 5.18±.10 | 5.00±**.37** | **4.87±.15** | 4.99±.13 |
| PP  | 0.928±.03 | 0.833±**.18** | **0.779±.04** | 0.876±.07 |
| HOF | 0.524±.02 | 0.501±.05 | **0.489±.02** | 0.487±.02 |
| reliab% | 97.068 | 97.077 | 97.087 | 97.080 |

- **N=25's "win" was a lucky checkpoint.** Its seeds were HO {5.52,4.72,4.77},
  PP {1.074,0.662,0.762} — highest-variance setting; the good number was one
  seed. Does not survive seeding.
- **Small real tail gain at N≈50**: PP 0.779±.04 vs N=10 0.928±.03 (~4σ, real),
  HO 4.87 vs 5.18 (~2σ), HOF 0.489 vs 0.524. Plateaus by N=100 (PP regresses to
  0.876). **reward/action, capacity, reliability are flat in N.**

WALL-TIME (isolated QR-DQN training, mean of 3 seeds — N drives only this; DQN
+ eval stripped out): N=10 31.2 min, N=25 37.1 (1.19×), N=50 73.3 (2.35×),
N=100 182.2 (5.85×). Super-linear (O(N²) QR Huber loss on a ~31-min fixed
env-stepping cost): N=25 is cheap (+19%), N=50 is the real ~2.35× hit. The
*dual-agent total* (36/42/79/188 min) understates this — it includes a
fixed-cost DQN independent of N.

DECISION: **PENDING the 6-seed tie-breaker** (do not treat as locked). 3-seed
means with N=20 added (N=20 = bmk_03_1, the α=1.0 truncated run ≡ midpoint N=20
mean, 3 seeds):

| metric | N=10 | N=20 | N=25 | N=50 | N=100 |
|---|---|---|---|---|---|
| HO  | 5.18±.10 | **5.24±.09** | 5.00±.37 | **4.87±.15** | 4.99±.13 |
| PP  | 0.928±.03 | 0.918±.06 | 0.833±.18 | **0.779±.04** | 0.876±.07 |
| HOF | 0.524±.02 | **0.555±.02** | 0.501±.05 | **0.489±.02** | 0.487±.02 |

- **N=20 is tail-WORST** (highest HO + HOF; PP ~tied N=10). Its single-best-ckpt
  number (PP 0.849) was a lucky seed — same trap as N=25 (PP 0.668). ALWAYS
  seed-average. N=20's case is **unification + cheap**, NOT tail quality.
- **N=50 is tail-BEST** (~2-3σ better than N=20), tight. 2.35× train cost.
- reward/action, capacity, reliability stay FLAT in N.

The 10/25/50 tie-breaker (seeds 45/46/47) → 6 seeds tests whether N=50's tail
edge is robust. THEN decide: **N=50 (tail-best, 2× cost, breaks unified grid) vs
N=20 (unified, cheap, tail-mediocre)** — paper leans on tails, so N=50 is live.
Quadrature + HP configs PROVISIONALLY at N=20 (simpson N=21); finalize when the
verdict lands (the first N-dependent run starts ~when the tie-breaker finishes,
so no commitment is lost by waiting).

---

## 2. Risk-return frontier — truncated CVaR α-sweep (N=20)

| α | reward/action | capacity | RLF | HO | PP | HOF | reliab% | resv% |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 3.534 | 3.970 | 0.046 | **4.15** | **0.479** | **0.384** | 96.75 | 9.46 |
| 0.25 | 3.582 | 4.020 | **0.042** | 4.47 | 0.679 | 0.383 | 96.93 | 9.44 |
| 0.50 | 3.621 | 4.047 | 0.045 | 4.73 | 0.821 | 0.452 | 96.98 | 9.19 |
| 0.75 | 3.650 | 4.068 | 0.048 | 5.02 | 0.970 | 0.465 | 97.05 | 9.09 |
| 1.00 | 3.641 | 4.060 | 0.048 | 5.11 | 0.849 | 0.522 | 97.06 | 8.99 |

**Monotone dial.** Lowering α (more risk-averse) monotonically **improves the
tails** (HO, PP, HOF↓) and **costs reward + capacity + reliability**, and holds
slightly more candidates ready (resv↑). There is **no interior reward optimum**:
reward/action rises with α up to ~0.75 then flattens. The "contribution" is the
*controllable frontier*, not a single winning α.

**Operating point = α≈0.25 (the knee).** It captures ~the best tails (HO 4.47,
PP 0.68, HOF 0.38 — within noise of α=0.10's) **and the best RLF of any run
(0.042)** for only a ~1.9% reward-per-action cost vs risk-neutral. Pushing to
α=0.10 buys marginal extra HO/PP for a steeper capacity/reliability/reward hit
(diminishing returns).

> NOTE for the paper abstract: this **replaces** the old "interior risk optimum
> at α≈0.5" claim. The de-confounded frontier is monotone; α is a dial, and we
> *choose* α≈0.25 as a risk-averse operating point, not because reward peaks there.

---

## 3. DQN < RN < RA hierarchy (the paper's core story)

Cleanest reward-tied pair — DQN vs risk-neutral QR-DQN (RN, N=10):

| agent | reward/action | capacity | RLF | HO | PP | HOF | reliab% |
|---|---|---|---|---|---|---|---|
| DQN scalar  | 3.646 | 4.060 | 0.051 | 5.62 | 1.082 | 0.715 | 97.00 |
| RN (QR mean)| 3.650 | 4.064 | 0.043 | 5.15 | 0.918 | 0.548 | 97.06 |
| RA (α=0.25) | 3.582 | 4.020 | 0.042 | 4.47 | 0.679 | 0.383 | 96.93 |

- **RN Pareto-dominates DQN**: at tied reward/action it is better on *every*
  KPI (HO, PP, HOF, RLF, capacity, reliability). The distributional
  representation cuts the tails **for free** (risk-neutrally).
- **RA trades reward for tails**: ~1.9% reward/action + small capacity/
  reliability cost to nearly halve HOF (0.72→0.38) and cut PP (1.08→0.68) and
  HO (5.62→4.47) vs DQN.
- Reward/action across DQN/RN/RA-highα ≈ tied (~3.65); the ordering lives in
  the tails. ✔ matches the abstract's "DQN beats LTM, distributional extends
  the lead on tail metrics."

---

## 4. Pending (training now, 1000ep×3seed, no-gate)

- **Truncated vs full-grid CVaR** (α=0.10/0.25/0.50, N=20): does truncation
  match/beat standard CVaR? → Marga §IV-D contribution claim.
- **Quadrature/positioning** (GL, trapezoidal, simpson, beta_equal,
  beta_weighted vs midpoint, N=10 mean): the "Quantile-Positioning Schemes"
  study + non-uniform-quantile angle.
- **HP** (lr 5e-5/2e-4, κ=1.0, γ=0.95): backbone robustness.
