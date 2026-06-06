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

---

## 5. Truncated vs full-grid CVaR (Marga §IV-D) — PARTIAL (α=0.5 pending)

Seed-mean (3 seeds), no-gate N=20. Truncated = predicts only k=⌈αN⌉ tail
quantiles; Full = predicts all 20, CVaR over the bottom k (same tail τ).

| | cap | HO | PP | HOF | RLF | rel% | rew/act |
|---|---|---|---|---|---|---|---|
| trunc α=0.10 | 3.988 | 4.22 | 0.534 | 0.344 | 0.043 | 96.78 | 3.534* |
| full  α=0.10 | 4.044 | 4.44 | 0.671 | 0.405 | 0.041 | 97.01 | 3.595 |
| trunc α=0.25 | 4.025 | 4.54 | 0.745 | 0.385 | 0.043 | 96.91 | 3.582* |
| full  α=0.25 | 4.053 | 4.75 | 0.782 | 0.463 | 0.042 | 97.05 | 3.614 |

(*truncated rew/act = reeval_old best-ckpt; full = auto-eval seed-mean — slight
method mismatch; 8 KPIs are both clean seed-mean.)

FINDING: at α=0.10/0.25 truncated and full are a **TRADEOFF, not dominance** —
truncated has better tails (HO/PP/HOF↓) but lower capacity/reliability/reward.
The truncated estimator behaves *more risk-averse* than full at the same nominal
α (sharper tail estimate → more conservative policy). This TENSIONS with the
paper §IV-D claim that truncated is strictly better (higher reward AND lower
failures) — BUT that claim is stated at α=0.5, and the α=0.5 full run
(cvarfull_a050) is still training. DO NOT edit §IV-D until the α=0.5 pair
(trunc bmk_19 vs full cvarfull_a050) is in. If the tradeoff holds at α=0.5 too,
§IV-D needs reframing from "strictly better" to "sharper tail estimate / more
risk-averse at equal α".

---

## 6. Truncated vs full CVaR at N=25 (decided backbone) — α=0.25 done, α=0.5 pending

Both seed-mean (3 seeds), reward/action clean for both (auto-eval has n_decisions).

| α=0.25 | cap | HO | PP | HOF | rel% | rew/act | head |
|---|---|---|---|---|---|---|---|
| truncated (k=7) | 4.019 | 4.33 | 0.627 | 0.367 | 96.90 | 3.598 | 7 outputs |
| full (N=25)     | 4.057 | 4.44 | 0.644 | 0.415 | 97.08 | 3.602 | 25 outputs |

At N=25, α=0.25: **reward/action TIED** (3.598≈3.602, vs N=20 where full led ~0.9%);
**truncated wins all tails** (HOF 0.367<0.415 ~12%, HO 4.33<4.44, PP 0.627<0.644);
full keeps a hair more capacity/reliability. So truncated ≥ full at the operating
point — better failures at tied reward with a 7-vs-25 head (3.5× cheaper). Stronger
than the N=20 tradeoff (§5): N=25 closed the reward gap while keeping the tail edge.
SUPPORTS §IV-D. STILL PENDING: α=0.5 pair (the paper's stated claim point) —
cvarfull_a050 + trunc_a050 running. Do not finalize §IV-D wording until that lands.

### 6b. α=0.5 truncated-vs-full at N=25 — THE VERDICT FLIPS (⚠️ §IV-D claim wrong)

| α=0.50 | cap | HO | PP | HOF | rel% | rew/act | head |
|---|---|---|---|---|---|---|---|
| truncated (k=13) | 4.048 | 4.76 | 0.826 | 0.442 | 97.00 | 3.613 | 13 |
| full (N=25)      | 4.065 | 4.68 | 0.735 | 0.432 | 97.09 | 3.633 | 25 |

At α=0.50 FULL is better on ~everything (rew/act 3.633>3.613, PP 0.735<0.826,
HO 4.68<4.76, cap, rel). OPPOSITE of α=0.25 (where truncated won). So truncated-
vs-full is **α-DEPENDENT and FLIPS**: truncated helps at low α (k=7, big 25→7
saving), hurts at α=0.5 (k=13, modest saving + loses the upper-lobe info).

⚠️ **THE PAPER'S §IV-D CLAIM IS WRONG.** It states truncated is "significantly
more accurate … at CVaR_0.5" — but at α=0.5 (N=25, no-gate) full-grid WINS.
ACTION: reframe §IV-D from "universally better" to "truncated helps at the
aggressive low-α operating point (α=0.25) we adopt; the advantage shrinks and
reverses by α=0.5." The RA-final (truncated α=0.25, k=7) is still justified — it's
≥ full at the operating point AND 3.5× cheaper. (3 seeds; α=0.10 pair pending to
complete the picture. Do NOT edit §IV-D prose until α=0.10 lands + user reviews.)

### 6c. COMPLETE truncated-vs-full at N=25 (all 3 alpha) — coherent story

| alpha | trunc PP | full PP | trunc HOF | full HOF | trunc rew | full rew | winner |
|---|---|---|---|---|---|---|---|
| 0.10 (k=3)  | 0.498 | 0.691 | 0.353 | 0.440 | 3.541 | 3.592 | trunc TAILS (big), full reward |
| 0.25 (k=7)  | 0.627 | 0.644 | 0.367 | 0.415 | 3.598 | 3.602 | TRUNC (tied reward, better tails) |
| 0.50 (k=13) | 0.826 | 0.735 | 0.442 | 0.432 | 3.613 | 3.633 | FULL (everything) |

UNIFYING EXPLANATION: the truncated head acts as if at a LOWER effective alpha
than nominal -> systematically MORE RISK-AVERSE. reward/action always slightly
lower (-0.05/-0.003/-0.02); tail bonus huge at a=0.10, small at a=0.25, negative
by a=0.50. So it is NOT "same policy, sharper estimate" (the paper's framing) —
it SHIFTS the policy conservative. Helps in aggressive-risk regime, hurts in mild.

PAPER §IV-D REWRITE (honest, still positive at the operating point):
"At the aggressive low-alpha operating point we adopt (alpha=0.25), the truncated
head matches full-grid reward-per-action with better tail metrics (PP, HOF, HO)
using a 3.5x smaller head (k=7 vs N=25). The advantage is regime-specific: it is
largest as alpha->0 (k=3 at 0.10) and reverses by alpha=0.5, where the full grid's
extra information wins." DROP the "universally / strictly more accurate at CVaR_0.5"
claim. RA-final = truncated alpha=0.25 remains the right operating choice.

---

## 7. Quantile-positioning / quadrature @ N=25 mean (seeds 42-44) — COMPLETE

| scheme | rew/act | cap | HO | PP | HOF | rel% |
|---|---|---|---|---|---|---|
| midpoint (vanilla) | ~3.65 | 4.068 | 5.00 | 0.833 | 0.501 | 97.08 |
| gauss_legendre | 3.658 | 4.068 | 4.86 | 0.725 | 0.516 | 97.08 |
| trapezoidal | 3.654 | 4.068 | 5.15 | 0.946 | 0.524 | 97.07 |
| simpson | 3.658 | 4.070 | 5.25 | 0.942 | 0.558 | 97.05 |
| beta_equal | 3.654 | 4.072 | 5.40 | 1.055 | 0.551 | 97.08 |
| beta_weighted | 3.656 | 4.074 | 5.29 | 0.984 | 0.547 | 97.09 |

(1) reward/action, capacity, reliability POSITIONING-INDEPENDENT (all tied) ->
vanilla MIDPOINT is the right default; GL ties within noise; fancier schemes are
equal-or-worse. (2) Tails order: midpoint≈GL (best) > trapezoidal/simpson > beta
(WORST). Center-clustered non-uniform (beta) STARVES THE TAILS where HO/PP/HOF
risk lives -> worst tails. Ties into §IV-D: non-uniform allocation must be
TAIL-ward (truncated CVaR helps), not CENTER-ward (beta hurts). "Spend the
quantile budget where the policy reads it — the lower tail." (3 seeds; extremes
GL-best / beta-worst robust, fine distinctions noisy.) Finals use midpoint. ✓

---

## 8. HP robustness @ N=25 mean (each varies one knob vs RN backbone) — COMPLETE

| variant | rew/act | cap | HO | PP | HOF | rel% |
|---|---|---|---|---|---|---|
| RN ref (lr1e-4,k.25,g.9) | ~3.65 | 4.068 | 5.00 | 0.833 | 0.501 | 97.08 |
| lr=5e-5 | 3.641 | 4.072 | 5.13 | 0.957 | 0.505 | 97.10 |
| lr=2e-4 | 3.656 | 4.066 | 5.08 | 0.808 | 0.547 | 97.07 |
| kappa=1.0 | 3.657 | 4.069 | 5.14 | 0.905 | 0.538 | 97.08 |
| gamma=0.95 | 3.627 | 4.055 | 6.74 | 1.797 | 0.756 | 96.90 |

lr (5e-5/1e-4/2e-4) and kappa (0.25/1.0) are ROBUST — all within noise of the RN
ref. **gamma=0.95 is clearly WORSE** (HO 6.74 vs 5.00, PP 1.80 vs 0.83, HOF 0.76
vs 0.50): higher discount = more far-sighted = chases marginal future SINR =
handover-churn explosion. So the backbone (lr1e-4, kappa0.25, **gamma=0.9**) is
validated; gamma=0.9 is the right choice, 0.95 is bad. N=25 CAMPAIGN COMPLETE.
