# Conclusions from the DQN ↔ QR-DQN experiments (for the paper)

Date: 2026-05-31. Synthesises the de-confounded N/α sweeps, the quantile-bridge
mechanism study (1000 ep × 3 seed), and the 2000 ep × 5 seed 3-way finals.

**Budget note.** Sweeps and the bridge study are 1000 ep × 3 seed (ablation
budget) — use them for *relative* trends. Absolute headline numbers come from the
2000 ep × 5 seed finals. Significance judged vs cross-seed std: SE = std/√n,
SE_diff = √(SE₁²+SE₂²); "real" ≈ ≥2σ.

---

## The design space

QR-DQN with truncated-CVaR has two independent knobs:
- **N** = number of quantiles the distribution is represented with (resolution).
- **α** = CVaR risk fraction; action selection averages the bottom-α of the return.
- **k = ⌈N·α⌉** = quantiles actually predicted (the truncated head outputs only these).

The headline trick was to **de-confound N and α** by holding effective k fixed
while sweeping α (via N = ⌊k/α⌋). Earlier "reward ∝ α" was an artifact of α also
changing k.

---

## Finding 1 — Quantile count saturates fast (N is a plateau)

N-sweep at α=0.50 (reward, 1000 ep × 3 seed): k=10 **1083.8** | k=13 1084.6 |
k=15 1083.4 | k=25 1082.4 | k=50 1083.1. Flat within noise (spread 2.2 < CI).
→ **~10 quantiles suffice; more add cost, not performance.** We adopt k=10.

## Finding 2 — Risk level matters, with an *interior* optimum (α does matter)

De-confounded α-sweep at fixed k=10 (reward / hof):
α=0.30 1081.2 / 1.18 · α=0.40 1081.5 / 1.26 · **α=0.50 1083.8 / 1.28** ·
α=0.70 1081.2 / 1.41 · α=1.00 (risk-neutral) **1074.6 / 1.83**.
→ Reward is an **inverted-U peaking at α≈0.5**; risk-neutral (α=1) is −9.2 (~7σ)
worse. Safety (hof, rlf) improves monotonically as α→0, but reward tops out at the
interior because over-caution costs handovers. **α matters; α* = 0.5.**

## Finding 3 — N=1 risk-neutral QR-DQN ≈ DQN (the lower anchor)

Single median quantile vs the matched DQN (1000 ep × 3 seed):
rn_N1 **1063.8 ± 5.8** (hof 2.47) vs dqn_soft **1060.9 ± 3.3** (hof 2.61).
Δ = +2.9 reward = **0.76σ → statistically indistinguishable.**
→ Confirms the theory: one quantile at τ=0.5 with mean selection *is* (median-)DQN;
the pinball loss at τ=0.5 reduces to a symmetric ½-Huber. **The distributional
machinery adds nothing at N=1** — QR-DQN's edge must come from N>1 and/or risk.

## Finding 4 — The distributional gain emerges as N grows (risk-neutral, Row A)

Risk-neutral reward vs N: N1 1063.8 → N2 1066.8 → N3 1072.3 → N5 1075.2 →
N10 1075.4 (hof 2.47 → 1.72). Smooth climb from DQN-level to a plateau by N≈5,
**+11–12 reward.** → Adding quantiles helps *even risk-neutral*; it's a genuine
multi-quantile effect, saturating around 5.

## Finding 5 — Risk-aversion helps even with a SINGLE quantile (Row B) ★

Take one quantile and slide it from the median into the lower tail (pure VaR):
τ=0.50 (=rn_N1) 1063.8 / hof 2.47 / ho 15.0 →
**τ=0.25 1078.2 / 1.54 / 16.8** → **τ=0.10 1077.0 / 1.30 / 18.7** →
**τ=0.05 1069.0 / 1.27 / 19.8** (3 seeds).
- Single-median → single-VaR(τ=0.10): **+13.2 reward (3.7σ)**, hof **2.47 → 1.30**
  (nearly halved, ≈ the full champion's 1.28).
- A **single VaR quantile (1077) matches ten risk-neutral quantiles (rn_N10 1075.4)
  on reward and beats them on safety** (hof 1.30 vs 1.72).
- The τ-sweep **recapitulates the α-sweep**: interior τ≈0.10–0.25 is best, τ=0.05 is
  over-cautious (ho 19.8, reward falls to 1069 — 3σ below τ=0.10).
→ **Risk-aversion is primarily about *where* you evaluate the return distribution,
not how finely you represent it.** A single well-placed (lower-tail) quantile
captures most of the safety benefit.

## Finding 6 — The two gains are separable and roughly additive

- Distributional axis (more quantiles, risk-neutral): DQN/N1 → N10 ≈ **+11–14**.
- Risk axis (lower quantile / lower α): median → VaR/CVaR ≈ **+13–14**.
- They combine in the champion (k=10, CVaR₀.₅ ≈ 1084 at 1000 ep).
Canonical 2000 ep × 5 seed 3-way (k=13): DQN **1068.0** → RN **1081.4**
(+13.4, 9.6σ — distributional) → RA **1086.0** (+4.6, 5.1σ — risk); hof
2.23 → 1.50 → 1.15. [k=10 finals running; expected similar.]
Decomposition: the single VaR quantile already delivers the *safety* (hof ≈ 1.3);
the full 10-quantile CVaR adds ~+7 more reward on top (capacity/stability).

## Finding 7 — Behavioural mechanism of risk-aversion

As the agent gets more risk-averse (lower τ / lower α): **HO rate ↑** (15.0 → 18.7
→ 19.2), **hof ↓, rlf ↓, capacity ↑**. The risk-averse policy **hands over more
proactively to pre-empt radio-link failures** — trading handover overhead for
reliability. The reward optimum is where that trade-off balances (interior τ/α).

## Finding 8 — Target-update scheme is within-noise (backbone justification)

{DQN,QR}×{hard,soft} at matched budget: dqn_hard 1062.6 vs dqn_soft 1060.9 (0.5σ);
rn_N10 soft 1075.4 vs hard 1075.9 (within noise). → τ (hard vs soft) is not a
significant axis for either agent; we standardise on **soft τ=0.01 for all agents**
so the DQN-vs-QR comparison is strict apples-to-apples (only the value head differs).

---

## Headline conclusions for the paper

1. **QR-DQN substantially outperforms DQN on LTM handover, and the gain has two
   separable sources:** the distributional representation (N>1) and risk-aware
   action selection (CVaR/VaR). Strip QR-DQN to a single median quantile and it
   collapses to DQN.
2. **Risk-aversion is the dominant lever for reliability** — it roughly halves
   handover failures, and remarkably, even a *single* lower-tail quantile captures
   most of this. Risk-aversion is about the *evaluation point*, not the resolution.
3. **Both axes have interior optima:** resolution plateaus by ~10 quantiles; risk
   level is an inverted-U peaking at α≈0.5 (equivalently τ≈0.1–0.25 for k=1). Too
   risk-neutral *and* too risk-averse both underperform.
4. **The risk-aware agent earns its reliability by handing over more proactively** —
   an interpretable, operationally sensible policy, not a black-box gain.

## Caveats to state in the paper

- Bridge study = 1000 ep × 3 seed (mechanism/relative); canonical numbers from the
  2000 ep × 5 seed finals (all bridge points are 3-seed).
- Evaluation uses all 1000 trajectories for train and eval (no split) — matches the
  prior paper's protocol; disclose as a limitation.
- κ=0.25 selected on an earlier backbone, not re-validated on the final shared one
  (low risk).

---

## ★ CANONICAL 3-WAY FINALS — k=10, 2000 ep × 5 seed, shared soft backbone

Supersedes the k=13 numbers in Finding 6 (they match → k-plateau holds at full
budget). All three on the identical backbone (soft τ=0.01, h=[256,256], tf=1,
κ=0.25, max_prep=5); only the value head differs. σ = Δ / √(σ²ₐ/5 + σ²_b/5).

| metric (per-min unless noted) | DQN-soft | QR-RN | QR-RA | DQN→RN | RN→RA |
|---|---|---|---|---|---|
| **Reward** | 1067.40 ± 2.08 | 1081.53 ± 1.74 | 1086.31 ± 1.16 | +14.1 (11.7σ) | +4.8 (5.1σ) |
| **HOF** ↓ | 2.233 ± 0.143 | 1.432 ± 0.134 | 1.176 ± 0.077 | −0.80 (9.1σ) | −0.26 (3.7σ) |
| **RLF** ↓ | 0.143 ± 0.008 | 0.107 ± 0.006 | 0.093 ± 0.007 | −0.036 (7.9σ) | −0.014 (3.2σ) |
| **Capacity** ↑ | 3.909 ± 0.008 | 3.951 ± 0.006 | 3.974 ± 0.009 | +0.042 (9.2σ) | +0.023 (4.6σ) |
| **Reliability %** ↑ | 95.823 ± 0.056 | 96.104 ± 0.065 | 96.248 ± 0.018 | +0.28 (7.3σ) | +0.14 (4.7σ) |
| **HO rate** | 13.855 ± 0.366 | 14.739 ± 0.332 | 15.473 ± 0.526 | +0.88 (4.0σ) | +0.73 (2.6σ) |
| **PP rate** | 1.827 ± 0.056 | 2.005 ± 0.098 | 1.946 ± 0.077 | +0.18 (3.5σ) | −0.06 (1.1σ, n.s.) |
| **Prep** | 1010.8 ± 11.4 | 1038.8 ± 8.4 | 1057.6 ± 14.9 | +28.0 (4.4σ) | +18.9 (2.5σ) |
| **ResReservation %** | 8.022 ± 0.091 | 8.244 ± 0.067 | 8.394 ± 0.119 | +0.22 (4.4σ) | +0.15 (2.5σ) |

### Per-metric reading

- **Reward** — monotone, both stages strongly significant; DQN→RA = **+18.9 (≈+1.8%)**.
  Decomposition: distributional (DQN→RN) **+14.1 = 75%**, risk (RN→RA) **+4.8 = 25%**.
  The representation is the larger lever; risk-aversion is a real, significant top-up.
- **HOF ↓** — the safety headline: **2.233 → 1.176, nearly halved (−47%)** (distributional
  −36%, risk a further −18% of RN; 9.1σ then 3.7σ). The paper's strongest claim.
- **RLF ↓** — 0.143 → 0.093 (**−35%**), both stages significant; tracks HOF (same
  SINR-collapse events avoided).
- **Capacity ↑** — 3.909 → 3.974 (+1.7%), monotone, significant. **Risk-aversion does
  NOT cost throughput — RA has the HIGHEST capacity:** avoiding outages (MCS→0) lifts
  the mean. Safety and throughput align.
- **Reliability % ↑** — 95.82 → 96.25; RA spends least time at zero-rate.
- **HO rate** — RISES 13.86 → 15.47 (+12%). The QR/risk-averse agents **hand over
  MORE** — the *mechanism*, not a regression: proactive HOs pre-empt link failures.
- **PP rate** — DQN 1.83 → RN 2.01 (+3.5σ) → RA 1.95 (n.s. vs RN). QR incurs slightly
  more ping-pong than DQN (cost of more HOs), but **risk-aversion doesn't worsen it.**
- **Prep** — 1010.8 → 1057.6 (+4.6%); more HOs ⇒ more preparations (resource cost).
- **ResReservation %** — 8.02 → 8.39 (+4.6%); RA reserves the most — the explicit price
  paid for reliability.

### Synthesis (finals + sweeps + bridge)

1. **Two separable, significant gains confirmed at full budget:** distributional
   (DQN→RN: +14.1 reward, HOF −36%, 9–12σ) and risk (RN→RA: +4.8, HOF −18%, 3–5σ).
2. **k=10 ≈ k=13** → the quantile-count plateau holds at 2000 ep; 10 quantiles suffice.
3. **A clean, favourable trade across ALL metrics:** value metrics (reward, HOF↓, RLF↓,
   capacity↑, reliability↑) improve monotonically and significantly; cost metrics (HO,
   PP, prep, reservation) rise only modestly (+5–12%). **Risk-aversion buys large
   reliability + capacity gains for a small, bounded overhead increase** — and reward,
   which prices the whole trade-off, rises, so the trade is net positive.
4. **Interpretable end-to-end:** more (proactive) handovers → fewer link failures →
   higher capacity and reliability, pushed furthest by risk-aversion at α*=0.5.

---

## ★ THE 4th RUN — single risk-averse quantile at full budget (a KEY CORRECTION)

QR-DQN risk-averse, **k=1** quantile at τ=0.25, α=0.50, 2000 ep × 5 seed (the
canonical twin of bridge `ra_k1_t25`). Vs the other finals:

| model | reward | HOF | HO | Prep | ResRsv% |
|---|---|---|---|---|---|
| QR-RN (k=10, mean) | 1081.53 ± 1.74 | 1.432 | 14.74 | 1038.8 | 8.244 |
| **RA-k1 (1 quantile, VaR)** | **1080.58 ± 1.17** | **1.469** | **16.12** | **1072.9** | **8.515** |
| RA-k10 (CVaR₀.₅, champion) | 1086.31 ± 1.16 | 1.176 | 15.47 | 1057.6 | 8.394 |

**Finding: at convergence the single risk-averse quantile collapses to RISK-NEUTRAL
performance.** RA-k1 ≈ QR-RN on reward (−0.9, ~1σ, n.s.) and HOF (+0.04, n.s.), and
sits **−5.7 reward (7.8σ) / +0.29 HOF (6.2σ) below the full k=10 CVaR champion.**

This **corrects the bridge study's low-budget reading (Finding 5).** At 1000 ep the
single VaR quantile (1078.2) *appeared* to beat 10 risk-neutral quantiles (1075.4) —
a transient: the single quantile trains faster early but **saturates** (1078.2 →
1080.6, only +2.4 with 2× the training), while the multi-quantile models keep
improving (RN +6.1; RA-k10 → 1086.3) and overtake it.

**Why VaR(k=1) underperforms CVaR(k=10):** RA-k1 has the **highest HO rate, prep,
and reservation of all four models** yet only RN-level reward/safety. A single tail
quantile is a **high-variance risk estimate** — it over-reacts (most handovers, most
overhead) without realizing the benefit. CVaR's averaging over 10 tail quantiles is
a **stable** estimate → fewer handovers, lower overhead, *and* the best outcome.

**Conclusion for the paper:** risk-aversion is necessary but **not sufficient alone**
— it needs a stable, multi-quantile (CVaR) estimate. You cannot shortcut to the
champion with one quantile; **both axes — ≈10-quantile resolution AND risk-aware CVaR
selection — are required.** (This *strengthens* the case for the full QR-DQN+CVaR
model and supersedes the optimistic single-quantile framing in Findings 5–6, which
should be presented as "cheap early approximation that saturates," not "free lunch.")
