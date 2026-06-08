# The single-scalar ranking metric `U` (decided 2026-06-03)

**Purpose.** One number per model such that `U(A) > U(B)` objectively means "A is
better" *per our Ainna ponderation* (throughput good; HO/PP/HOF bad with weights
α = 0.8 / 0.9 / 0.1; reliability via the out-of-sync `N_OOS` factor). Used
throughout the article to rank models. The 8 KPIs remain the ground truth; `U` is
the headline summary scalar.

---

## The decided formula

```
U  =  util_throughput  ×  α_HO^(HO/min)  ×  α_PP^(PP/min)  ×  α_HOF^(HOF/min)

        util_throughput = ⟨ MCS(t) · reliability(t) ⟩_t          (time-averaged)
        reliability(t)  = 1 / (1 + exp(2·(N_OOS − 2)))           (the Ainna factor)
        α_HO, α_PP, α_HOF = 0.8, 0.9, 0.1                        (Ainna weights)
        rates are per minute (as already reported in the 8 KPIs)
```

- **Continuous half (`util_throughput`)** — MCS weighted by the reliability factor,
  averaged over time. This is the Ainna reward's throughput×reliability term.
  Time-averaging a continuous quantity is exact — no cadence artifact.
- **Discrete half (the α factors)** — each event type penalizes **by count**
  (per-minute rate), once per event. No duration-weighting.
- **No `α_RLF`.** Radio-link failures are already paid for twice: (1) the
  reliability factor collapses as `N_OOS` builds toward the N310/T310 threshold
  (an RLF *is* the tail of that process), and (2) `MCS = 0` through the outage
  pulls `util_throughput` down. Adding `α_RLF` would triple-count. (The Ainna
  reward itself has no `α_RLF` for exactly this reason.)

**Computation.** `util_throughput` is emitted by the env per episode
(`info["metrics"]["util_throughput"]`, plumbed to the eval summary); the rates come
straight from the 8 KPIs. For legacy runs without the field, `capacity_avg` (mean
MCS) is a ranking-equivalent proxy for `util_throughput`.

---

## Why this form (the two pitfalls it avoids)

**Pitfall 1 — cadence confound (kills the old "compare by total reward").** The
episode-sum reward adds one term per *decision*, and different agents face
different numbers of decisions (event-driven / no-gate). An agent that stays more
gets more short cycles → more terms → a bigger sum, independent of quality. So the
old "model A has higher reward" is no longer valid. `U` is built from time-averages
and per-minute rates → cadence-invariant.

**Pitfall 2 — duration-weighting / boundary-sensitivity.** A natural fix (the
env `composite_reward`) time-averages the *full* per-decision reward, but it spreads
the discrete event penalties over the inter-decision span. That makes *physically
identical* behaviour score differently depending on where decision boundaries
happen to fall:

> 1 handover then MCS 5 held for 10 s scores **4.0** if it's one long span, but
> **4.9** if a candidate flickers once and splits it into two spans — same HO, same
> throughput.

`U` removes this by applying the event penalties **by count**, not by span. The
continuous parts (MCS, reliability) are the only things time-averaged, and those
*should* be.

This also answers the "1 long HO vs 2 HOs then stable" question correctly: a second
handover adds one `α_HO` factor, while reaching a stable cell raises
`util_throughput`; `U` weighs the two transparently, with no boundary artifact.

---

## Validation (2026-06-03)

Five different aggregations of the same KPIs — multiplicative per-minute, the same
× reliability%, per-second exponents (unit change), additive with `−ln α` weights,
and one with a spurious `α_RLF` — **all agree on the order:**

| Formula | baseline | DQN | RN | **RA** |
|---|---|---|---|---|
| `U` (decided, cap-proxy) | 0.014 | 0.222 | 0.347 | **0.485** |
| × reliability% | 0.014 | 0.216 | 0.337 | **0.471** |
| per-second exponents | 3.421 | 3.867 | 3.902 | **3.909** |
| additive (`−ln α`) | 3.197 | 3.769 | 3.820 | **3.838** |
| + spurious `α_RLF` (rejected) | 0.014 | 0.215 | 0.336 | **0.471** |

**Order is always baseline ≪ DQN < RN < RA.** The conclusion does not depend on the
exact formula — which is the strongest argument that ranking by `U` is sound. RA
gives up a sliver of mean capacity for materially fewer tail events, and `U` (which
prices those events) puts it on top.

---

## Usage policy for the paper

1. **Ground truth = the 8 KPIs (Pareto).** RA dominates the baseline on 7/8 — so
   "RA is better" needs no weighting argument at all.
2. **Headline scalar = `U`.** One number, the Ainna ponderation, cadence- and
   boundary-clean. Use it in the abstract / summary ("RA's risk-adjusted utility is
   ~34× the baseline's").
3. **`composite_reward` (env) = training-faithful cross-check only.** It shows the
   agent optimizes what we trained; don't use it for fine model-vs-model claims
   (boundary-sensitive — Pitfall 2).
4. **Never rank by episode-sum reward again** (Pitfall 1).
