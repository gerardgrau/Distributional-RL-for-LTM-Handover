

# Meeting prep — Josep (Pepe) & Marga — 3 June 2026

## TL;DR — what to get out of the meeting
1. **Josep:** confirm the RL replaces only the **+3 dB execution trigger** (per ~100 ms
   decision, *not* every 10 ms), preparation stays LTM, HOs only to prepared cells.
2. **Marga:** OK to re-run everything (reward bug + masking)? Do we also need the
   non-masked comparison, or is the masked version enough?
3. **Both:** how to report the corrected Ainna prep number (≈26 vs the paper's 780);
   what fits in 8 pages; green-light the 2000×5 finals (backbone locked; we're choosing
   the RA risk level on the frontier).

---

# COMENTARIS POST-REUNIÓ:

ABSTRACT:
2 improvements:
també mencionar que DQN de per si és una Millora respecte LTM?


IV - C: Risk-Aware - Redactar emfatitzant el TRUNCATED -
Millora molt respecte el risk aware "estàndard"
Posar-ho en dos sub-apartats
posar el "truncated" com a millora meva, i relacionar-ho lleguremant amb el les non-uniform quantile distributions

---

**Context (30 s):** We re-ran the whole RL↔LTM pipeline after two corrections (a
reward bug + prepared-cell masking) and reformulated the RL's role as Pepe asked
(replace the +3 dB execution trigger). Below: what changed, the current numbers,
what we need decided, and a short note on a bug we found in the reference (Ainna)
code.

---

## 1. What we've changed since last time

**(a) Fixed a reward / HO-counting bug.** The original reward and handover count
penalized a "handover" even when the agent's chosen action was the *serving* cell
(i.e. no actual change). Now a stay/decline carries no HO penalty and is not
counted as a handover. **This is the main reason we are re-running all numbers.**

**(b) Prepared-cell masking (Josep's request).** The RL can now only hand over to
a **prepared** cell (its action set = prepared list + "stay"). Before, ~25 % of the
RL's handovers targeted non-prepared cells — inconsistent with the LTM two-stage
model. Masking is enforced both in action selection and in the Q-learning target.

**(c) The RL now replaces the LTM +3 dB EXECUTION trigger (core new idea, per
Pepe).** Instead of the fixed "+3 dB over serving" rule deciding *when* to execute
a handover, the RL decides *when* and *to which* prepared cell to hand over.
**Preparation** (the −3 dB / timer logic that builds the candidate list, ≤5 cells)
**stays the LTM heuristic.** So the RL is "a better execution criterion than the
+3 dB threshold", not a replacement of all of LTM.

**(d) Event-driven cadence (fairness fix).** The RL is now consulted at every
handover decision point (~once per 100 ms while ≥1 candidate is prepared),
matching how often the LTM baseline is consulted. Previously the RL acted on ~1 s
bundles and was consulted about half as often — an undisclosed handicap. Fixing it
closed the one metric where the RL trailed LTM (radio-link failures).

**(e) Distributional, risk-aware agent (the paper's angle).** We compare vanilla
**DQN**, risk-neutral QR-DQN (**RN**), and risk-averse QR-DQN with **CVaR** (**RA**).
On *reward* the three are essentially **tied** (and all ≫ LTM) — the gains live in the
**tail metrics**, which improve **DQN < RN < RA**. The distributional representation
already cuts ping-pong / HO-failures **risk-neutrally** (RN, at *no* reward cost),
and CVaR (RA) cuts them further by trading a small slice of reward (~2–3 %). The CVaR
level α is a tunable **risk–return dial** — that controllable frontier is the
contribution, not a claim that RA "wins everything."

**(f) Methodology check (can skip in the meeting).** We also tested an alternative
"time-integrated" reward to see if it was more principled; it was much worse (the
agent over-handed-over), so we kept the original per-decision reward — now with a
documented justification.

**(g) How we rank models — reward per action.** Ranking by *total* reward is invalid
under the event-driven cadence: the reward sum grows with how many decisions an agent
triggers (a confound — it even falsely crowned the *most* risk-averse agent). We rank
by **reward per action** (total reward ÷ number of decisions), which removes that
confound and judges *decision quality* — consistent with the per-decision objective
the agent is trained on. We lead with the 8 KPIs; reward-per-action is the single
summary scalar. (An earlier hand-weighted utility reproduced the same cadence bias and
was dropped.)

---

## 2. Current results — no-gate, 1000 ep × 3 seeds, all 1,000 UEs

| Metric (per min unless noted) | LTM baseline | DQN | RN | **RA** |
|---|---|---|---|---|
| Capacity (avg MCS) ↑ | 3.75 | 4.06 | 4.07 | 4.05 |
| Radio-link failures ↓ | 0.065 | 0.049 | 0.047 | **0.044** |
| Handovers ↓ | 11.3 | 5.6 | 5.2 | **4.7** |
| Ping-pong ↓ | 3.63 | 1.10 | 0.93 | **0.80** |
| HO failures ↓ | 1.15 | 0.66 | 0.52 | **0.43** |
| Reliability % ↑ | 95.2 | 97.0 | 97.1 | 97.0 |
| Resource reservation % ↓ | 6.4 | 9.0 | 9.0 | 9.2 |

**RA beats the baseline on 7 of 8 metrics** (the lone trade-off is resource
reservation — it keeps more candidates ready). Among the *learned* agents, **reward
per action is ~tied** (DQN 3.65 ≈ RN 3.65 ≈ RA, all ≫ LTM); the **DQN < RN < RA**
ordering is entirely in the **tail** metrics (HO, PP, HOF) — RN improves them over DQN
at no reward cost, RA further at a small one. These are 1000-ep probes; the canonical
finals (2000 ep × 5 seeds) are queued, pending today's decisions.

---

## 3. Questions / decisions for the meeting

### For Pepe (the reformulation)
1. **Cadence** — we made the RL act once per ~100 ms decision period (whenever ≥1
   candidate is prepared), **not** literally every 10 ms tick. Is that the right
   reading of "decide at every instant"? (Every 10 ms would ~10× the decisions with
   almost no new information — the prepared set barely changes within a period.)
2. **Scope** — the RL replaces only the *execution* trigger; *preparation* stays the
   LTM heuristic and handovers are restricted to prepared cells. Is that the intended
   scope, or did you also want the RL to learn *which cells to prepare* (closer to the
   CMAB direction)?
3. **Recovery exception** — after a radio-link failure the UE reconnects greedily to
   the strongest cell (unrestricted, identical for RL and baseline). OK to keep this
   outside the "prepared-only" rule?

### For Marga (scope / process)
4. **Re-run justification** — confirm she's aligned that we're re-running everything
   because of (a) the reward bug and (b) the prepared-masking.
5. **"Prepared vs non-prepared" comparison** — do we need to present *both* (RL with
   and without the prepared-mask) as a comparison, or is the masked (Josep-aligned)
   version sufficient? The comparison means extra runs.

### For both
6. **Ainna prep-rate number** — we found the reference's "≈780 preparations/min" is a
   code artifact (see §4); the true value is ≈26/min, and reservation is ≈6.3 %
   (paper says 5.7 %). How do we report this, given it revises a number from the
   group's prior paper?
7. **Paper scope** — 8 pages max (excl. references). Which studies make the cut: the
   DQN/RN/RA comparison (core), the **CVaR-α risk–return frontier** (strong), the
   cadence study, the quantile-count sweep (weak — N barely matters), the reward check?
8. **Green-light the finals** — any objection before we commit the 2000 ep × 5 seed
   finals: no-gate + prepared-mask backbone (N≈20), **DQN / RN (mean) / RA (CVaR at
   α≈0.25 on the frontier)**?

---

## 4. The bug in the reference (Ainna) code — the "missing-prep hole"

The reference simulator increments time (`t = t+1`) at the **top** of each decision
cycle, *before* it records the prepared-cell grid `ReservedBSSectors(:,t)`. This
leaves the first tick of every cycle as a 0 ("hole") in the grid.

- **On the preparation count** (which counts 0→1 transitions with `>0`): the hole
  inserts a spurious 0 between two prepared ticks, faking a brand-new "preparation"
  every cycle for every prepared cell → **inflates the count ~30×**. Reproduced
  exactly: holed = **787.9/min** (≈ the paper's 780), fixed = **25.7/min**.
- **On reservation %** (occupancy): the hole *removes* occupied ticks → **deflates**
  it: holed = **5.73 %** (≈ paper's 5.7 %), fixed = **6.33 %**.
- Handover / RLF / capacity are unchanged (handovers identical at 10.80 both ways),
  so the simulator's *dynamics* are faithful — only these two reported numbers were
  affected.
- **Fix:** one line — record `ReservedBSSectors(:,t) = ListBSPrepared` *before*
  `t = t+1`. Our code already does this. Ainna confirmed the Matlab uses the `>0`
  formula; our reproduction shows the 780 itself is the hole artifact, not a real
  preparation rate.
