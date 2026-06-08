# Paper Revision TODO — Distributional RL for LTM Handover

Tracking every change needed to bring `overleaf-paper.tex` from the **stale
version** (100 ms fixed-cadence env, total reward, N=10/20, α≈0.5 interior
optimum) to the **final version** (no-gate event-driven env, reward-per-decision,
N=25, monotone risk dial, RA-1q champion, soft-step).

**Do NOT edit `overleaf-paper.tex` directly — Gerard edits it himself.** This
file is the source of truth for *what* must change.

Status legend: `[x]` done · `[~]` partially done · `[ ]` pending

---

## Canonical KPI order (every table, every plot, every prose list)

**Reward · Capacity · Reliability · HO · PP · HOF · RLF · Prep · Reservation**

Rationale: benefits (higher is better) first → mobility outcomes by increasing
severity (HO → PP → HOF → RLF; also decreasing frequency; all driven down by risk
aversion) → resource cost last (reservation is the one metric the risk-averse
agent worsens). Decided 2026-06-07.

Applied: `tab:dist` ✓; `generate_final_plots.py` (bars + radar) ✓;
`summarize_bmk.py` ✓. **Pending:** `tab:final`, `tab:alpha`, the frontier and
saturation figures (`plot_risk_frontier.py`, `plot_n_sweep.py` — set order when
built), and any metric list in prose.

---

## 0. Foundational changes (these gate everything else)

- [x] **Environment**: 100 ms fixed decision clock → **no-gate event-driven**
  (agent replaces the LTM +3 dB execution trigger; acts once per LTM cycle;
  HO only to prepared cells). *Done in §III.*
- [~] **Metric**: total episode reward → **reward-per-decision**
  (total reward is cadence-confounded). *Done in §III + §V (committed to as THE
  reported scalar). §VI still reports totals (~1081) → pending there. Abstract/§I
  pending.*
- [~] **Quantile backbone**: N=10/20 → **N=25** (midpoint, [256,256], lr 1e-4,
  γ 0.9, τ 0.01, train_freq 8, κ 0.25). *Done in §IV/§V. §VI/Abstract/§I still
  say N=10/20 → pending there.*

---

## 1. Section-by-section status

### Abstract — [ ] pending
- [ ] Remove "interior risk optimum at α≈0.5" (REVERSED — see §6 below).
- [ ] Remove "risk-aware agent ... attains best reward" framing.
- [ ] Rewrite to the new arc (see "New contribution arc" at bottom).
- [ ] "1,000-trajectory" benchmark phrasing still fine; KPI claims need no-gate numbers.

### §I Introduction — [ ] pending
- [ ] Contributions list (lines ~88–106):
  - [ ] "truncated-CVaR head" bullet → reframe as *cheaper estimator at equal
    quality*, not "better".
  - [ ] "interior risk optimum at α*≈0.5" bullet → REVERSED (monotone dial,
    knee α≈0.25; de-confounding cadence is the methodological point).
  - [ ] "quantile-bridge ... both necessary" bullet → REVERSED conclusion
    (single dedicated quantile / RA-1q is the tail champion).
  - [ ] Add bullets for the new contributions: soft-step weighted-quantile
    (spectral) policy; "learn what you act on"; risk-aversion at scalar
    inference cost.

### §II Background and Related Work — [x] DONE
- [x] Ordering claim rephrased ("no canonical total order / scalarization
  encodes a risk attitude").
- [x] Spectral/distortion risk sentence added + `acerbi` cite (foreshadows
  soft-step).
- [x] `rockafellar` moved to the CVaR-history cite.
- [x] Spelling sweep: `signalling→signaling` (line ~115), `modelling→modeling`
  (line ~126) — DONE for §II (whole section re-scanned, US-clean); the *global*
  US/UK pass over other sections still pending (§Housekeeping).
- [x] Drop "known" / fix `i.e.\ ` spacing in the CVaR example — DONE
  (now "for example … `i.e.\ $0.1$`").
- **Dependency created**: §II-C now promises a "weighted-quantile design space" →
  the revised §IV/§VI MUST deliver soft-step (confirmed: it's going in).

### §III System Model and Problem Formulation — [x] DONE
- [x] §III-A deployment + simulator rewritten (event-driven, learned execution
  trigger; preparation stays LTM heuristic; numbers pushed to table).
- [x] Table I: "event-driven (per LTM cycle)", preparation offset, execution
  offset annotated as the LTM trigger the agent replaces.
- [x] §III-B MDP: decision-epoch framing (dropped "semi-Markov"); masked action
  set {serving}∪{prepared}; per-decision reward; reward-per-decision motivation.

### §IV Method — [x] DONE (in overleaf-paper.tex, verified 2026-06-07) — RESTRUCTURED to 2 subsections
**Broken refs left for §V/§VI rewrite:** `sec:results-risk` (new, §IV-B ¶5, no label yet)
and `sec:trunc-cvar` (old label deleted; still ref'd at stale §V line 457 + §VI line 621 →
should become `sec:method-risk`).
**Organizing principle (verified in `quantile_modes.py`):** risk enters in two
distinct places — (a) WEIGHTS = `cvar_weights`, applied at ACTION SELECTION only,
backbone trained faithfully (uniform `mean_weights`/`predictor_weights`) →
"risk at scalar inference cost, shared backbone" [hard CVaR, soft-step];
(b) GRID = train only `k=⌈αN⌉` tail quantiles → a TRAINING-time change
[truncated CVaR; RA-1q is the k=1 endpoint, one quantile at τ=α/2]. This
weights-vs-grid split is the section scaffolding.

- **§IV-A QR-DQN & Quantile Loss** — keep; apply the 5 review edits
  (scheme rename, forward sentence, Huberized, optional /κ, median-vs-mean);
  **move the midpoint def (`τ_i=(i-½)/N, w_i=1/N`) up into §IV-A**; keep the
  N=1,τ=0.5 → median-Huber observation.
- **OLD §IV-B "Quantile-Positioning Schemes" (quadrature) → DELETE from Method**
  (deferral decision). GL/trap definitions leave Method; if §VI keeps a
  quadrature sentence, put a 1-line def in §V Setup.
- **NEW §IV-B "Risk-Aware Quantile Aggregation"** — big expansion, 5 labeled ¶:
  - [ ] ¶1 Risk-neutral (mean), `w_i=1/N`.
  - [ ] ¶2 Hard CVaR_α — 0/1 step weights over `τ_i≤α`; action-time only.
  - [ ] ¶3 **Soft-step (spectral)** [NEW, §II-C payoff]: two-level step
    `w_i∝r (τ≤α) else 1`, renorm; r=1→mean, r→∞→CVaR; weights non-increasing
    in τ ⇒ valid spectral measure; distorts WEIGHTS not POSITIONS.
  - [ ] ¶4 **Truncated CVaR** [REFRAME]: train only `k=⌈αN⌉` in `[0,α]`, mean =
    CVaR_α. **Punchline FLIPS**: full ≥ truncated on reward/decision (equal
    quality); truncation buys cost/speed, NOT better returns. Numbers → §VI.
  - [ ] ¶5 **Single dedicated quantile (RA-1q)** [NEW, climax]: k=1 limit, one
    trained quantile at τ=α/2; "learn what you act on." One sentence on the
    learn-full/act-single capstone control (results → §VI).
- **OLD §IV-C "Truncated CVaR" → dissolved** into NEW §IV-B ¶4–¶5.
- RA-1q DECISION: *defined* in §IV-B ¶5; *empirics* (champion, dilution ladder,
  failed capstone, 3-dial frontier) stay in §VI.

### §V Experimental Setup — [x] DONE (in overleaf-paper.tex, verified 2026-06-07)
**DECISIONS (all applied):** roster = **DQN, RN, soft-step, RA-1q (+LTM, CMAB)**
(full-CVaR / trunc-k7 / dilution / capstone are §VI study-only, NOT final-table
columns); **robustness compressed to sentences, tab:kappa dropped**; target
network + soft τ=0.01 introduced here (vanilla QR-DQN, not double-DQN); N=25,
train_freq=8; tab:hparams = single shared-backbone column; reward-per-decision
committed to as THE reported scalar; `sec:trunc-cvar`→`sec:method-risk` fixed.
**SCALE CHANGE: every reward number total `~1081` → reward-per-decision `~3.6`.**
*Record of the original plan kept below for traceability.*

**A. Protocol (§V-A):**
- [ ] Cadence: drop "one HO decision every 100 ms" (line 433) → event-driven (per §III).
- [ ] Metric: commit to **reward-per-decision** (= total ÷ #decisions) as THE reported
  scalar; total reward NOT reported (§III payoff).
- [ ] Budgets explicit: finals 5×2000, studies/ablations 3×1000.
- [ ] Spelling: generalisation→generalization (437), optimiser→optimizer (471).
- [ ] Keep 8-KPI list + same-trajectory train/eval justification (correct).

**B. Models & hyperparameters (§V-B):**
- [ ] `N` 10/20 → **25**; `train_freq` 1 → **8** (+1-line rationale: ~8× faster,
  reward unchanged vs tf=1 — locked).
- [ ] **Introduce target network + soft τ=0.01** here (vanilla QR-DQN, NOT double-DQN;
  a⋆ & θ_j both from frozen target net). Deferred from §IV-A.
- [ ] Fix dangling ref `sec:trunc-cvar` → `sec:method-risk` (457).
- [ ] Re-cast roster: drop stale 3-agent {DQN/RN/RA=truncCVaR0.5,k10,N20}; §IV-B now
  defines the family. §V-B = shared-backbone table + short roster naming the run
  instances. Pick **headline soft-step r + α** (canonical has soft r=2 fully
  populated, α=0.25) and **RA-1q α=0.25, k=1**.
- [ ] Restructure tab:hparams → ONE shared-backbone column + tiny per-agent risk-knob
  row (DQN scalar / RN mean / soft-step r / RA-1q α,k=1). Drop stale
  "Quantile support [0,1]/[0,0.5]" and "Predicted k=10" rows.

**C. Robustness (compress):**
- [ ] κ / width / lr / n-step → 1–2 qualitative sentences (no σ tables). **DROP tab:kappa.**
  Frees ~1/3 page. (Optionally re-confirm in no-gate later if space/credibility needs it.)

### §VI Results — IN PROGRESS — PLAN SET + GROUNDED (2026-06-07)
**Status:** §VI-A (`sec:results-dist`) and §VI-B (`sec:results-risk`) DONE in
overleaf-paper.tex (sober noun-phrase run-ins, reward-per-decision, tab:dist,
tab:alpha, fig:frontier). §VI-C/§VI-D drafted 2026-06-07; remaining stale:
Abstract, §I, §VII, §VIII. Broken refs still to fix in stale sections:
`sec:results-sweep` (§I L98 → `sec:results-risk`), `sec:results-qmode`
(§VII L826 → `sec:results-dist` or drop), `sec:trunc-cvar` (was §VI-C L641 →
`sec:method-risk`; resolved once §VI-C is replaced).

**NEW STRUCTURE — 3 subsections (down from 6), reorganized around the
contribution arc. The 3 labels are FORCED by locked §IV-B forward-refs
(`sec:results-risk` L420, `sec:results-bridge` L417, `sec:results-trunc` L401) —
they MUST exist. Stale subsections `sec:results-sweep`, `sec:results-qmode`,
`sec:results-3way`, `sec:results-baselines` are dissolved/merged.**

All numbers below are CURRENT (verified 2026-06-07 from eval CSVs). Reward =
**reward per decision** (~3.6). Finals = 2000 ep, 5 seeds; studies = 1000 ep,
3 seeds.

#### §VI-A — "Risk is a monotone dial: a de-confounded sweep" → `\label{sec:results-risk}`
(resolves §IV L420; absorbs the stale `sec:results-sweep`)
- [ ] **De-confounding setup**: separate the two entangled knobs — quantile count
  `k` (= number of final-layer outputs used) vs CVaR level `α`. A naive
  fixed-`N`, vary-`α` sweep silently changes `k` too. We hold one fixed and move
  the other.
- [ ] **REVERSAL (b)**: "concave curve, interior optimum α≈0.5" → **MONOTONE
  dial, NO interior optimum.** De-confounded CVaR α-sweep (fixed N=25, vary α),
  reward/decision: α=.10→3.542, .20→3.580, .25→3.598, .30→3.595, .40→3.619,
  .50→3.613 (rising toward risk-neutral); HOF falls monotonically toward
  risk-averse (.10→0.353 … .50→0.442). **Knee ≈ α=0.25.** No magic α; it's a
  controllable reward↔tail trade. Refresh `tab:alpha` with these.
- [ ] **Unified frontier** (the headline risk figure, `risk_frontier.png` — TO
  GENERATE): hard CVaR, soft-step (r dial), and the single dedicated quantile all
  trace the SAME risk–reliability frontier. Soft-step is the smooth member:
  r=1→mean, r→∞→hard CVaR (confirmed empirically — soft r=2: 3.655, r=3: 3.648,
  r=10: 3.620, r=30: 3.614, r=100: 3.606 → converging to hard CVaR α.25 ≈ 3.598).
- [ ] **Quadrature foil sentence** (absorbs `sec:results-qmode` + drops `tab:qmode`):
  one sentence — repositioning quantiles to better integrate the *mean* (midpoint
  / Gauss–Legendre / trapezoidal / Simpson / β-warp all within ±0.006:
  3.654–3.660) is **inert**; it is reweighting/repositioning for the *tail* that
  moves the policy. (Negative result, orthogonal to the arc → 1 line, no table.)
  NOTE: §VII L793 ref to `sec:results-qmode` must be dropped in the §VII rewrite.

#### §VI-B — "Spending quantile capacity: dilution and the dedicated quantile" → `\label{sec:results-bridge}` (+ `\label{sec:results-trunc}` on the truncation ¶)
(resolves §IV L417 and L401)
- [ ] **The bridge (mean side)**: 1 risk-neutral quantile (τ=0.5) ≈ DQN
  (median≈mean on near-symmetric returns); reward climbs with quantile count and
  saturates by N≈4–5 → the distributional gain *for the mean* is a multi-quantile
  effect. (Use dilution-ladder/bridge numbers; refresh from no-gate.)
- [ ] **REVERSAL (d) — dilution for RISK is the OPPOSITE**: averaging MORE tail
  quantiles HURTS. k-sweep (fixed α, vary averaged count): k=1→3.634, 2→3.627,
  3→3.611, 5→3.597, 15→3.568, 25→3.555. Dilution ladder (fixed α=.25, vary N):
  N=4→3.651, 12→3.627, 25→3.605. Dense (pack more quantiles into the tail) is
  worst (ds_dense a040 3.587 < trunc 3.619 < full 3.628). **More tail quantiles =
  worse.**
- [ ] **RA-1q is the failure-tail champion** (was "single RA quantile not enough"
  — REVERSED): single dedicated quantile at τ=α/2 gives the **lowest HOF (0.399,
  −40% vs DQN 0.667)** at **scalar DQN inference cost** and **fastest training (76
  min vs DQN 87, RN 141)**. "Learn what you act on."
- [ ] **The failed capstone (causal control)**: learn the full N=25 distribution
  but ACT on a single τ=α/2 quantile (vqn25_rf*: 3.591–3.628, HOF 0.42–0.48)
  UNDERPERFORMS RA-1q (3.641, HOF 0.399). ⇒ it is not enough to *act* on one
  quantile; you must *train* the one you act on. Nails causality for "learn what
  you act on."
- [ ] **REVERSAL (c) — truncated vs full** (`sec:results-trunc` ¶, drop `tab:trunc`):
  "truncated is a better estimator" → **full ≈ truncated** (statistically
  indistinguishable on reward/decision and tails: ds_full a040 3.628 vs ds_trunc
  3.619; full-CVaR final 3.622 vs trunc-k7 3.589). Truncation buys a smaller head
  + faster training (trunc-k7 105 min vs full ≈ RN 141), **not a better policy.**
  Matches the wording §IV-B L401-404 already commits to.

#### §VI-C — "Comparison with the LTM heuristic and contextual bandit" → reuse `\label{sec:results-3way}` or rename `sec:results-main` (merge stale 3way + baselines)
- [ ] **Refresh `tab:final`** to no-gate reward-per-decision, roster
  **DQN / RN / soft (r=2) / RA-1q + LTM + CMAB** (drop the stale RA=truncCVaR0.5
  column; RA-1q is now a headline agent, not an "ablation"). Numbers in the
  canonical table below. Remove the "Arrows mark the desirable direction" — encode
  ↑/↓ in the KPI label text or a caption note, NOT as in-prose symbols.
- [ ] **Conclusion 1 — distributional cuts tails for free**: DQN→RN ties reward
  (3.667 vs 3.663, within noise) but cuts HOF 17% (0.667→0.551), PP 20%, HO 9% —
  RN **Pareto-improves** DQN. (was "+14.1 reward (11.7σ)" — REVERSED to tied
  reward / free tail reduction.)
- [ ] **Conclusion 2 — risk trades a sliver of reward for big tail cuts**: RN→soft
  →RA-1q steps reward down slightly (3.663→3.659→3.641, <1%) while HOF falls
  0.551→0.512→0.399 and HO 5.12→4.85→4.59. Reframe — NOT a reward gain; an
  interpretable, controllable reward↔reliability trade.
- [ ] **Conclusion 3 — risk-aversion at scalar cost**: RA-1q gets the best tail
  with DQN's output dimension/inference cost and the fastest training.
- [ ] **REVERSAL (e) — RL is MORE frugal than LTM (and CMAB)**: "RA is the most
  active / more HOs than LTM" → **REVERSED**. RL HO 4.6–5.6/min < CMAB 8.0 < LTM
  11.34. RL also lower PP (0.66–1.07 vs LTM 3.63), lower HOF (0.40–0.67 vs 1.151),
  lower RLF (0.045 vs 0.065). The "buys reliability with signaling activity" story
  is GONE.
- [ ] **vs LTM (same simulator, apples-to-apples)**: RL **dominates on every
  user/safety KPI** — Cap +7.8% (4.044 vs 3.753), Rel +1.7pt (96.94 vs 95.25),
  HOF −65% (0.399 vs 1.151), RLF −31% (0.045 vs 0.065), PP −82%, HO −60% (more
  frugal). **The ONE honest trade-off**: resource reservation higher (RL ~8.8–9.0%
  vs LTM 6.44%). Report it plainly. (Note: HOF no longer "matches" LTM — it BEATS
  it; old draft was wrong.)
- [ ] **vs CMAB (cross-simulator, indicative)**: RL ahead on Cap (4.04 vs 3.68),
  Rel (96.9 vs 94.0), HOF (0.40 vs 0.97), RLF (0.045 vs 0.101), and is MORE frugal
  on HO (4.6 vs 8.0). **CORRECT CMAB numbers: prep 24.05 (NOT 20), res 6.82 (NOT
  6.5).** Prep is now COMPARABLE across all agents (~24–29) — the stale "CMAB 50×
  fewer preparations" claim is FALSE; delete it. CMAB's only edge: resource
  reservation (6.82 vs ~8.8). Keep "complementary pre-selection layer, could
  stack" framing.
- [ ] **REMOVE RIS/LMMSE entirely** (tutor + user agree): the
  "orthogonal RIS and LMMSE-prediction enhancements" sentence goes. Compare only
  vs LTM and CMAB. (Maybe a single neutral mention in Conclusions; likely not even
  there.)
- [ ] **Cross-sim caveat footnote**: keep but refresh — our LTM reproduces paper
  LTM within ~5% (HO 11.34 vs 11.0, HOF 1.151 vs 1.10, Cap 3.753 vs 3.75); res
  reservation is the largest gap.
- [ ] **Master figure (Fig.1, fig:bars)**: regenerate `master_bar_plots.png` for
  no-gate numbers; drop the "paper published numbers" striped-hatch series if no
  longer shown; update caption coloring. Remove RIS/LMMSE from the figure.

#### §VI tutor comments (Margarita) — apply throughout
- [ ] **"head"** first use: define as "the number of outputs of the final layer"
  (do it in §V or first §VI use).
- [ ] **"2000×5 budget" notation**: tutor doesn't follow budget-vs-seed. Replace
  everywhere with explicit prose: "trained for 2000 episodes and averaged over 5
  random seeds" (and "1000 episodes, 3 seeds" for studies). No "N×M budget"
  shorthand.
- [ ] **Formality / no symbols in prose**: remove arrows (→) and informal phrasing
  ("reach the champions", "not enough", "tellingly"). Write transitions in words;
  reserve symbols for variables/equations. Spell percentages as needed but avoid
  bare "−47%" style inline where prose reads better.
- [ ] **"ablation" → "study"** everywhere (quantile-bridge study, etc.).

### §VII Discussion, Limitations, Future Work — [ ] pending
- [ ] Remove "our risk-aware policy is the most active agent" lever / α_HO
  re-tuning framing (premise reversed — RL is now frugal).
- [ ] Limitation "(iv) CVaR is the only risk measure" → no longer true (spectral
  soft-step family now studied) — move to a contribution.
- [ ] Soft-step moves from "future work" to results.

### §VIII Conclusion — [ ] pending
- [ ] "optimal level α≈0.5 we establish" → monotone dial / knee α≈0.25.
- [ ] "RA attains the best reward" → corrected arc.
- [ ] Bridge sentence ("single RA quantile only reaches RN") → REVERSED.

---

## 2. New results / content to ADD (absent from the stale draft)

- [ ] **Soft-step distortion — "weights beat positions"**: soft-step weights
  Pareto-beat hard CVaR; distortion study (weights vs density vs β-warp);
  `beta_weighted ≈ RN` causal control.
- [ ] **Unified risk frontier** (3 dials: hard-CVaR / single-quantile / soft-step)
  — the new headline risk figure.
- [ ] **RA-1q champion** + **dilution ladder** (N=1/4/12/20) + **"learn what you
  act on"** + the **failed capstone** (learn-full/act-single).
- [ ] **Full hard-CVaR is dominated** at canonical scale (textbook recipe loses).
- [ ] **Per-UE fairness** (worst-user RLF capped; failing-user fraction).
- [ ] **Interpretability** (return tracks SINR, corr +0.95; tail dispersion rises
  as link degrades).
- [ ] **Inference cost** (RA-1q ≡ DQN params → risk-aversion at scalar cost).
- [ ] **Dense study** (full > trunc > dense; packing tail quantiles doesn't help).
- [ ] **soft-step → hard CVaR convergence** (one-parameter family, empirical).
- [ ] **N=25 backbone decision** (paper still says N=10/20).

---

## 3. Figures

- [ ] Add `risk_frontier.png` (headline).
- [ ] Add `per_ue_tails.png`.
- [ ] Add `return_distributions.png`.
- [ ] Regenerate `master_bar_plots.png` for no-gate numbers (Fig.1).
- [ ] Ensure all `\includegraphics` figures are uploaded to Overleaf.

---

## 4. Housekeeping (pre-submission)

- [~] Global US/UK spelling sweep (signaling, modeling, optimize, penalize, …;
  abstract still has "optimises/penalised"). §II done; §III–§VIII + abstract pending.
- [ ] Remove all `\TODO` / `\PLACEHOLDER` markers and the red reminder note (line ~807).
- [ ] `cmab_ho` bib entry: fill venue/year/pages (currently placeholder).
- [ ] Acknowledgment: fill funding placeholder (line ~790).
- [ ] Page limit: **8 pages max, excluding references** — watch length as new
  results go in.

---

## 5. Verified good — keep as-is (just renumber/recite to no-gate)

- QR-DQN math & quantile-Huber loss (§IV-A) — incl. N=1,τ=0.5 → median-Huber.
- Quadrature-scheme descriptions (§IV-B); midpoint-wins conclusion.
- Reward equation (2) — unchanged in form.
- §II related work (now revised).
- κ-immaterial and width-saturates ablations (numbers → no-gate).

---

## New contribution arc (for Abstract / Intro rewrite)

1. Distributional learning cuts tails **for free** (RN ⪰ DQN, tied reward).
2. CVaR is a **monotone, controllable** risk dial — no magic α; de-confounding
   cadence is itself a methodological contribution (overturns the apparent α≈0.5).
3. **How to encode risk efficiently**: soft-step *weights* beat *positions*, and
   a single *dedicated* quantile (RA-1q) is the failure-tail optimum — "learn
   what you act on."
4. Beats the LTM heuristic on user KPIs while being **more frugal**; risk-aversion
   at scalar inference cost.

---

## Canonical target numbers (verified 2026-06-07 from eval CSVs)

### Finals — `tab:final` (no-gate, reward-per-decision, 2000 ep, 5 seeds)
Headline roster = DQN, RN, soft (r=2), RA-1q. full-CVaR / RA-k7 / RA-trunc.5 are
study-only (NOT final-table columns). LTM = our simulator; CMAB = paper (indicative).

| Agent          | rew/act | HO/min | PP/min | HOF/min | RLF/min | Cap (bps/Hz) | Rel (%) | Prep/min | Res (%) |
|----------------|---------|--------|--------|---------|---------|--------------|---------|----------|---------|
| DQN            | 3.667   | 5.63   | 1.065  | 0.667   | 0.0472  | 4.062        | 97.00   | 28.07    | 8.85    |
| RN             | 3.663   | 5.12   | 0.857  | 0.551   | 0.0456  | 4.068        | 97.07   | 27.71    | 8.91    |
| **soft (r=2)** | 3.659   | 4.85   | 0.732  | 0.512   | 0.0458  | 4.067        | 97.08   | 27.59    | 9.00    |
| soft (r=3)     | 3.652   | 4.69   | 0.683  | 0.468   | 0.0449  | 4.065        | 97.08   | 27.57    | 9.10    |
| **RA-1q (α.25)** | 3.641 | 4.59   | 0.656  | **0.399** | 0.0446 | 4.044       | 96.94   | 24.70    | 8.82    |
| full-CVaR α.25 | 3.622   | 4.69   | 0.703  | 0.467   | 0.0446  | 4.050        | 97.03   | 28.50    | 9.45    |
| RA-trunc.5 (k7)| 3.589   | 4.59   | 0.704  | 0.419   | 0.0422  | 4.011        | 96.87   | 25.96    | 9.25    |
| **LTM** (ours) | —       | 11.34  | 3.625  | 1.151   | 0.065   | 3.753        | 95.25   | 27.32    | 6.44    |
| **CMAB** (paper)| —      | 8.00   | 1.30   | 0.97    | 0.101   | 3.68         | 94.0    | 24.05    | 6.82    |

Train times (min): DQN 87 · RN 141 · full-CVaR ≈141 · RA-trunc.5(k7) 105 ·
RA-1q 76 · soft 122. (Re-pull from train logs before quoting in prose.)

CMAB CORRECTIONS vs stale draft: prep **24.05** (was 20), res **6.82** (was 6.5).
Prep is now COMPARABLE across ALL agents (~24–29) — no "50× fewer prep" story.

### Risk dial — `tab:alpha` (de-confounded CVaR α-sweep, fixed N=25, 1000 ep × 3 seeds)
| α        | 0.10  | 0.20  | 0.25  | 0.30  | 0.40  | 0.50  |
|----------|-------|-------|-------|-------|-------|-------|
| rew/act  | 3.542 | 3.580 | 3.598 | 3.595 | 3.619 | 3.613 |
| HOF/min  | 0.353 | 0.378 | 0.367 | 0.400 | 0.413 | 0.442 |
Single-quantile counterpart (uf_q1, k=1): 3.573 / 3.618 / 3.634 / 3.645 / 3.658 /
3.658. → MONOTONE dial, knee ≈ 0.25, NO interior optimum.

### Soft→hard convergence (1000 ep × 3 seeds): r=2 3.655 · r=3 3.648 · r=10 3.620 · r=30 3.614 · r=100 3.606  (→ hard CVaR α.25 ≈ 3.598)

### Dilution (1000 ep × 3 seeds)
k-sweep (fixed α, vary averaged count): k=1 3.634 · 2 3.627 · 3 3.611 · 5 3.597 · 15 3.568 · 25 3.555.
N-ladder (fixed α=.25): N=4 3.651 · 12 3.627 · 25 3.605.
dense vs trunc vs full (a040): dense 3.587 < trunc 3.619 < full 3.628.

### Failed capstone — learn-full-N25 / act-single (vqn25_rf*, 1000 ep × 3 seeds):
rf.12 3.591 · rf.20 3.610 · rf.28 3.628 · rf.36 3.626 (HOF 0.42–0.48) — all BELOW
RA-1q 3.641/HOF 0.399 ⇒ must TRAIN the quantile you act on.

### Quadrature foil (risk-neutral N=25, 1000 ep × 3 seeds): midpoint 3.660 · GL 3.658 · trap 3.654 · Simpson 3.658 · β-eq 3.654 · β-wt 3.656 (±0.006, inert).

Key reversals vs stale draft: (a) reward ranking DQN≈RN > soft > RA-1q > full-CVaR
> trunc (stale had RA best); (b) no interior α optimum, monotone dial knee≈0.25
(stale α≈0.5); (c) full ≈ truncated CVaR, equal-but-cheaper (stale truncated
better); (d) RA-1q is the tail champion, beats diluting more quantiles (stale "not
enough"); (e) RL more frugal than LTM AND CMAB (stale "most active").
