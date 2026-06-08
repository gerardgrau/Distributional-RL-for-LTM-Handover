# Paper experiments to repeat under the new regime (2026-06-03)

**New regime** = no-gate (learned_trigger) + prepared-mask + event-driven cadence +
mean-MCS reward (A/B-confirmed) + tf8 + γ0.9. **New metrics** = the 8 KPIs (bug-fixed
HO/reward) + the single ranking scalar `U` (see `ranking_metric.md`). Every RL number
in the paper was produced under the OLD env (gated, unmasked, buggy reward) and must
be regenerated.

| # | Paper element (line in paper.tex) | Status | Produced by |
|---|---|---|---|
| T1 | CVaR-α sweep, k=10 (≈ln 490) | **running** | no-gate α-sweep ablation (a010/025/050/075/100) |
| T2 | Quantile-count effect (part of bridge / N) | **running** | no-gate N-sweep ablation (N=10/25/50/100/200) |
| T3 | Quantile-positioning schemes: midpoint/GL/trapezoidal (≈ln 526) | **pending** | needs a no-gate quantile-mode sweep (RN, N=10) |
| T4 | Truncated vs full CVaR head (≈ln 572) | **pending** | needs no-gate truncate A/B at CVaR0.5 |
| T5 | Huber κ sweep (≈ln 442) | **pending** | needs no-gate κ sweep on RA backbone |
| T6 | **3-way finals** DQN/RN/RA, 5×2000 (≈ln 635) | **running tonight** | `configs/masked/no_gate/finals/*` via waiter |
| F1 | master_bar_plots.png (8 KPIs + U) (≈ln 662) | after T6 | `src/tools/generate_final_plots.py` |
| T7 | Quantile-bridge, single-quantile RA k=1 (≈ln 683) | **pending** | needs no-gate k=1 RA run + bridge |
| T8 | LTM/CMAB comparison (≈ln 734) | after T6 | finals + baseline re-eval (for `U`) |
| — | Baseline (LTM) `U` / util_throughput | **pending** | re-eval LTM baseline under new code |
| — | Sim-constants table (≈ln 212), HP table (≈ln 416) | static | update text only if backbone changed |

## Priority / sequencing
1. **Tonight (autonomous):** ablations (T1, T2) → finals (T6) → regen F1 + summaries.
   Baseline `U` re-eval folds into the regen.
2. **Still pending after tonight (need a session / more compute):** T3, T4, T5, T7 —
   the supporting distributional studies (quantile-positioning, truncate A/B, κ,
   quantile-bridge). Each is ~1000 ep × 3 seed (~40 min). They are *secondary*
   (the headline is T6 + T1); schedule them next, trimming to whatever fits the
   8-page budget — several may collapse to a sentence rather than a full table.
3. **All tables must add the `U` column** and use the bug-fixed KPIs.

## Notes
- The α-sweep (T1) also re-confirms the RA `risk_fraction` for the finals — if it
  moves off 0.5, update `finals/qrdqn_ra.yaml` before the RA final starts.
- The N-sweep (T2) similarly re-confirms RN's `num_quantiles` (currently 10).
- `composite_reward` is logged too but is a training-faithful cross-check only, not a
  ranking metric (boundary-sensitive — see `ranking_metric.md`).
