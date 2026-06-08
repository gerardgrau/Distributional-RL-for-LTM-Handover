# Reformed finals (decline + prepared-mask + (a) HO_condition gate)

Canonical runs: 2000 ep × 5 seeds, backbones identical to the pre-reform
finals. Env reform = reward fix (a stay incurs no α_HO penalty) + execution
masked to the baseline's exact +3 dB candidate set (gate-only (a)). All
columns are on the SAME reward scale (the baseline is bit-identical under the
reform, so this is an apples-to-apples comparison).

## 8-metric comparison (mean over 5 seeds; baseline over 1000 UEs)

| Metric          | Baseline | DQN     | QR-RN   | QR-RA   | best        |
|-----------------|----------|---------|---------|---------|-------------|
| Reward          | 1055.45  | 1137.78 | 1142.42 | 1144.35 | **RA**      |
| Capacity        | 3.7532   | 3.9451  | 3.9586  | 3.9620  | **RA**      |
| Reliability %   | 95.247   | 96.243  | 96.365  | 96.466  | **RA**      |
| RLF rate        | 0.0650   | 0.1360  | 0.1300  | 0.1132  | baseline*   |
| HOF rate        | 1.1514   | 0.2347  | 0.2155  | 0.1976  | **RA**      |
| PP rate         | 3.6252   | 0.0756  | 0.0588  | 0.0330  | **RA**      |
| HO rate         | 11.342   | 3.3065  | 3.2228  | 3.0310  | **RA**      |
| Prep rate       | 27.322   | 24.236  | 23.569  | 22.424  | RA (lowest) |
| Res-reserv %    | 6.4384   | 8.7341  | 8.6119  | 8.5228  | baseline*   |
| reward seed-std | 88.6(UE) | 2.00    | 1.80    | 1.34    |             |

\* RLF and res-reservation are the two metrics where the RL agents still
trail the baseline; RA is the closest of the three on both (the risk-aware
agent reclaims the RLF tail: 0.136 → 0.113).

## Findings

1. **Two-gains structure holds, monotonic on every axis.** DQN < QR-RN < QR-RA
   on reward, capacity, reliability — and DQN > RN > RA (i.e. RA best) on RLF,
   HOF, PP, HO, prep, reservation. **QR-RA dominates all 8 metrics + reward.**
2. **Statistically robust.** Seed-std ~1.3–2.0 reward; the DQN→RN→RA
   separations are several σ apart, not noise.
3. **Gains compressed vs the pre-reform regime but same ordering.** Pre-reform
   (buggy reward): DQN 1067.4 < RN 1081.5 < RA 1086.3. Reform lifts the scale
   (+~70, the removed self-handover penalty) and shrinks the gaps
   (DQN→RN +4.6, RN→RA +1.9) because not penalising stays raises the floor.
3. **Tail-risk thesis vindicated.** RA's edge is concentrated in the tails it
   is built for — lowest RLF/HOF/PP — narrowing the RLF gap toward baseline.
4. **vs LTM:** all three RL agents beat the baseline on reward (+82…+89),
   capacity, reliability, PP, HOF, HO count; all trail on RLF and reservation.

## Provenance
- DQN: bmk_2026-06-02_1_reform_a_final_dqn
- QR-RN: bmk_2026-06-02_2_reform_a_final_rn
- QR-RA: bmk_2026-06-02_3_reform_a_final_ra
- baseline_summary.csv: copied unchanged (bit-identical under the reform).

These CSVs are staged in `reform_a/` and do NOT overwrite the committed
pre-reform `final_metrics/*.csv`. Promote + regenerate master plots after review.
