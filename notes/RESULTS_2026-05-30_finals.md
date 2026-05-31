# Canonical 3-Way Finals + De-confounded Risk Sweep — 2026-05-30

Closes the provenance gaps from the 2026-05-28 2-way run by (a) re-tuning the
risk-aware backbone at the operating point and (b) adding a matched risk-neutral
QR-DQN, giving a clean three-agent ablation on one backbone.

## Pipeline (all V3 backbone: h=[256,256], tf=1, κ=0.25, lr 1e-4, γ 0.9)

1. **N-sweep** at α=0.50, N∈{25,50,100} (k=13/25/50), 1000ep×3seed → lock k*.
2. **Adaptive α-sweep** at fixed k=13, α∈{0.30,0.40,0.50,0.70,1.00}, 1000ep×3seed
   (N=floor(13/α) so ⌈αN⌉=13 every point — identical 13-output head) → lock α*.
3. **3 canonical finals** at 2000ep×5seed on locked (k*,α*)=(13,0.50).

## Sweep 1 — quantile count (α=0.50, 1000ep×3seed), reward

| N (nominal) | k (in use) | reward |
|---|---|---|
| 25 | 13 | **1084.6 ± 1.4** |
| 50 | 25 | 1082.4 ± 4.8 |
| 100 | 50 | 1083.1 ± 1.1 |

Flat axis; smallest head best+cheapest → **k* = 13**.
Dirs: `bmk_2026-05-29_{1,2,3}_nsweep_a050_N{25,50,100}`.

## Sweep 2 — de-confounded CVaR level (fixed k=13, 1000ep×3seed)

| α | 0.30 | 0.40 | 0.50 | 0.70 | 1.00 |
|---|---|---|---|---|---|
| reward | 1083.4±5.0 | 1085.1±2.4 | 1084.6±1.4 | 1079.7±2.7 | 1073.0±4.2 |
| hof | 1.247 | 1.153 | 1.241 | 1.510 | 1.877 |

Concave with interior peak α≈0.4–0.5. α=1.0 (≡ risk-neutral, CVaR₁=mean) is
**−11.5 reward (4.6σ)** vs α=0.5 and worst on every safety KPI. This REVERSES
the old N-fixed sweep's "reward∝α" — that trend was 100% the k-confound. Top
three (0.3/0.4/0.5) tied within noise; **α* = 0.50** chosen (tightest std,
cleanest interpretation = CVaR of the worse half, already confirmed at 2000ep).
Dirs: `bmk_2026-05-30_{1,3,2,4}_alpha_v3_a{030,040,070,100}_k13` (α=0.50 reuses
N-sweep N=25). Configs: `configs/{n_sweep_a050,alpha_v3}/`.

## Three-way finals (5 seeds × 2000 ep, mean ± std)

| KPI | DQN | QR-DQN RN (α=1) | QR-DQN RA (α=0.5) | LTM [1] | LTM-CMAB [1] |
|---|---|---|---|---|---|
| reward          | 1068.0±2.9 | 1081.4±1.3 | **1086.0±1.6** | — | — |
| hof_rate        | 2.233 | 1.496 | **1.147** | 1.10 | 0.97 |
| rlf_rate        | 0.145 | 0.115 | **0.088** | 0.068 | 0.101 |
| capacity        | 3.911 | 3.954 | **3.972** | 3.75 | 3.68 |
| reliability %   | 95.85 | 96.11 | **96.26** | 95.0 | 94.0 |
| ho_rate         | **13.82** | 14.83 | 15.51 | 11.0 | 8.0 |
| pp_rate         | **1.840** | 1.983 | 1.986 | 3.45 | 1.3 |
| prep_rate       | **1009.9** | 1041.2 | 1056.7 | 780 | 20 |
| res_reservation%| **8.015** | 8.264 | 8.387 | 5.7 | 6.5 |

Dirs: DQN `bmk_2026-05-28_2`, RA `bmk_2026-05-28_1` (both reused), RN
`bmk_2026-05-30_5_riskneutral_final_2000ep` (new).

## Conclusion — two cleanly separable gains

- **Distributional gain (DQN → RN):** +13.4 reward (**9.6σ**), hof −33%, rlf −21%,
  capacity & reliability up. Same mean policy + architecture — the distributional
  learning target alone.
- **Risk gain (RN → RA):** +4.6 reward (**5.1σ**), hof −23%, rlf −23%, at identical
  capacity (k=13). Pure mean→CVaR₀.₅ policy switch.
- Every objective/safety KPI improves monotonically DQN→RN→RA; every
  activity/overhead KPI rises monotonically (the cost). Risk-aware QR-DQN is
  more willing to act but its handovers fail half as often as DQN (−49% hof,
  −39% rlf) and yield higher throughput.
- **vs prior LTM [1]** (same bare-LTM substrate, no RIS/LMMSE/CMAB): +5.9%
  capacity, +1.26pt reliability, −42% ping-pong, matches hof (1.147 vs 1.10);
  costs more HOs/preps/reservation and slightly higher rlf. RIS/LMMSE/CMAB are
  orthogonal (signal/target-selection, not value estimate) → stackable, future work.

## Artifacts updated
- NEW `results/final_metrics/qrdqn_riskneutral_summary.csv`.
- `src/tools/generate_final_plots.py`: added QR-DQN-RN series; relabelled
  qrdqn→"QR-DQN-RA (Ours)" (9-agent master plots).
- Regenerated `results/final_metrics/master_{bar_plots,radial_plot}.png`.
- Paper: filled `notes/draft-paper.md` → "## Results on the LTM Handover
  procedure" (Tables III/IV/V + de-confounding study + 3-way + vs-prior-work).
- `qrdqn_summary.csv` (α=0.50) and `dqn_summary.csv` unchanged (still canonical).
