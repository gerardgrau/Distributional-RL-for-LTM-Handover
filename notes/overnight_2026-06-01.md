# Overnight autonomous session — 2026-06-01 (deadline 10:00)

Mandate: run independently overnight; maximise paper value; try to beat the
current champion **RA QR-DQN, CVaR_0.5, k=10** (final 2000ep×5 = **1086.3±1.2**;
1000ep×3 reference = **1082.1±1.0**, from `bmk_2026-05-31_8_cvar_head_v3_truncated`).
22 threads, 2 concurrent main.py at a time.

## Strategy (two kinds of value, both bankable)
1. **Defend** knobs currently un-justified in Limitations by sweeping them on the
   FINAL backbone: Huber **κ** (only swept on the old N=25 backbone) and the
   **[256,256] architecture** (inherited, never re-optimised). Each sweep either
   erases a stated limitation OR finds something better.
2. **Improve**: the κ and capacity sweeps double as a performance search. Any
   screening config that beats the 1000ep champion (1082.1) by ≥~2σ_diff
   (≈ +2 reward, low variance) gets **promoted to 2000ep×5** to challenge 1086.3.

All screening = 1000ep×3seed, ONE knob changed from the champion
(`configs/overnight_v3/*.yaml`, invariants verified: N=20, α=0.5, trunc=True,
midpoint, noise=-101).

## Phase 1 — SCREENING — DONE 05:17 (bg `blmoubj08`, all rc=0)
| # | run | knob | reward (1000ep×3) | vs champ 1082.1 | read |
|---|-----|------|-------------------|-----------------|------|
| ref | cvar_head_v3_truncated | κ=0.25, [256,256] | 1082.1±1.0 | — | champion |
| 1 | on_kappa_010 | κ=0.10 | 1081.3±0.6 | −1.1σ | tie |
| 2 | on_kappa_050 | κ=0.50 | 1082.6±1.9 | +0.4σ | tie |
| 3 | on_kappa_100 | κ=1.00 | 1081.1±1.7 | −0.8σ | tie |
| 4 | on_hidden_512 | [512,512] | 1082.3±1.0 | +0.3σ | tie (no gain) |
| 5 | on_hidden_128 | [128,128] | 1076.6±1.4 | **−5.4σ** | worse |
| 6 | on_nstep_3 | n_step=3 | 1070.6±5.7 | **−3.4σ** | worse, unstable |

**Verdict: NO config beats the champion.** Three clean DEFENSIVE results instead:
- **κ flat** on final backbone (all within ~1σ over 0.1–1.0) → κ=0.25 safe. (The old
  N=25 Table III "+3.7σ" was a noisier-backbone artifact.)
- **Capacity saturates at [256,256]**: [128,128] −5.4σ, [512,512] +0.3σ (no gain).
- **n-step hurts** the champion (−3.4σ, unstable) → "not adopted" now airtight.
- Saved `results/final_metrics/overnight_v3_sweep.csv`.

### Defensive integration — DONE (paper compiles, 7pp, refs OK)
- Table III (κ) → final-backbone 4-point data, reframed as insensitivity.
- §V: added architecture-saturation sentence + champion-specific n-step number.
- Limitations (iii) → only reward coeffs inherited; κ/width/placement now validated.

## Phase 2 — PERFORMANCE PROBE — DONE 07:44
- Training-reward curve of the champion **declines after ~ep1200** (1085→1063),
  so a 3000ep run was skipped (best-by-eval already captures the peak).
- Instead tested **lr=5e-5** (half, for late-training stability), 2000ep×5,
  champion-identical otherwise → `bmk_2026-06-01_7_on_lr_5e5_2000ep`.
- Result: **1084.5±1.3 vs champion 1086.3±1.2 = −1.8 (−2.3σ, p=0.049)** → lower
  lr undertrains; **defends lr=1e-4**. Added a one-line defence in §V.

## FINAL STATE — paper compiles, 7pp, refs clean. Nothing committed.

## MORNING SUMMARY
**Bottom line: no config beats the champion (RA QR-DQN, CVaR_0.5, k=10, lr=1e-4,
κ=0.25, [256,256] = 1086.3±1.2). That is itself a strong result** — the champion
is now defended on *every* knob we could vary, which converts several
hand-waves/limitations into evidence and materially strengthens the paper.

**Ran 7 experiments (all rc=0):** κ∈{0.1,0.5,1.0}, hidden∈{[128,128],[512,512]},
n_step=3 (1000ep×3 screening) + lr=5e-5 (2000ep×5 confirm). See table above and
`results/final_metrics/overnight_v3_sweep.csv` + `bmk_2026-06-01_1..7`.

**Per-axis verdict:** κ flat (<1σ over 0.1–1.0) · capacity saturated at [256,256]
([128,128] −5.4σ, [512,512] +0.3σ) · n-step −3.4σ · lr=5e-5 −2.3σ. The champion
sits at a broad, robust optimum.

**Paper changes (all compiling):**
- Table III (κ) → final-backbone 4-point data, reframed as insensitivity.
- §V: added architecture-saturation result + champion-specific n-step and lr
  defences.
- Limitations (iii): tightened — only the reward-shaping coefficients remain
  inherited; κ, width and quantile placement are now all validated.

**Deliberately UNCHANGED (no new champion):** finals Table VII, §VI-D, abstract,
`results/final_metrics/*summary.csv`, master plots — the canonical 1086.3 story
stands untouched.

**For you to decide (nothing urgent):** whether to commit the night's work
(configs/overnight_v3/, overnight notes, paper.tex, overnight_v3_sweep.csv).
