# Overnight queue state (live)

Status legend: [Q]ueued [R]unning [D]one [F]ailed [K]illed

## Active (running)
- [R] cvar_trunc_N250_a010_matchedK → (new bmk) (bg id ba0wjj76v)
- [R] cvar_a005_N25 → (new bmk) (bg id blxmt30ys)

## Killed (low-value, reclaimed compute)
- [K] midpoint_N200 (bg id br9d163ik) — killed at ep ~485, flat axis confirmed by N=10/25/50/100

## Done
- [D] midpoint_N10  → bmk_2026-05-24_1_n_sweep_mid_N10   reward=1043.2±6.8 (35 min)
- [D] midpoint_N50  → bmk_2026-05-24_2_n_sweep_mid_N50   reward=1042.3±7.2 (69 min)
- [D] midpoint_N100 → bmk_2026-05-24_3_n_sweep_mid_N100  reward=1043.5±9.1 (147 min)
- [D] cvar_trunc_N25_a010  → bmk_2026-05-24_5_n_sweep_ct_N25_a010
       reward=1030.4±1.5, hof=1.15 (38 min)
- [D] cvar_trunc_N50_a010  → bmk_2026-05-24_6_n_sweep_ct_N50_a010
       reward=1031.7±5.7, hof=1.14 (39 min, ~identical to N=25)
- [D] cvar_trunc_N100_a010 → bmk_2026-05-24_7_n_sweep_ct_N100_a010
       reward=1035.5±3.0, hof=1.10 (40 min, monotonic up vs N=25/50)
- [D] cvar_trunc_N200_a010 → bmk_2026-05-24_8_n_sweep_ct_N200_a010
       reward=1031.5±8.9, hof=1.05 (44 min, NEW HOF leader)

## Tier 1 summary
- midpoint axis FLAT (N=10/25/50/100: reward ~1042-1050)
- cvar_trunc axis: N=100 wins on REWARD (1035.5), N=200 wins on HOF (1.05)
- N* for Tier 2/3 sweeps = N=25 (cheap) initial, N=100 if interesting
- [Q] cvar_trunc_N25_a010  → configs/n_sweep/cvar_trunc_N25_a010.yaml
- [Q] cvar_trunc_N50_a010  → configs/n_sweep/cvar_trunc_N50_a010.yaml
- [Q] cvar_trunc_N100_a010 → configs/n_sweep/cvar_trunc_N100_a010.yaml
- [Q] cvar_trunc_N200_a010 → configs/n_sweep/cvar_trunc_N200_a010.yaml

## Queue (Tier 2 — CVaR alpha sweep; regenerate at N* if needed)
- [Q] cvar_trunc_a005 → configs/cvar_alpha/cvar_trunc_a005.yaml
- [Q] cvar_trunc_a020 → configs/cvar_alpha/cvar_trunc_a020.yaml
- [Q] cvar_trunc_a030 → configs/cvar_alpha/cvar_trunc_a030.yaml
- [Q] cvar_trunc_a050 → configs/cvar_alpha/cvar_trunc_a050.yaml

## Queue (Tier 3 — kappa sweep at (N*, alpha*))
- [Q] cvar_trunc_k025 → configs/kappa_study/cvar_trunc_k025.yaml
- [Q] cvar_trunc_k05  → configs/kappa_study/cvar_trunc_k05.yaml
- [Q] cvar_trunc_k20  → configs/kappa_study/cvar_trunc_k20.yaml

## Queue (Tier 4 — DQN HP refresh)
- [Q] dqn_hard_update → configs/dqn_hp/dqn_hard_update.yaml      (--agents dqn)
- [Q] dqn_lr_3e-4    → configs/dqn_hp/dqn_lr_3e-4.yaml          (--agents dqn)
- [Q] dqn_tau_005    → configs/dqn_hp/dqn_tau_005.yaml          (--agents dqn)

## Queue (Tier 5 — n-step on both)
- [Q] nstep_3 → configs/n_step/nstep_3.yaml  (--agents dqn,qrdqn)
- [Q] nstep_5 → configs/n_step/nstep_5.yaml  (--agents dqn,qrdqn)

## Queue (Tier 6 — Quadrature v2)
- [Q] qmode_simpson       → configs/quadrature_v2/qmode_simpson.yaml
- [Q] qmode_beta_equal    → configs/quadrature_v2/qmode_beta_equal.yaml
- [Q] qmode_beta_weighted → configs/quadrature_v2/qmode_beta_weighted.yaml

## Queue (Tier 7 — Rainbow loose ends)
- [Q] grad_clip_10  → configs/hp_refresh_v2/grad_clip_10.yaml
- [Q] adam_eps_15e-4 → configs/hp_refresh_v2/adam_eps_15e-4.yaml

## Queue (Tier 8 — Matched-K paper comparison)
- [Q] cvar_trunc_N250_a010_matchedK → configs/optional_v2/cvar_trunc_N250_a010_matchedK.yaml

## Queue (Optional — only if time permits)
- [Q] cvar_trunc_N500_a010 → configs/optional_v2/cvar_trunc_N500_a010.yaml
- [Q] cvar_full_N100       → configs/optional_v2/cvar_full_N100.yaml
- [Q] gamma_095            → configs/optional_v2/gamma_095.yaml
- [Q] hidden_256           → configs/optional_v2/hidden_256.yaml
- [Q] train_freq_1         → configs/optional_v2/train_freq_1.yaml

## Headline (after all tiers; config built at runtime from joint optimum)
- [Q] headline_winner → ad-hoc config, 5 seeds x 1000 ep, --agents qrdqn
- [Q] 2000ep_headline → ad-hoc config, 5 seeds x 2000 ep (if time)

## Launch protocol

When a [R] task completes (notification fires):
1. Update its line to [D] and record bmk dir.
2. Run `./venv-RL/bin/python3 src/tools/summarize_bmk.py <bmk_dir>` for metrics.
3. Append to notes/overnight_2026-05-24.md "Variant log" + leaderboards.
4. Mark next [Q] item as [R] and launch with run_in_background.
5. Maintain target: 2 concurrent main.py invocations (load ≤20/22).
