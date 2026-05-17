# Feina feta:
- [x] Hardcore Benchmark (500 ep, multi-seed) amb estat de 88 dimensions.
- [x] Refactorització a arquitectura modular (`src/distrl/`).
- [x] Optimització de rendiment (23x més ràpid, pre-calculat .npz d'alta velocitat).
- [x] Calcular les 8 mètriques (ja implementat i integrat al benchmark).
- [x] QR-DQN: triar millor acció en funció del risc (CVaR implementat).
- [x] Canviar numero de quantils (N) de qr-dqn (Configurable).
- [x] Optimització de l'entrenament (4.2x més ràpid, train_freq=4, pinned memory, pre-allocated tensors).
- [x] Fix HOF calculation
- [x] Programar baseline (algoritme hardcodejat legacy).
- [x] Simplificació de codi (Consolidació de física, refactorització d'agents).
- [x] Definitive Benchmark Comparison (RL vs Legacy Baseline).
- [x] Merge all optimizations and fixes to master.
- [x] Calibració paritat física (paper Table I / II) — 2026-05-15.
- [x] Reward function pure-Ainna (PR #27).
- [x] Dashboard: BS positions hexagonals + sector patterns + MCS panel + HO markers.

# Feina oberta:
- Risk-aware avaluació: passar CVaR `k ∈ {0.05, 0.1, 0.25, 0.5}` per veure l'impacte real (és el sweep ja implementat a `tools/run_cvar_study.py`).
- Millorar plots de quantils per als models de risc. ==> ensenyar com s'està fent la tria de coses, quin percentatge es fa servir. Mostrar el "Q-value" (e.g. la mitjana, o el CVaR) al plot
- Mètrica composta (e.g. HOF · PP) a maximitzar — complicat, posposable.
- Escriure el document final: títol, subtítol, estructura.

- Quantile positioning (e.g. non-uniform, gaussian weights, only use bottom k quantiles (with risk mode)???)
- Provar els meus models al atari gym (per tenir una idea de com de bé funcionen allà). Fer un benchmark dels models de risc i de les quadratures no uniformes


# Tasques que potser no em dona temps:
- Provar xarxes FQF, IQN?




# Paritat amb el paper — estat actual (2026-05-16)

**Constants alineades amb el paper:**
- TxPower 25 dBm (Table I), ExecPowerOffset 3.0 dB (Table II),
  MaxNumberPreparedBS 4 (Table II) — actualitzat de 5→4 avui,
  Channels `ChannelBS2UE_noRIS` (Table I), Reward α HOF/HO/PP = 0.1/0.8/0.9
  (secció III.B), Receiver sensitivity -95 dBm, time resolution 10 ms,
  300 s × 1000 UEs.
- NoiseLevel **-174** (mantingut, veure experiment sota).

**Resultats LTMBaselineAgent vs paper (1000 UEs, 10 ms high-res):**

| Mètrica          | Paper  | Codi  | Gap     |
|------------------|--------|-------|---------|
| ho_rate          | 11.00  | 11.36 | ≈ ✓     |
| pp_rate          | 3.45   | 3.22  | ≈ ✓     |
| prep_rate        | 780    | 810   | +4%     |
| res_reservation  | 5.70%  | 6.43% | +0.7 pt |
| capacity_avg     | 3.75   | 3.23  | **−14%**|
| hof_rate         | 1.10   | 2.06  | **1.9×**|
| rlf_rate         | 0.068  | 0.19  | **2.8×**|
| reliability_pct  | 95.00  | 94.35 | −0.65 pt|

## Experiment de noise floor (2026-05-16) — descartat

El paper Table I dóna densitat **−174 dBm/Hz**, que integrat sobre BW=200 MHz
seria **−91 dBm**. El codi (referència i nostre) fa servir −174 directament
com a noise floor lineal en dBm. Vam testar la hipòtesi canviant
NoiseLevel: −174 → −91, regenerant el cache `no_ris` complet i re-corrent
verify_simulation_parity.py (1000 UEs):

| Mètrica          | Paper  | −174 (actual) | −91 (test) |
|------------------|--------|---------------|------------|
| capacity_avg     | 3.75   | 3.23          | **3.03**   |
| rlf_rate         | 0.068  | 0.19          | **0.23**   |
| hof_rate         | 1.10   | 2.06          | **2.12**   |
| reliability_pct  | 95.00  | 94.35         | 94.25      |
| ho/pp/prep/res   | —      | —             | sense canvi|

**Conclusió: −91 empitjora totes les mètriques.** Hipòtesi: les taules SINR
(16 esglaons) i BLER (27 nivells) del codi de referència ja estan calibrades
assumint floor "−174 raw" (sistema interference-limited). Mantenim −174.
El gap residual ve d'algun altre lloc (no del soroll).

## Preguntes per al tutor

1. **MaxNumberPreparedBS = 4 confirmat?** Paper Table II diu top-4; codi de
   referència `ltm_ho_codi_ainna.py` posa 5. He triat 4.

2. **El gap residual de capacity/RLF/HOF/reliability** (~14–280%) ve de:
   (a) la taula SINR/MCS de 16 esglaons,
   (b) la taula BLER de 27 nivells,
   (c) el factor `M=3` a `Inter_Noise = M · AllInter + noise` (no és estàndard),
   o d'alguna altra calibració del codi de referència no documentada?
   L'experiment de noise descarta el soroll com a causa.

3. **Interpretació del −174 dBm/Hz de Table I**: el codi el tracta com a
   floor lineal directe (no integra sobre BW). És intencionat o és que les
   taules SINR ja absorbeixen aquesta calibració?

4. **Model d'interferència `M=3` vs ICIC** — el nostre `physics.py` i
   `legacy_simulation.py` fan servir una mescla Bernoulli (1/3 high ·×3,
   2/3 low ·0.095) per modelar reuse-3, però aquest bloc està **comentat**
   al codi de referència actual (`ltm_ho_codi_ainna.py` línies 148–158).
   La referència viva fa servir `get_realistic_interference()`, un model
   ICIC: si el veí més fort és a <7 dB, redueix interferència en −10 dB
   i puja el senyal +3 dB. És a dir, **el nostre codi correspon a una
   versió antiga de la referència, no a l'activa**. Hauríem de migrar a
   ICIC? Podria explicar part del gap residual.
