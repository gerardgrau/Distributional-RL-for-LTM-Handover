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
- Risk-aware avaluació: passar CVaR `k ∈ {0.05, 0.1, 0.25, 0.5}` per veure
  l'impacte real (és el sweep ja implementat a `tools/run_cvar_study.py`).
- Millorar plots de quantils per als models de risc.
- Mètrica composta (e.g. HOF · PP) a maximitzar — complicat, posposable.
- Escriure el document final: títol, subtítol, estructura.

# Configuració de Paritat Final (actualitzada 2026-05-16):

- **Loop**: 10ms High-Resolution
- **TxPower**: **25 dBm** (Table I) ✓
- **NoiseLevel**: **-174** (codi: linear floor; paper: density dBm/Hz) ⚠️ veure pregunta sota
- **ExecPowerOffset**: **3.0 dB** (Table II) ✓
- **MaxNumberPreparedBS**: **4** (Table II) ✓ — actualitzat de 5→4 el 2026-05-16
- **Channels**: `ChannelBS2UE_noRIS` (amb les obstruccions de 20 dB) ✓
- **Reward α**: HOF=0.1, HO=0.8, PP=0.9 (secció III.B) ✓

Resultats 1000 UEs (LTMBaselineAgent en LTMEnv, MaxPrep=4, 2026-05-16):
ho_rate=11.36 · hof_rate=2.06 · pp_rate=3.22 · capacity=3.23 ·
rlf_rate=0.19 · reliability=94.35% · prep_rate=810.04 · res_reservation=6.43%

Δ vs run anterior amb MaxPrep=5: prep_rate −2.36 i res_reservation −0.02
(la resta inalterada). Confirmat que el canvi 5→4 només toca prep_rate
i res_reservation lleugerament, i no tanca els forats grans amb el paper.

Gaps que persisteixen vs paper LTM:
- capacity_avg: 3.23 vs 3.75 (−14%)
- hof_rate: 2.06 vs 1.10 (1.9× alt)
- rlf_rate: 0.19 vs 0.068 (2.8× alt)
- reliability_pct: 94.35 vs 95.00 (−0.65 pts)

Aquests gaps quadren amb la hipòtesi del noise floor (pregunta 2 sota):
soroll efectiu massa baix faria SNIR més alta, però l'efecte real depèn
de com les taules SINR/BLER van ser calibrades. Cal validar amb el tutor.

# Preguntes per al tutor (paritat física amb el paper):

1. **MaxNumberPreparedBS — paper diu 4, codi de referència diu 5.**
   Table II del paper: "Target cell pre-selection | Predicted top-4 neighbors".
   El codi `docs/reference/ltm_ho_codi_ainna.py` (l. 75) fixa `MaxNumberPreparedBS = 5`.
   He posat 4 al codi (config.yaml, physics.py, legacy_simulation.py) per
   coherència amb el paper. Confirmar.

2. **Noise floor — paper diu densitat -174 dBm/Hz, codi fa servir -174 com a
   floor lineal directe sense integrar sobre l'ample de banda.**
   Table I del paper: "Noise power density | -174 dBm/Hz". Amb BW=200 MHz,
   això integrat seria N0·B = -174 + 10·log10(200e6) ≈ **-91 dBm**.
   Però `physics.py` i `legacy_simulation.py` fan `noise_linear = 10^(-174/10)`
   directament — tractant-lo com a noise floor lineal en dBm, no com a
   densitat. Això fa la diferència de ~10⁸ en la potència de soroll vs senyal,
   convertint el sistema en pràcticament noise-free (interference-limited).
   Possibilitats:
   - El codi de referència ho fa malament i caldria canviar a -91 dBm.
   - Les taules SINR (16 esglaons) ja estan calibrades assumint aquest floor
     artificial i canviar-lo trencaria la paritat.
   - El paper Table I és una notació informal i el valor numèric -174 ja
     és el que es fa servir directament.
   Pendent decidir amb el tutor abans de tocar res.

3. **Bandwidth 200 MHz — no apareix explícitament al codi.**
   Connecta amb el punt 2 (integració del soroll). Si fem -91 dBm com a
   noise floor, ja s'aplica implícitament. Si no, BW no s'utilitza enlloc.
