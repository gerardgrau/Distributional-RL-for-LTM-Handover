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

# Configuració de Paritat Final (actualitzada 2026-05-15):

- **Loop**: 10ms High-Resolution
- **TxPower**: **25 dBm** (Table I)
- **NoiseLevel**: **-91 dBm** = -174 dBm/Hz integrat sobre 200 MHz
- **ExecPowerOffset**: **3.0 dB** (Table II)
- **MaxNumberPreparedBS**: **4** (Table II)
- **SINR Table**: la de 26 esglaons (la "Outage < -3dB" inclosa)
- **Bandwidth**: 200 MHz
- **Channels**: `ChannelBS2UE_noRIS` (amb les obstruccions de 20 dB)

Resultats 1000 UEs (baseline LTM, paritat gym ↔ legacy bit-exacta):
ho_rate=11.69 · hof_rate=0.35 · pp_rate=4.25 · capacity=2.75 ·
rlf_rate=0.20 · reliability=90.94% · prep_rate=807.24 · res_reservation=6.41%

Gap vs paper: capacity i reliability són més baixes que el paper (3.75 i 95%)
i el RLF és més alt (0.198 vs 0.068). Pendent de confirmar amb el tutor si
volem revisar la interpretació de NoiseLevel.
