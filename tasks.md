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

# Feina a fer:
- Executar el benchmark definitiu (5000 ep) amb les noves mètriques i CVaR.
- Benchmark amb diferents #quantiles
Anar pensant títol, subtítol, estructura del document a entregar


# Possible feina a fer:
- Canviar hiperparàmetres del reward (alpha_ho...)
- Intentar buscar alguna mètrica absoluta a maximitzar (e.g. HOF x PP) - COMPLICAT


# Preguntes resoltes:
Train / test split per les 8 mètriques finals? Quins usuaris?
- Entrenar amb tots els 1000 (i evaluar amb tots els 1000 quan acabi, congelar model i eps=0)

# Preguntes:
Quina posició (x, y) tenen les Base Stations?
Quina orientació segueixen?

- **Assumpció Taules SINR i Reliability (CONCLOSA)**: Hem provat la taula comentada (`SINRThreshold` a partir de -3dB per l'Outage). Això ha baixat la Reliability de 98.6% a **97.8%**, apropant-nos molt al 95% del paper. El gap restant són segurament els RLFs que no estem disparant. **Decisió**: Usarem la taula de -3dB per a la calibració final.
- **Assumpció ExecPowerOffset i HO Rate (CONCLOSA)**: Hem trobat que amb un offset de **3.4 dB** (en lloc dels 3.0 del paper/taula) clavem la HO Rate a **~11.5** (Paper: 11.0). El valor de 5.0 comentat al codi era massa agressiu (baixava a 7). **Decisió**: Usarem 3.4 dB per a la calibració final.
- **Assumpció TxPower i Capacitat (CONCLOSA)**: Utilitzant **45 dBm** (el valor actiu al codi legacy) pugem la capacitat a **3.41**, molt més a prop dels 3.75 del paper que no pas els 3.1 que teníem amb 25 dBm. **Decisió**: Usarem 45 dBm per a la calibració final.
- **Assumpció RLF i Dataset (PENDENT)**: Tot i clavar la HO Rate i la Prep Rate, els RLF segueixen sent 0.0. Hem confirmat que fins i tot l'script original dels autors dona 0.008 sobre aquest dataset. **Conclusió**: La taxa de 0.068 del paper depèn de dades o soroll temporal (fading) que no tenim.
- **Assumpció Resolució Temporal (VALIDADA)**: La resolució de 10ms és imprescindible per tenir paritat de temps de preparació (40ms). **Acció**: El baseline d'ara endavant s'avaluarà sempre amb el loop de 10ms.

# Configuració de Paritat Final:
- **Loop**: 10ms High-Resolution
- **TxPower**: 45 dBm
- **ExecPowerOffset**: 3.4 dB
- **SINR Table**: Alternative (Outage < -3dB)
- **Bandwidth**: 200 MHz

