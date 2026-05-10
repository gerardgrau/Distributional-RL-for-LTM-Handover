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

- **Assumpció RLF/Bloqueig i Dades Originals**: Vam descobrir que l'arxiu `.mat` conté tres matrius (`ChannelBS2UE`, `ChannelBS2UE_noRIS`, `ChannelBS2UE_RIS`). Utilitzant `noRIS` aconseguim la caiguda de 20dB a les zones bloquejades, però **els RLFs segueixen sent pràcticament zero**. Hem executat l'script original (`legacy_simulation.py`) proporcionat pels autors sobre 50 usuaris d'aquest mateix dataset i ens dona una taxa de RLF de **0.008** (molt lluny del **0.068** del paper). **Pregunta**: El 0.068 del paper s'ha calculat amb un dataset de trajectòries diferent on els usuaris es queden més temps aturats a les zones mortes? Cal afegir soroll de "Fast Fading" (Rayleigh-Jakes) artificialment a sobre d'aquestes dades perquè l'SNIR caigui de forma més agressiva?
- **Assumpció Resolució Temporal (10ms vs 100ms)**: El nostre entorn RL funciona a passes de 100ms. Si el baseline també pren decisions cada 100ms, s'escapa de les zones de bloqueig "instantàniament" (perquè preparar la cel·la triga 40ms, i amb un pas de 100ms ho fa dins del mateix pas), evitant esgotar el temporitzador T310 (500ms) i impedint els RLF. **Acció**: Hem modificat l'entorn perquè el baseline avalui el senyal i prepari les cel·les cada 10ms internament (fidel al codi original). **Pregunta**: Valideu que és correcte que els agents RL segueixin funcionant a 100ms per entrenar, però que per avaluar el Baseline justament s'hagi de fer a resolució de 10ms?
- **Assumpció Càlcul de Soroll (Bandwidth)**: Al principi la capacitat ens sortia molt alta (~4.4 vs 3.75 del paper). Al codi `legacy_simulation.py` el `NoiseLevel` és -174 dBm/Hz, però faltava aplicar l'amplada de banda per obtenir el Noise Floor real. **Acció**: Hem multiplicat pel Bandwidth (200MHz) al càlcul de l'SNIR. Amb això la capacitat ha baixat a ~3.2 bps/Hz i la reliability a ~98.4% (molt més proper als valors del paper). **Pregunta**: És correcte haver inclòs l'amplada de banda per empitjorar l'SNIR, o l'SNIR alt del principi era l'esperat pels autors?
- **Assumpció HO Trigger (L1 vs L3)**: El nostre baseline tenia una HO Rate molt baixa (~7 vs 11 del paper). Ens vam adonar que fèiem servir el senyal filtrat (L3) tant per preparar com per executar l'HO. **Acció**: Hem canviat la lògica perquè executi l'HO utilitzant **L1 RSRP** (més ràpid/inestable) però mantingui **L3 RSRP** per la preparació. Això dispara els handovers a ~13.7 (més proper a 11.0). **Pregunta**: Al codi original del paper, utilitzaven RSRP capa L1 o L3 per la condició d'execució final?
