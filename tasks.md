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
- [x] Definitive Benchmark Comparison (RL vs Legacy).

# Feina a fer:
- Executar el benchmark definitiu (5000 ep) amb les noves mètriques i CVaR.
- Benchmark amb diferents #quantiles
Anar pensant títol, subtítol, estructura del document a entregar


# Possible feina a fer:
- Canviar hiperparàmetres del reward (alpha_ho...)
- Intentar buscar alguna mètrica absoluta a maximitzar (e.g. HOF x PP) - COMPLICAT


# Preguntes:
Train / test split per les 8 mètriques finals? Quins usuaris?
- Entrenar amb tots els 1000 (i evaluar amb tots els 1000 quan acabi, congelar model i eps=0)

