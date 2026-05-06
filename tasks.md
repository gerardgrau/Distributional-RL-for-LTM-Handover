# Feina feta:
- [x] Hardcore Benchmark (500 ep, multi-seed) amb estat de 88 dimensions.
- [x] Refactorització a arquitectura modular (`src/distrl/`).
- [x] Optimització de rendiment (23x més ràpid, pre-calculat i cache de trajectòries).
- [x] Calcular les 8 mètriques (ja implementat i integrat al benchmark).
- [x] QR-DQN: triar millor acció en funció del risc (CVaR implementat).
- [x] Canviar numero de quantils (N) de qr-dqn (Configurable).

# Feina a fer:

1. Executar el benchmark definitiu (5000 ep) amb les noves mètriques i CVaR.
com es calcula ho_rate i hof_rate? (tinc més `hof` que `ho`!)
Anar pensant títol, subtítol, estructura del document a entregar


# Possible feina a fer:
- Canviar hiperparàmetres del reward (alpha_ho...)
- Intentar buscar alguna mètrica absoluta a maximitzar (e.g. HOF x PP) - COMPLICAT


# Preguntes:
Train / test split per les 8 mètriques finals? Quins usuaris?
- Entrenar amb tots els 1000 (i evaluar amb tots els 1000 quan acabi, congelar model i eps=0)

