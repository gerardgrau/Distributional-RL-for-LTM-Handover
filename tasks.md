# Feina feta:
- [x] Hardcore Benchmark (500 ep, multi-seed) amb estat de 88 dimensions.
- [x] Refactorització a arquitectura modular (`src/distrl/`).
- [x] Optimització de rendiment (23x més ràpid, pre-calculat i cache de trajectòries).

# Feina a fer:

1. QR-DQN: triar millor acció en funció del risc (mitjana dels k quantils més petits)
2. Calcular les 8 mètriques (gràfics del paper CMAB HO) (ja implementat a ltm-env.py, falta visualitzar-les al benchmark)
3. Canviar numero de quantils (N) de qr-dqn

# Possible feina a fer:
- Canviar hiperparàmetres del reward (alpha_ho...)
- Intentar buscar alguna mètrica absoluta a maximitzar (e.g. HOF x PP) - COMPLICAT
