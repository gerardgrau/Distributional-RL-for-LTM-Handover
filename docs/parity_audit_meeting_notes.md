# Meeting Notes: Simulation Parity Audit

## Goal
The goal of this audit was to find out why our modular RL simulation results were diverging from the paper's results, specifically regarding:
- **HO Rate:** Ours was ~7 (Paper: 11.0)
- **Prep Rate:** Ours was ~80 (Paper: 780.0)
- **RLF Rate:** Ours was 0.00 (Paper: 0.068)
- **Reliability:** Ours was 99.4% (Paper: 95.0%)

Below is the summary of the changes we tested to fix these discrepancies. They are strictly divided into **Sustained Changes** (mathematical/structural fixes we are keeping) and **Reverted/Doubtful Changes** (hyperparameter assumptions we reverted to match the teacher's original code, which you need to ask your tutor about).

---

## 1. SUSTAINED CHANGES (Kept in our codebase)
These changes fix mathematical errors or structural differences between our Gym environment and the `legacy_simulation.py` code. **These affect the Baseline and/or all algorithms as noted.**

*   **Temporal Resolution (10ms vs 100ms)**
    *   *The Problem:* The paper requires a 40ms continuous preparation time before a handover. Our RL environment evaluates decisions every 100ms. Because 100ms > 40ms, our agent was preparing and jumping cells instantly in a single step, effectively teleporting out of dead zones.
    *   *The Fix:* We decoupled the baseline. The RL environment still ticks at 100ms, but the Baseline Agent now evaluates the signal internally every 10ms.
    *   *Impact:* **Affects Baseline Only.** This mathematically enforces the 40ms preparation delay. This immediately fixed our Preparation Rate (now ~838, very close to the paper's 780).
*   **Using the `ChannelBS2UE_noRIS` Matrix**
    *   *The Problem:* We were using the clean `ChannelBS2UE` data matrix. The paper states the LTM baseline suffers a 20dB signal degradation in specific 40x40m blockage zones.
    *   *The Fix:* We updated `preprocess_dataset.py` to use the `noRIS` matrix, which has these 20dB blockage drops pre-baked into the data.
    *   *Impact:* **Affects All Algorithms.** It faithfully reproduces the physical environment constraints the authors designed.
*   **L1 RSRP for Execution Trigger**
    *   *The Problem:* We were using the heavily smoothed L3 RSRP signal for both preparation *and* execution. This made the agent too slow to react, giving us an HO Rate of ~7 (Paper: 11.0).
    *   *The Fix:* We updated the agent to use the raw, volatile L1 RSRP for the final handover execution trigger (while keeping L3 for preparation).
    *   *Impact:* **Affects Baseline Only.** This matches standard 3GPP/LTM fast-switching protocols and the active logic in `legacy_simulation.py`. It brought our HO rate up to ~13.7.
*   **Reporting Cycle Parity for Prep Rate**
    *   *The Problem:* Our `metrics.py` was counting the *number of transitions* into the prepared state (~80/min). The legacy code simply averages the boolean state over time.
    *   *The Fix:* We matched the legacy code's mathematical aggregation.
    *   *Impact:* **Affects All Algorithms.** It ensures we are comparing "apples to apples" when looking at the paper's 780.0 events/min.

---

## 2. DOUBTFUL CHANGES (Reverted - Questions for the Tutor)
These are parameters we tested that pushed our metrics closer to the paper, but we **reverted them back** to the active `legacy_simulation.py` defaults because they contradicted the paper's text or were found in commented-out blocks.

If your tutor says "yes, change them", **they will affect ALL algorithms.**

**Question 1: Radio Link Failures & Fast Fading (The 0.068 Gap)**
*   *Context:* Even with the 20dB blockages and perfect 10ms delays, our RLF rate is **0.00**. We ran the authors' original `legacy_simulation.py` script directly on the dataset and it also produced an RLF rate of nearly zero (**0.008**).
*   *Ask the Tutor:* *"How was the 0.068 RLF rate generated? Did the final simulation use a different set of UE trajectories where users get trapped in dead zones longer? Or was an artificial 'Fast Fading' (Rayleigh) noise model added to the SNIR to force deeper drops?"*

**Question 2: The SINR Outage Table (-6.5dB vs -3.0dB)**
*   *Context:* In `legacy_simulation.py`, the active table triggers an Outage (`MCS=0`) at **-6.5dB**. However, there is a commented-out table that triggers Outage much earlier, at **-3.0dB**. When we tested the -3.0dB table, our Reliability dropped from 98.7% to 97.8%, getting much closer to the paper's 95.0%.
*   *Ask the Tutor:* *"Which SINR-to-MCS mapping table was used for the final publication? Was it the one that cuts off at -6.5dB or the commented one that cuts off at -3.0dB?"*

**Question 3: Tx Power (25 dBm vs 45 dBm)**
*   *Context:* The paper text explicitly states the Tx Power is **25 dBm**. However, the active code in `legacy_simulation.py` sets `"TxPower": 45`, with a comment next to it saying `"# Al paper està a 25"`.
*   *Ask the Tutor:* *"Should we train our RL agents using 25 dBm (as published) or 45 dBm (as active in the provided code)? At 45 dBm, the signal is so strong that it almost completely masks thermal noise."*

**Question 4: Thermal Noise Calculation (Bandwidth)**
*   *Context:* The `NoiseLevel` is defined as **-174 dBm**. Technically, this is a power spectral density (dBm/Hz) and should be multiplied by the 200MHz bandwidth to get the true noise floor. The legacy code does *not* multiply by bandwidth, treating -174 dBm as the total noise.
*   *Ask the Tutor:* *"Should we multiply the -174 dBm noise level by the 200MHz bandwidth to get the true noise floor, or was the simulation run assuming -174 dBm as the total noise?"*