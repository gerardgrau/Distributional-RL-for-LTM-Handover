import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.distrl.envs.physics import (
    System, NBS, vectorized_oracle, vectorized_hof
)

def get_hof_prob(serving_sector, channels, System):
    """
    Simplified reference implementation of HOF probability math for parity checking VectorizedHOF.
    """
    BLER = np.array([
        1, 0.2617, 0.2370, 0.2103, 0.1828, 0.1558, 0.1302, 0.1067, 0.0859,
        0.0678, 0.0526, 0.0401, 0.0301, 0.0221, 0.0160, 0.0114, 0.0080,
        0.0056, 0.0038, 0.0025, 0.0017, 0.0011, 0.0007, 0.0004, 0.0003,
        0.0002, 0.0001
    ])
    SNR_level = np.array([
        -np.inf, -1.7609, -1.6609, -1.5609, -1.4609, -1.3609, -1.2609,
        -1.1609, -1.0609, -0.9609, -0.8609, -0.7609, -0.6609, -0.5609,
        -0.4609, -0.3609, -0.2609, -0.1609, -0.0609, 0.0391, 0.1391,
        0.2391, 0.3391, 0.4391, 0.5391, 0.6391, 0.7391
    ])
    
    # Simple mock of interference logic to match VectorizedHOF
    Ps = 10 ** ((channels[serving_sector] + System["TxPower"]) / 10.0)
    group_sums = np.zeros(3)
    linear_powers = 10 ** ((channels + System["TxPower"]) / 10.0)
    for g in range(3):
        group_sums[g] = np.sum(linear_powers[g::3])
    AllInter = group_sums[serving_sector % 3] - Ps
    
    # Neighbor logic for ICIC
    neighbor_indices = np.delete(np.arange(len(channels)), serving_sector)
    target_idx = neighbor_indices[np.argmax(channels[neighbor_indices])]
    serving_pwr = channels[serving_sector]
    target_pwr = channels[target_idx]
    ho_margin_db = 10 * np.log10(serving_pwr / (target_pwr + 1e-15))
    icic_active = ho_margin_db < 7.0
    
    noise_floor = 10**(System["NoiseLevel"] / 10.0)
    reduction_factor = 0.1
    if icic_active:
        inter_noise = (AllInter * reduction_factor) + noise_floor
    else:
        inter_noise = AllInter + noise_floor
        
    SNIR = 10 * np.log10(Ps) - 10 * np.log10(inter_noise)
    idx = np.where(SNR_level <= SNIR)[0]
    return BLER[idx[-1]]

def test_parity(num_trials=1000):
    print(f"Starting Advanced Parity Test: {num_trials} trials...")
    
    sync = {"N310": 4, "N311": 2, "T310": 50, "out_sync_count": 0, "in_sync_count": 0, "t310_running": False, "t310_counter": np.inf}
    discrepancies = 0
    
    for t in range(num_trials):
        channels = np.random.uniform(-120, -40, size=NBS)
        
        # 1. Oracle Parity
        orig_mcs = np.zeros(NBS)
        orig_snir = np.zeros(NBS)
        for i in range(NBS):
            m, _, _, s = MCSEvaluation(i, channels, System, sync.copy())
            orig_mcs[i] = m
            orig_snir[i] = s
            
        vec_mcs, vec_snir = VectorizedOracle(channels, System)
        snir_diff = np.abs(orig_snir - vec_snir)
        mcs_match = np.array_equal(orig_mcs, vec_mcs)
        
        # 2. HOF Parity
        orig_pe = np.zeros(NBS)
        for i in range(NBS):
            orig_pe[i] = get_hof_prob(i, channels, System)
            
        vec_pe = VectorizedHOF(channels, System)
        pe_diff = np.abs(orig_pe - vec_pe)
        
        if not mcs_match or np.any(snir_diff > 1e-6) or np.any(pe_diff > 1e-10):
            discrepancies += 1
            if discrepancies <= 3:
                print(f"\nDiscrepancy in Trial {t}:")
                if not mcs_match: print(f"  MCS mismatch!")
                if np.any(snir_diff > 1e-6): print(f"  SNIR Diff Max: {np.max(snir_diff)}")
                if np.any(pe_diff > 1e-10): print(f"  Pe Diff Max: {np.max(pe_diff)}")
    
    if discrepancies == 0:
        print("\nSUCCESS: 100% Parity Verified for Oracle and HOF.")
    else:
        print(f"\nFAILURE: Found {discrepancies} discrepancies.")
        sys.exit(1)

if __name__ == "__main__":
    test_parity()
