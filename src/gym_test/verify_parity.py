import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.distrl.envs.physics import (
    System, NBS, vectorized_oracle, vectorized_hof
)

def reference_mcs_eval(serving_sector, channels, system_params):
    """
    Reference (scalar) implementation of MCS and SNIR calculation.
    """
    tx_power = system_params.get("TxPower", 25)
    noise_level = system_params.get("NoiseLevel", -174)
    
    ps_linear = 10 ** ((channels[serving_sector] + tx_power) / 10.0)
    
    # Interference (Reuse-3)
    group_sums = np.zeros(3)
    linear_powers = 10 ** ((channels + tx_power) / 10.0)
    for g in range(3):
        group_sums[g] = np.sum(linear_powers[g::3])
    all_inter_linear = group_sums[serving_sector % 3] - ps_linear
    
    # ICIC
    neighbor_indices = np.delete(np.arange(len(channels)), serving_sector)
    target_idx = neighbor_indices[np.argmax(channels[neighbor_indices])]
    target_pwr = channels[target_idx]
    
    ho_margin_db = channels[serving_sector] - target_pwr
    icic_active = ho_margin_db < 7.0
    
    noise_floor = 10**(noise_level / 10.0)
    reduction_factor = 0.1
    inter_noise = (all_inter_linear * reduction_factor + noise_floor) if icic_active else (all_inter_linear + noise_floor)
    
    snir = 10 * np.log10(ps_linear + 1e-15) - 10 * np.log10(inter_noise + 1e-15)
    
    # MCS
    idx = np.searchsorted(system_params.get("SINRThreshold"), snir, side='right') - 1
    mcs = system_params.get("SpectralEff")[max(0, idx)]
    
    return mcs, snir

def reference_hof_eval(serving_sector, channels, system_params):
    """
    Reference (scalar) implementation of Handover Failure probability math.
    """
    tx_power = system_params.get("TxPower", 25)
    noise_level = system_params.get("NoiseLevel", -174)
    
    ps_linear = 10 ** ((channels[serving_sector] + tx_power) / 10.0)
    
    # Interference (Reuse-3)
    group_sums = np.zeros(3)
    linear_powers = 10 ** ((channels + tx_power) / 10.0)
    for g in range(3):
        group_sums[g] = np.sum(linear_powers[g::3])
    all_inter_linear = group_sums[serving_sector % 3] - ps_linear
    
    # ICIC
    neighbor_indices = np.delete(np.arange(len(channels)), serving_sector)
    target_idx = neighbor_indices[np.argmax(channels[neighbor_indices])]
    target_pwr = channels[target_idx]
    
    ho_margin_db = channels[serving_sector] - target_pwr
    icic_active = ho_margin_db < 7.0
    
    noise_floor = 10**(noise_level / 10.0)
    reduction_factor = 0.1
    inter_noise = (all_inter_linear * reduction_factor + noise_floor) if icic_active else (all_inter_linear + noise_floor)
    
    snir = 10 * np.log10(ps_linear + 1e-15) - 10 * np.log10(inter_noise + 1e-15)
    
    # BLER Table (Standard 5G)
    bler = np.array([
        1, 0.2617, 0.2370, 0.2103, 0.1828, 0.1558, 0.1302, 0.1067, 0.0859,
        0.0678, 0.0526, 0.0401, 0.0301, 0.0221, 0.0160, 0.0114, 0.0080,
        0.0056, 0.0038, 0.0025, 0.0017, 0.0011, 0.0007, 0.0004, 0.0003,
        0.0002, 0.0001
    ])
    snr_level = np.array([
        -np.inf, -1.7609, -1.6609, -1.5609, -1.4609, -1.3609, -1.2609,
        -1.1609, -1.0609, -0.9609, -0.8609, -0.7609, -0.6609, -0.5609,
        -0.4609, -0.3609, -0.2609, -0.1609, -0.0609, 0.0391, 0.1391,
        0.2391, 0.3391, 0.4391, 0.5391, 0.6391, 0.7391
    ])
    
    idx = np.searchsorted(snr_level, snir, side='right') - 1
    return bler[max(0, idx)]

def test_parity(num_trials=1000):
    print(f"Starting Parity Test: {num_trials} trials...")
    discrepancies = 0
    
    for t in range(num_trials):
        channels = np.random.uniform(-120, -40, size=NBS)
        
        # 1. Oracle Parity
        orig_mcs = np.zeros(NBS)
        orig_snir = np.zeros(NBS)
        for i in range(NBS):
            m, s = reference_mcs_eval(i, channels, System)
            orig_mcs[i] = m
            orig_snir[i] = s
            
        vec_mcs, vec_snir = vectorized_oracle(channels, System)
        snir_diff = np.abs(orig_snir - vec_snir)
        mcs_match = np.array_equal(orig_mcs, vec_mcs)
        
        # 2. HOF Parity
        orig_pe = np.zeros(NBS)
        for i in range(NBS):
            orig_pe[i] = reference_hof_eval(i, channels, System)
            
        vec_pe = vectorized_hof(channels, System)
        pe_diff = np.abs(orig_pe - vec_pe)
        
        if not mcs_match or np.any(snir_diff > 1e-10) or np.any(pe_diff > 1e-10):
            discrepancies += 1
    
    if discrepancies == 0:
        print("\nSUCCESS: 100% Parity Verified.")
    else:
        print(f"\nFAILURE: Found {discrepancies} discrepancies.")
        sys.exit(1)

if __name__ == "__main__":
    test_parity()
