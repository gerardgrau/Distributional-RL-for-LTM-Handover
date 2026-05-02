import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.distrl.envs.ltm_env import (
    System, BS, NBS, MCSEvaluation, VectorizedOracle
)

def test_parity(num_trials=1000):
    print(f"Starting Parity Test: {num_trials} trials...")
    
    # Mock Sync state (doesn't change for the Oracle loop in the original code)
    sync = {"N310": 4, "N311": 2, "T310": 50, "out_sync_count": 0, "in_sync_count": 0, "t310_running": False, "t310_counter": np.inf}

    discrepancies = 0
    
    for t in range(num_trials):
        # Generate random channel gains for 21 sectors
        # Typical RSRP values range from -120 to -40
        channels = np.random.uniform(-120, -40, size=NBS)
        
        # 1. Original Loop Logic
        orig_mcs = np.zeros(NBS)
        orig_snir = np.zeros(NBS)
        for i in range(NBS):
            m, _, _, s = MCSEvaluation(i, channels, System, sync.copy())
            orig_mcs[i] = m
            orig_snir[i] = s
            
        # 2. Vectorized Logic
        vec_mcs, vec_snir = VectorizedOracle(channels, System)
        
        # 3. Compare SNIR (Float precision)
        snir_diff = np.abs(orig_snir - vec_snir)
        mcs_match = np.array_equal(orig_mcs, vec_mcs)
        
        # We allow for very small floating point noise, but MCS must match exactly
        if not mcs_match or np.any(snir_diff > 1e-6):
            discrepancies += 1
            if discrepancies <= 3:
                print(f"\nDiscrepancy in Trial {t}:")
                print(f"  Orig MCS: {orig_mcs}")
                print(f"  Vec  MCS: {vec_mcs}")
                print(f"  SNIR Diff Max: {np.max(snir_diff)}")
                
                # Check ICIC logic specifically if SNIR differs
                # The original code deletes indices and uses argmax(delete)
                # I'll double check my sorting logic
    
    if discrepancies == 0:
        print("\nSUCCESS: 100% Parity Verified. Vectorized logic is numerically identical to scalar loops.")
    else:
        print(f"\nFAILURE: Found {discrepancies} discrepancies out of {num_trials} trials.")
        sys.exit(1)

if __name__ == "__main__":
    test_parity()
