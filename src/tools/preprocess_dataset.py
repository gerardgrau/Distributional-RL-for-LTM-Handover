import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
from tqdm import tqdm
import sys

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.envs.physics import (
    System, Time, HO, NBS, vectorized_oracle, vectorized_hof
)

def preprocess():
    data_dir = "data/ChannelGains"
    output_dir = "data/Precomputed"
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = sorted(glob.glob(os.path.join(data_dir, "ChannelGainBSUE_User*.mat")))
    print(f"Found {len(all_files)} files to preprocess.")
    
    for filename in tqdm(all_files, desc="Preprocessing"):
        user_id = os.path.basename(filename).split('_')[-1].split('.')[0]
        out_filename = os.path.join(output_dir, f"{user_id}_precomputed.npz")
        
        mat_data = loadmat(filename)
        
        # Paper Alignment: The .mat file contains three versions:
        # 1. ChannelBS2UE: Clean environment (no blockage)
        # 2. ChannelBS2UE_noRIS: Realistic environment with 20dB spatial blockage
        # 3. ChannelBS2UE_RIS: Realistic environment with 20dB blockage + RIS assistance
        # We use 'ChannelBS2UE' for the paper parity
        raw_channel = mat_data['ChannelBS2UE'] 
        
        total_time = raw_channel.shape[0]
        ch_bs2ue = np.zeros((NBS, total_time), dtype=np.float32)
        idx = 0
        for b in range(raw_channel.shape[1]):
            for s in range(raw_channel.shape[2]):
                ch_bs2ue[idx, :] = raw_channel[:, b, s]
                idx += 1
        
        # Reproducible Stochasticity for Parity Audit
        import re
        numeric_id = int(re.search(r'\d+', user_id).group())
        np.random.seed(42 + numeric_id)
        
        # Positions
        ue_pos_complex = mat_data['UE'][0, 0]['Position'][0]
        ue_positions = np.stack([ue_pos_complex.real, ue_pos_complex.imag], axis=1).astype(np.float32)
        
        # Physics calculations
        all_mcs, all_snir = vectorized_oracle(ch_bs2ue, System)
        all_pe = vectorized_hof(ch_bs2ue, System)
        
        # Filtering (L1 and L3 RSRP)
        M = int(np.ceil(HO["Prep"]["PeriodicityRSRPMeasurement"] / Time["TimeStep"]))
        b_filt = np.ones(HO["Prep"]["AverageRSRPMeasument_NL1"]) / HO["Prep"]["AverageRSRPMeasument_NL1"]
        L1 = lfilter(b_filt, 1, ch_bs2ue[:, ::M], axis=1)
        pl1 = np.repeat(L1, M, axis=1)[:, :total_time].astype(np.float32)
        pl3 = np.repeat(lfilter(HO["Prep"]["alphaIIRfilter"], [1, -1 + HO["Prep"]["alphaIIRfilter"]], L1, axis=1), M, axis=1)[:, :total_time].astype(np.float32)
        
        # Save as compressed NumPy binary
        np.savez_compressed(
            out_filename,
            total_time=total_time,
            ch_bs2ue=ch_bs2ue,
            all_mcs_episode=all_mcs.astype(np.float32),
            all_snir_episode=all_snir.astype(np.float32),
            all_pe_episode=all_pe.astype(np.float32),
            ue_positions=ue_positions,
            pl1=pl1,
            pl3=pl3
        )

if __name__ == "__main__":
    preprocess()
