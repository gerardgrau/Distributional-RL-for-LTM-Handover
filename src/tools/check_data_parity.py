import numpy as np
import scipy.io
import os

def check_data():
    user_id = 1
    mat_path = f"data/ChannelGains/ChannelGainBSUE_User{user_id}.mat"
    npz_path = f"data/Precomputed/User{user_id}_precomputed.npz"
    
    mat_data = scipy.io.loadmat(mat_path)
    raw_mat = mat_data['ChannelBS2UE_noRIS']
    
    with np.load(npz_path) as data:
        raw_npz = data['ch_bs2ue']
        
    # Re-order mat to match npz (21, T)
    NBS = 21
    T = raw_mat.shape[0]
    raw_mat_reshaped = np.zeros((NBS, T))
    idx = 0
    for b in range(raw_mat.shape[1]):
        for s in range(raw_mat.shape[2]):
            raw_mat_reshaped[idx, :] = raw_mat[:, b, s]
            idx += 1
            
    diff = np.abs(raw_mat_reshaped - raw_npz)
    print(f"Max raw diff: {np.max(diff)}")
    print(f"Mean raw diff: {np.mean(diff)}")

if __name__ == "__main__":
    check_data()
