import numpy as np
from typing import Any

# ============================================================
# SYSTEM CONSTANTS (3GPP Aligned)
# ============================================================

ReceiverSensitivity = -95

# SINR-to-MCS Mapping (Standard 5G Table)
SINR_THRESHOLD = np.array([
    -np.inf, -6.5, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.5, 10.5, 
    12.5, 14.5, 16.5, 19.0, 21.5, 24.0
])

SPECTRAL_EFF = np.array([
    0, 0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41, 
    2.73, 3.32, 3.90, 4.52, 5.12, 5.55
])

# Default Physics Configuration (Legacy fallback)
System = {
    "TxPower": 25,  # dBm (Aligned with paper)
    "NoiseLevel": -174,  # dBm
    "SINRThreshold": SINR_THRESHOLD,
    "SpectralEff": SPECTRAL_EFF
}

Time = {
    "TotalSimTime": 300,
    "TimeStep": 0.01
}

BS = {"Number": 7}
NBS = 21

HO = {
    "Prep": {
        "PeriodicityRSRPMeasurement": 0.02,
        "AverageRSRPMeasument_NL1": 10,
        "kL3": 8,
        "alphaIIRfilter": 2 ** (-8 / 4),
        "PreparationPowerOffset": -3,
        "PreparationTime": 0.04,
        "ExecPowerOffset": 3,
        "MaxNumberPreparedBS": 5
    }
}

# ============================================================
# PHYSICS FUNCTIONS
# ============================================================

def calculate_snir_matrix(channels_2d: np.ndarray, system_params: dict[str, Any]) -> np.ndarray:
    """
    Computes the (NBS, T) SNIR matrix using vectorized ICIC logic.
    """
    nbs, total_t = channels_2d.shape
    tx_power = system_params.get("TxPower", 25)
    noise_level = system_params.get("NoiseLevel", -174)
    
    # 1. Powers and Interference (Reuse-3)
    ps_linear = 10 ** ((channels_2d + tx_power) / 10.0)
    group_sums = np.zeros((3, total_t))
    for g in range(3):
        group_sums[g, :] = np.sum(ps_linear[g::3, :], axis=0)
        
    all_sectors = np.arange(nbs)
    all_inter_linear = group_sums[all_sectors % 3, :] - ps_linear

    # 2. ICIC (Neighbor detection and power reduction)
    # Target powers for margin calculation: use the strongest overall signal at each t
    # excluding the current sector.
    sorted_indices = np.argsort(channels_2d, axis=0)
    best_overall_idx = sorted_indices[-1, :]
    second_best_overall_idx = sorted_indices[-2, :]
    
    best_pwrs = channels_2d[best_overall_idx, np.arange(total_t)]
    second_best_pwrs = channels_2d[second_best_overall_idx, np.arange(total_t)]
    
    target_pwrs = np.tile(best_pwrs, (nbs, 1))
    target_pwrs[best_overall_idx, np.arange(total_t)] = second_best_pwrs
    
    # Parity: The legacy simulation calculates the margin by dividing the negative dB values
    # e.g., 10 * log10(-80 / -90) which yields values around [-1, +1], always < 7.0.
    # This means ICIC is effectively always ON in the legacy baseline. We replicate this exact math.
    ho_margin_db = 10 * np.log10(channels_2d / (target_pwrs - 1e-15))
    icic_active = ho_margin_db < 7.0
    
    noise_floor = 10**(noise_level / 10.0)
    reduction_factor = 0.1
    inter_noise = np.where(icic_active, 
                           (all_inter_linear * reduction_factor) + noise_floor, 
                           all_inter_linear + noise_floor)
                           
    return 10 * np.log10(ps_linear + 1e-15) - 10 * np.log10(inter_noise + 1e-15)

def vectorized_oracle(channels: np.ndarray, system_params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes SNIR and MCS for all NBS sectors simultaneously.
    """
    input_ndim = channels.ndim
    if input_ndim == 1:
        channels_2d = channels.reshape(-1, 1)
    else:
        channels_2d = channels

    snir = calculate_snir_matrix(channels_2d, system_params)
    
    # MCS Vectorized Mapping
    idx = np.searchsorted(system_params.get("SINRThreshold", SINR_THRESHOLD), snir, side='right') - 1
    idx = np.clip(idx, 0, len(system_params.get("SpectralEff", SPECTRAL_EFF)) - 1)
    mcs = system_params.get("SpectralEff", SPECTRAL_EFF)[idx]
    
    if input_ndim == 1:
        return mcs.flatten(), snir.flatten()
    return mcs, snir

def vectorized_hof(channels: np.ndarray, system_params: dict[str, Any]) -> np.ndarray:
    """
    Computes Handover Failure Probability (Pe) for all NBS sectors simultaneously.
    """
    input_ndim = channels.ndim
    if input_ndim == 1:
        channels_2d = channels.reshape(-1, 1)
    else:
        channels_2d = channels

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

    snir = calculate_snir_matrix(channels_2d, system_params)
    
    # BLER Mapping (Pe)
    idx = np.searchsorted(snr_level, snir, side='right') - 1
    idx = np.clip(idx, 0, len(bler) - 1)
    pe = bler[idx]
    
    if input_ndim == 1:
        return pe.flatten()
    return pe
