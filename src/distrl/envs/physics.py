import hashlib

import numpy as np

# ============================================================
# SYSTEM CONFIGURATIONS
# ============================================================
ReceiverSensitivity = -95

System = {
    "TxPower": 25,  # dBm (Table I)
    "NoiseLevel": -174,  # dBm — raw linear floor (matches legacy_simulation)
    "SINRThreshold": np.array([
        -np.inf, -6.5, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.5, 10.5,
        12.5, 14.5, 16.5, 19.0, 21.5, 24.0
    ]),
    "SpectralEff": np.array([
        0, 0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41,
        2.73, 3.32, 3.90, 4.52, 5.12, 5.55
    ])
}

Time = {
    "TotalSimTime": 300,
    "TimeStep": 10e-3
}

BS = {"Number": 7}
NBS = BS["Number"] * 3

HO = {
    "Prep": {
        "PeriodicityRSRPMeasurement": 20e-3,
        "AverageRSRPMeasument_NL1": 10,
        "kL3": 8,
        "alphaIIRfilter": 2 ** (-8 / 4),
        "PreparationPowerOffset": -3,
        "PreparationTime": 40e-3,
        "ExecPowerOffset": 3.0,
        "MaxNumberPreparedBS": 5
    }
}

# Time variables
Time_PingPong = 1 
Time_InitialSync_0 = 0.0115
Time_MeasReportL3_1 = 0.01
Time_RRCTransfer2 = 0.01
Time_RRCConf3 = 0.022
Time_RRCReconf4_5 = 0.02
Time_MeasReportL1_67 = 0.01
Time_HOdecision_8 = 0.01
Time_LLHOCommand_9 = 0.02
Time_RA_10 = 10e-3
Time_ContextRelease_11 = 20e-3

# ============================================================
# PHYSICS HASH
# ============================================================
def physics_hash() -> str:
    """Short hash of the physics constants that drive the precomputed cache.

    Precomputed `.npz` files are stamped with this value; `LTMEnv.reset()`
    cross-checks it on load and refuses to run if it doesn't match — the
    cache must be regenerated whenever TxPower / NoiseLevel / SINR table
    changes, otherwise the agent's observations and the runtime physics
    diverge silently.
    """
    h = hashlib.sha256()
    h.update(str(System["TxPower"]).encode())
    h.update(str(System["NoiseLevel"]).encode())
    h.update(System["SINRThreshold"].tobytes())
    h.update(System["SpectralEff"].tobytes())
    return h.hexdigest()[:16]


# ============================================================
# PHYSICS FUNCTIONS
# ============================================================
def MCSEvaluation(serving_sector, channels, System, Sync):
    RLF = 0
    serving_channel = channels[serving_sector]
    reuse_factor = 3
    all_sectors = np.arange(len(channels)) 
    target_mask = (all_sectors % reuse_factor) == (serving_sector % reuse_factor)
    Ps = 10 ** ((serving_channel + System["TxPower"]) / 10)
    relevant_channels = channels[target_mask]
    Inter = (relevant_channels + System["TxPower"]) / 10.0

    AllInter = np.sum(10**Inter) - Ps
    M = 3
    noise_linear = 10**(System["NoiseLevel"] / 10.0)

    if np.random.rand() < (1 / M):
        Inter_Noise = M * AllInter + noise_linear
    else:
        Inter_Noise = M * AllInter * 10**(-1.5) + noise_linear
    
    SNIR = 10 * np.log10(Ps) - 10 * np.log10(Inter_Noise) 
    idx = np.where(System["SINRThreshold"] <= SNIR)[0]
    MCS = System["SpectralEff"][idx[-1]] if len(idx) > 0 else 0

    if SNIR <= 0:
        Sync["out_sync_count"] += 1
        Sync["in_sync_count"] = 0
        if Sync["out_sync_count"] >= Sync["N310"] and not Sync["t310_running"]:
            Sync["t310_running"] = True
            Sync["t310_counter"] = Sync["T310"]

    if SNIR > 2:
        Sync["in_sync_count"] += 1
        Sync["out_sync_count"] = 0
        if Sync["in_sync_count"] >= Sync["N311"]:
            Sync["t310_running"] = False

    if Sync["t310_running"]:
        Sync["t310_counter"] -= 1
        RLF = (Sync["t310_counter"] == 0)

    return MCS, RLF, Sync

def CheckHO_Failure(serving_sector, channels, System):
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

    serving_channel = channels[serving_sector]
    reuse_factor = 3
    all_sectors = np.arange(len(channels))
    target_mask = (all_sectors % reuse_factor) == (serving_sector % reuse_factor)
    Ps = 10 ** ((serving_channel + System["TxPower"]) / 10)

    relevant_channels = channels[target_mask]
    Inter = (relevant_channels + System["TxPower"]) / 10.0
    AllInter = np.sum(10**Inter) - Ps
    M = 3
    noise_linear = 10**(System["NoiseLevel"] / 10.0)

    if np.random.rand() < (1 / M):
        Inter_Noise = M * AllInter + noise_linear
    else:
        Inter_Noise = M * AllInter * 10**(-1.5) + noise_linear

    SNIR = 10 * np.log10(Ps) - 10 * np.log10(Inter_Noise)
    idx = np.where(SNR_level <= SNIR)[0]
    Pe = BLER[idx[-1]]

    return np.random.rand() < Pe

def PerformanceEvaluation(MCS, ServingCell_channel, ping_pong, ReservedBSSectors, HOF, HO_event, RLF, Time):
    Minutes = MCS.shape[0] * Time["TimeStep"] / 60
    Performance = {
        "Capacity": MCS,
        "RL_problems": np.sum(RLF) / Minutes,
        "Number_HO": np.sum(HO_event) / Minutes,
        "Number_ping_pongs": np.sum(ping_pong) / Minutes,
        "Reliability": 100 - np.mean(MCS == 0) * 100,
        "Number_cell_preparations": np.sum((ReservedBSSectors[:, 1:] - ReservedBSSectors[:, :-1]) >= 0) / Minutes,
        "Resource_reservation": np.sum(ReservedBSSectors) / ReservedBSSectors.size * 100,
        "HOF": np.sum(HOF) / Minutes
    }
    return Performance

def calculate_snir_matrix(channels_2d: np.ndarray, system_params: dict) -> np.ndarray:
    nbs, total_t = channels_2d.shape
    tx_power = system_params.get("TxPower", 25)
    noise_level = system_params.get("NoiseLevel", -174)
    
    ps_linear = 10 ** ((channels_2d + tx_power) / 10.0)
    group_sums = np.zeros((3, total_t))
    for g in range(3):
        group_sums[g, :] = np.sum(ps_linear[g::3, :], axis=0)
        
    all_sectors = np.arange(nbs)
    all_inter_linear = group_sums[all_sectors % 3, :] - ps_linear

    M = 3
    noise_floor = 10**(noise_level / 10.0)
    
    rand_mask = np.random.rand(nbs, total_t) < (1.0 / M)
    high_inter_noise = (M * all_inter_linear) + noise_floor
    low_inter_noise = (M * all_inter_linear * (10**(-1.5))) + noise_floor
    
    inter_noise = np.where(rand_mask, high_inter_noise, low_inter_noise)
                           
    return 10 * np.log10(ps_linear + 1e-15) - 10 * np.log10(inter_noise + 1e-15)

def vectorized_oracle(channels: np.ndarray, system_params: dict) -> tuple:
    input_ndim = channels.ndim
    if input_ndim == 1:
        channels_2d = channels.reshape(-1, 1)
    else:
        channels_2d = channels

    snir = calculate_snir_matrix(channels_2d, system_params)
    
    idx = np.searchsorted(system_params.get("SINRThreshold"), snir, side='right') - 1
    idx = np.clip(idx, 0, len(system_params.get("SpectralEff")) - 1)
    mcs = system_params.get("SpectralEff")[idx]
    
    if input_ndim == 1:
        return mcs.flatten(), snir.flatten()
    return mcs, snir

def vectorized_hof(channels: np.ndarray, system_params: dict) -> np.ndarray:
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
    
    idx = np.searchsorted(snr_level, snir, side='right') - 1
    idx = np.clip(idx, 0, len(bler) - 1)
    pe = bler[idx]
    
    rand_vals = np.random.rand(channels_2d.shape[0], channels_2d.shape[1])
    hof = (rand_vals < pe).astype(np.int8)
    
    if input_ndim == 1:
        return hof.flatten()
    return hof