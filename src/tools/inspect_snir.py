import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import os
import glob
from typing import Any

ChannelDirectory = "data/ChannelGains"

def get_realistic_interference(ch_vector, serving_idx, all_inter_linear, system_params):
    noise_floor = 10**(system_params["NoiseLevel"] / 10.0)
    if serving_idx == -1:
        return all_inter_linear + noise_floor, False
    neighbor_indices = np.delete(np.arange(len(ch_vector)), serving_idx)
    target_idx = neighbor_indices[np.argmax(ch_vector[neighbor_indices])]
    serving_pwr = ch_vector[serving_idx]
    target_pwr = ch_vector[target_idx]
    ho_margin_db = 10 * np.log10(serving_pwr / (target_pwr + 1e-15))
    icic_active = ho_margin_db < 7.0  
    if icic_active:
        reduction_factor = 0.1
        inter_noise = (all_inter_linear * reduction_factor) + noise_floor
    else:
        inter_noise = all_inter_linear + noise_floor
    return inter_noise, icic_active

def MCSEvaluation(serving_sector, channels, System, Sync, t_debug=0):
    RLF = 0
    serving_channel = channels[serving_sector]
    reuse_factor = 3
    all_sectors = np.arange(len(channels)) 
    target_mask = (all_sectors % reuse_factor) == (serving_sector % reuse_factor)
    Ps = 10 ** ((serving_channel + System["TxPower"]) / 10)
    relevant_channels = channels[target_mask]
    Inter = (relevant_channels + System["TxPower"]) / 10.0
    AllInter = np.sum(10**Inter) - Ps
    Inter_Noise, icic_on = get_realistic_interference(channels, serving_sector, AllInter, System)
    SNIR = 10 * np.log10(Ps) - 10 * np.log10(Inter_Noise) 
    
    ps_all = 10 ** ((channels + System["TxPower"]) / 10.0)
    group_sum = np.sum(ps_all[all_sectors % 3 == serving_sector % 3])
    
    if t_debug < 50:
        print(f"DEBUG_SNIR: t={t_debug} s={serving_sector} snir={SNIR:.2f} ps={Ps:.2e} group_sum={group_sum:.2e} inter={Inter_Noise:.2e} icic={'ON' if icic_on else 'OFF'}")
    
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

def run_debug_sim():
    System = {"TxPower": 25, "NoiseLevel": -174, 
              "SINRThreshold": np.array([-np.inf, -6.5, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.5, 10.5, 12.5, 14.5, 16.5, 19.0, 21.5, 24.0]),
              "SpectralEff": np.array([0, 0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32, 3.90, 4.52, 5.12, 5.55])}
    Time = {"TotalSimTime": 60, "TimeStep": 10e-3}
    HO = {"Prep": {"PeriodicityRSRPMeasurement": 20e-3, "AverageRSRPMeasument_NL1": 10, "kL3": 8, "alphaIIRfilter": 2 ** (-8 / 4), "PreparationPowerOffset": -3, "PreparationTime": 40e-3, "ExecPowerOffset": 3, "MaxNumberPreparedBS": 5}}
    
    filename = "data/ChannelGains/ChannelGainBSUE_User1.mat"
    mat_data = loadmat(filename)
    Channel = mat_data['ChannelBS2UE_noRIS']
    NBS = 21
    ChBS2UE = np.zeros((NBS, Channel.shape[0]))
    idx = 0
    for b in range(Channel.shape[1]):
        for s in range(Channel.shape[2]):
            ChBS2UE[idx, :] = Channel[:, b, s]
            idx += 1
            
    M = 2
    b_filt = np.ones(10) / 10.0
    L1 = lfilter(b_filt, 1, ChBS2UE[:, ::M], axis=1)
    PL1 = np.repeat(L1, M, axis=1)[:, :ChBS2UE.shape[1]]
    PL3 = np.repeat(lfilter(HO["Prep"]["alphaIIRfilter"], [1, -1 + HO["Prep"]["alphaIIRfilter"]], L1, axis=1), M, axis=1)[:, :ChBS2UE.shape[1]]

    Max_iter = ChBS2UE.shape[1]
    ServingBSSector = np.zeros(Max_iter, dtype=int)
    MCS = np.zeros(Max_iter)
    RLF = np.zeros(Max_iter)
    Sync = {"N310": 4, "N311": 2, "T310": 50, "out_sync_count": 0, "in_sync_count": 0, "t310_running": False, "t310_counter": np.inf}
    ListBSPrepared = np.zeros(NBS, dtype=int)
    TimerEntering = np.zeros(NBS)
    TimerLeaving = np.zeros(NBS)
    
    t = 0
    NextBSSector = -1
    while t < (Max_iter - 10):
        # ... top evaluation skipped for t=0 ...
        
        while NextBSSector == -1:
            Pbest = np.max(ChBS2UE[:, t]); Best = np.argmax(ChBS2UE[:, t])
            if Pbest + System["TxPower"] > -95:
                NextBSSector = Best
                MCS[t], RLF[t], Sync = MCSEvaluation(NextBSSector, ChBS2UE[:, t], System, Sync, t)
                if MCS[t] == 0: NextBSSector = -1
            t += 1
        ServingBSSector[t] = NextBSSector
        MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync, t)
        if RLF[t]: NextBSSector = -1; continue
        
        # 5G/LTM Procedural Delays (Matches original exactly)
        # Skip 1 step for L3 report
        t += 1; ServingBSSector[t] = ServingBSSector[t-1]
        MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync, t)
        # Skip 4 steps for RRC Transfer/Conf
        for _ in range(4):
            if t < Max_iter-1:
                t += 1; ServingBSSector[t] = ServingBSSector[t-1]
                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync, t)
        
        # Preparation Logic
        PL3_report = PL3[:, t]
        in_cond = PL3_report > (PL3_report[ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
        TimerEntering = (TimerEntering + 0.01) * in_cond
        ListBSPrepared = np.logical_or(ListBSPrepared, (TimerEntering > HO["Prep"]["PreparationTime"]))
        # Skip 2 steps for RRC Reconf
        for _ in range(2):
            if t < Max_iter-1:
                t += 1; ServingBSSector[t] = ServingBSSector[t-1]
                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync, t)
        
        # Execution logic
        PL1_report = PL1[:, t]
        HO_condition = np.logical_and(ListBSPrepared, PL1_report > (PL1_report[ServingBSSector[t]] + HO["Prep"]["ExecPowerOffset"]))
        # Skip 2 steps for L1 report/decision
        for _ in range(2):
            if t < Max_iter-1:
                t += 1; ServingBSSector[t] = ServingBSSector[t-1]
                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync, t)
        
        if np.any(HO_condition):
            # ... HO ...
            pass
            
    print(f"Final mean MCS: {np.mean(MCS)}")
    print(f"Reliability: {100 - np.mean(MCS == 0) * 100}%")

if __name__ == "__main__":
    run_debug_sim()
