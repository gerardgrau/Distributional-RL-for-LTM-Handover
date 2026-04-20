import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import os
import glob
import gymnasium as gym
from gymnasium import spaces
from typing import Any
import pandas as pd

from src.distrl.envs.ltm_env import System, Time, HO, BS, NBS, ReceiverSensitivity, ChannelDirectory, MCSEvaluation

class LTMEnv(gym.Env):
    """
    Gymnasium wrapper for the LTM Handover simulation.
    Acts as the 'glue' between the procedural simulation in `ltm_env.py` and RL agents.
    """
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or {}
        
        # Observation Space: 67-dim Markovian vector
        # [Speed, Tenure, Serving_OneHot(21), RSRP(21), DeltaRSRP(21), MCS_Avg, SNIR_Avg]
        self.observation_space = spaces.Box(low=-200, high=2000, shape=(67,), dtype=np.float32)
        
        # Action Space: Choose which sector to hand over to
        self.action_space = spaces.Discrete(NBS)
        
        # Load data paths
        self.files = sorted(glob.glob(os.path.join(ChannelDirectory, "ChannelGainBSUE_User*.mat")))
        self.current_ue_idx = 0
        
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        
        # Load next UE data
        filename = self.files[self.current_ue_idx % len(self.files)]
        mat_data = loadmat(filename)
        raw_channel = mat_data['ChannelBS2UE'] # (T, BS, Sectors)
        self.current_ue_idx += 1
        
        # Flatten to (NBS, T)
        self.total_time = raw_channel.shape[0]
        self.ch_bs2ue = np.zeros((NBS, self.total_time))
        idx = 0
        for b in range(raw_channel.shape[1]):
            for s in range(raw_channel.shape[2]):
                self.ch_bs2ue[idx, :] = raw_channel[:, b, s]
                idx += 1
        
        # Store real UE positions (X + Yj)
        ue_pos_complex = mat_data['UE'][0, 0]['Position'][0]
        self.ue_positions = np.stack([ue_pos_complex.real, ue_pos_complex.imag], axis=1)
        
        # Apply L1/L3 Filters (Logic from ltm_env.py)
        M = int(np.ceil(HO["Prep"]["PeriodicityRSRPMeasurement"] / Time["TimeStep"]))
        b = np.ones(HO["Prep"]["AverageRSRPMeasument_NL1"]) / HO["Prep"]["AverageRSRPMeasument_NL1"]
        L1 = lfilter(b, 1, self.ch_bs2ue[:, ::M], axis=1)
        self.pl3 = np.repeat(lfilter(HO["Prep"]["alphaIIRfilter"], [1, -1 + HO["Prep"]["alphaIIRfilter"]], L1, axis=1), M, axis=1)[:, :self.total_time]
        
        # Initial State
        self.t = 0
        self.serving_sector = -1
        self.prev_rsrp = self.pl3[:, 0].copy()
        self.serving_tenure = 0
        self.mcs_history = []
        self.snir_history = []
        
        # Load UE Speeds (Mock or from fcd.pkl)
        self.ue_speeds = np.full(self.total_time, 10.0) # Default
        traj_file = "data/SUMO_Network/fcd.pkl"
        if os.path.exists(traj_file):
            try:
                df = pd.read_pickle(traj_file)
                veh_list = sorted(df["vehicle"].unique())
                v_id = veh_list[self.current_ue_idx % len(veh_list)]
                v_df = df[df["vehicle"] == v_id].sort_values("time")
                speeds = v_df["speed"].values
                self.ue_speeds[:len(speeds)] = speeds
            except Exception as e:
                print(f"Warning: Could not load speeds from {traj_file}: {e}")
        
        self.sync = {
            "N310": 4, "N311": 2, "T310": 50, 
            "out_sync_count": 0, "in_sync_count": 0, 
            "t310_running": False, "t310_counter": np.inf
        }
        
        # Initial Cell Selection (Automatic as per procedural logic)
        while self.serving_sector == -1 and self.t < self.total_time:
            pbest = np.max(self.ch_bs2ue[:, self.t])
            best = np.argmax(self.ch_bs2ue[:, self.t])
            if pbest + System["TxPower"] > ReceiverSensitivity:
                self.serving_sector = best
                mcs, rlf, self.sync, snir = MCSEvaluation(self.serving_sector, self.ch_bs2ue[:, self.t], System, self.sync)
                self.mcs_history.append(float(mcs))
                self.snir_history.append(float(snir))
            self.t += 1
            
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        t = min(self.t, self.total_time - 1)
        curr_rsrp = self.pl3[:, t]
        delta_rsrp = curr_rsrp - self.prev_rsrp
        
        # Serving One-Hot
        serving_one_hot = np.zeros(NBS)
        if self.serving_sector != -1:
            serving_one_hot[self.serving_sector] = 1.0
            
        # Stability Averages (Last 10 samples)
        avg_mcs = np.mean(self.mcs_history[-10:]) if self.mcs_history else 0.0
        avg_snir = np.mean(self.snir_history[-10:]) if self.snir_history else -100.0
        
        # Speed
        speed = self.ue_speeds[t]
        
        obs = np.concatenate([
            [speed],
            [float(self.serving_tenure)],
            serving_one_hot,
            curr_rsrp,
            delta_rsrp,
            [avg_mcs],
            [avg_snir]
        ])
        return obs.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_serving = self.serving_sector
        ho_occurred = (action != prev_serving)
        
        # Save RSRP for Delta calculation in next obs
        self.prev_rsrp = self.pl3[:, min(self.t, self.total_time - 1)].copy()
        
        # Update Tenure
        if ho_occurred:
            self.serving_tenure = 0
            self.serving_sector = action
        else:
            self.serving_tenure += 1
            
        reward = 0.0
        done = False
        
        # 1 step in RL = 100ms of simulation (10 samples)
        step_duration = 10 
        for _ in range(step_duration):
            if self.t >= self.total_time - 1:
                done = True
                break
            
            mcs, rlf, self.sync, snir = MCSEvaluation(self.serving_sector, self.ch_bs2ue[:, self.t], System, self.sync)
            
            self.mcs_history.append(float(mcs))
            self.snir_history.append(float(snir))
            
            reward += float(mcs)
            if rlf:
                reward -= 50.0 # RLF Penalty
                self.serving_sector = -1
                done = True
                break
                
            self.t += 1
            
        if ho_occurred:
            reward -= 5.0 # HO Penalty
            
        return self._get_obs(), reward, done, False, {}
