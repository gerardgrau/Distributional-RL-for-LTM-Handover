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
    """
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or {}
        
        # Observation Space: 67-dim Markovian vector
        self.observation_space = spaces.Box(low=-5, high=5, shape=(67,), dtype=np.float32)
        self.action_space = spaces.Discrete(NBS)
        
        # Load data paths
        self.files = sorted(glob.glob(os.path.join(ChannelDirectory, "ChannelGainBSUE_User*.mat")))
        self.current_ue_idx = 0
        
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        
        filename = self.files[self.current_ue_idx % len(self.files)]
        mat_data = loadmat(filename)
        raw_channel = mat_data['ChannelBS2UE'] 
        self.current_ue_idx += 1
        
        self.total_time = raw_channel.shape[0]
        self.ch_bs2ue = np.zeros((NBS, self.total_time))
        idx = 0
        for b in range(raw_channel.shape[1]):
            for s in range(raw_channel.shape[2]):
                self.ch_bs2ue[idx, :] = raw_channel[:, b, s]
                idx += 1
        
        # Store real UE positions
        ue_pos_complex = mat_data['UE'][0, 0]['Position'][0]
        self.ue_positions = np.stack([ue_pos_complex.real, ue_pos_complex.imag], axis=1)
        
        # Filters
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
        
        self.ue_speeds = np.full(self.total_time, 10.0) 
        traj_file = "data/SUMO_Network/fcd.pkl"
        if os.path.exists(traj_file):
            try:
                df = pd.read_pickle(traj_file)
                veh_list = sorted(df["vehicle"].unique())
                v_id = veh_list[(self.current_ue_idx-1) % len(veh_list)]
                v_df = df[df["vehicle"] == v_id].sort_values("time")
                speeds = v_df["speed"].values
                self.ue_speeds[:len(speeds)] = speeds
            except Exception as e:
                pass
        
        self.sync = {"N310": 4, "N311": 2, "T310": 50, "out_sync_count": 0, "in_sync_count": 0, "t310_running": False, "t310_counter": np.inf}
        
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
        
        # --- NORMALIZATION ---
        # 1. RSRP: Map [-120, -30] to [-1, 1]
        norm_rsrp = (curr_rsrp + 75) / 45
        # 2. Delta RSRP: Scale by 10 (since changes are small)
        norm_delta = delta_rsrp * 10
        # 3. Tenure: Scale by 1000
        norm_tenure = float(self.serving_tenure) / 1000.0
        # 4. Speed: Scale by 30 m/s
        norm_speed = self.ue_speeds[t] / 30.0
        # 5. MCS/SNIR: MCS [0, 9] -> [0, 1], SNIR [-10, 40] -> [-1, 1]
        norm_mcs = (np.mean(self.mcs_history[-10:]) / 9.0) if self.mcs_history else 0.0
        norm_snir = ((np.mean(self.snir_history[-10:]) - 15) / 25.0) if self.snir_history else -1.0
        
        serving_one_hot = np.zeros(NBS)
        if self.serving_sector != -1:
            serving_one_hot[self.serving_sector] = 1.0
            
        obs = np.concatenate([
            [norm_speed],
            [norm_tenure],
            serving_one_hot,
            norm_rsrp,
            norm_delta,
            [norm_mcs],
            [norm_snir]
        ])
        return obs.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_serving = self.serving_sector
        ho_occurred = (action != prev_serving)
        self.prev_rsrp = self.pl3[:, min(self.t, self.total_time - 1)].copy()
        
        if ho_occurred:
            self.serving_tenure = 0
            self.serving_sector = action
        else:
            self.serving_tenure += 1
            
        reward = 0.0
        done = False
        
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
                reward -= 50.0 
                self.serving_sector = -1
                done = True
                break
            self.t += 1
            
        if ho_occurred:
            reward -= 5.0 
            
        return self._get_obs(), reward, done, False, {}
