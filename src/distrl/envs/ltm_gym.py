import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import os
import glob
import gymnasium as gym
from gymnasium import spaces
from typing import Any

from src.distrl.envs.ltm_env import System, Time, HO, BS, NBS, ReceiverSensitivity, ChannelDirectory, MCSEvaluation

class LTMEnv(gym.Env):
    """
    Gymnasium wrapper for the LTM Handover simulation.
    Acts as the 'glue' between the procedural simulation in `ltm_env.py` and RL agents.
    """
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or {}
        
        # Observation Space: L3 RSRP measurements for all NBS sectors
        self.observation_space = spaces.Box(low=-200, high=0, shape=(NBS,), dtype=np.float32)
        
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
        
        # Reformat to (NBS, T)
        self.total_time = raw_channel.shape[0]
        self.ch_bs2ue = np.zeros((NBS, self.total_time))
        idx = 0
        for b in range(raw_channel.shape[1]):
            for s in range(raw_channel.shape[2]):
                self.ch_bs2ue[idx, :] = raw_channel[:, b, s]
                idx += 1
        
        # Apply L1/L3 Filters (Logic from ltm_env.py)
        M = int(np.ceil(HO["Prep"]["PeriodicityRSRPMeasurement"] / Time["TimeStep"]))
        b = np.ones(HO["Prep"]["AverageRSRPMeasument_NL1"]) / HO["Prep"]["AverageRSRPMeasument_NL1"]
        L1 = lfilter(b, 1, self.ch_bs2ue[:, ::M], axis=1)
        self.pl3 = np.repeat(lfilter(HO["Prep"]["alphaIIRfilter"], [1, -1 + HO["Prep"]["alphaIIRfilter"]], L1, axis=1), M, axis=1)[:, :self.total_time]
        
        # Initial State
        self.t = 0
        self.serving_sector = -1
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
            self.t += 1
            
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return self.pl3[:, min(self.t, self.total_time - 1)].astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_serving = self.serving_sector
        ho_occurred = (action != prev_serving)
        
        reward = 0.0
        done = False
        
        # 1 step in RL = 100ms of simulation (10 samples)
        step_duration = 10 
        for _ in range(step_duration):
            if self.t >= self.total_time - 1:
                done = True
                break
            
            if ho_occurred:
                self.serving_sector = action
            
            mcs, rlf, self.sync = MCSEvaluation(self.serving_sector, self.ch_bs2ue[:, self.t], System, self.sync)
            
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
