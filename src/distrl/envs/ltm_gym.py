import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import os
import glob
import gymnasium as gym
from gymnasium import spaces
from typing import Any
import pandas as pd

from src.distrl.envs.ltm_env import (
    System, Time, HO, BS, NBS, ReceiverSensitivity, 
    ChannelDirectory, VectorizedOracle, VectorizedHOF
)

# Global Cache to store pre-calculated trajectory data (Performance Optimization)
# Format: {ue_filename: {all_mcs: ..., all_snir: ..., pl3: ..., ue_positions: ..., total_time: ..., ch_bs2ue: ...}}
_GLOBAL_UE_CACHE = {}

class LTMEnv(gym.Env):
    """
    Gymnasium wrapper for the LTM Handover simulation.
    """
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or {}

        # Observation Space: 88-dim Markovian vector (Paper Aligned)
        # [Speed(1), Tenure(1), OneHot(21), RSRP(21), MCS_All(21), SNIR_All(21), X(1), Y(1)]
        self.observation_space = spaces.Box(low=-5, high=5, shape=(88,), dtype=np.float32)
        self.action_space = spaces.Discrete(NBS)

        
        # Load data paths
        all_files = sorted(glob.glob(os.path.join(ChannelDirectory, "ChannelGainBSUE_User*.mat")))
        ue_count = self.config.get('simulation', {}).get('ue_number', len(all_files))
        self.files = all_files[:ue_count]
        self.current_ue_idx = 0


    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        
        filename = self.files[self.current_ue_idx % len(self.files)]
        self.current_ue_idx += 1
        
        if filename in _GLOBAL_UE_CACHE:
            # --- PERFORMANCE OPTIMIZATION: Cache Hit (Zero-Cost Reset) ---
            cache_data = _GLOBAL_UE_CACHE[filename]
            self.total_time = cache_data['total_time']
            self.ch_bs2ue = cache_data['ch_bs2ue']
            self.all_mcs_episode = cache_data['all_mcs_episode']
            self.all_snir_episode = cache_data['all_snir_episode']
            self.all_pe_episode = cache_data['all_pe_episode']
            self.ue_positions = cache_data['ue_positions']
            self.pl3 = cache_data['pl3']
        else:
            # --- PERFORMANCE OPTIMIZATION: Cache Miss (Calculate and Store) ---
            mat_data = loadmat(filename)
            raw_channel = mat_data['ChannelBS2UE'] 
            
            self.total_time = raw_channel.shape[0]
            self.ch_bs2ue = np.zeros((NBS, self.total_time))
            idx = 0
            for b in range(raw_channel.shape[1]):
                for s in range(raw_channel.shape[2]):
                    self.ch_bs2ue[idx, :] = raw_channel[:, b, s]
                    idx += 1
            
            # Pre-calculate MCS and SNIR for ALL sectors across the entire episode.
            self.all_mcs_episode, self.all_snir_episode = VectorizedOracle(self.ch_bs2ue, System)
            
            # --- PERFORMANCE OPTIMIZATION: Vectorized HOF ---
            from src.distrl.envs.ltm_env import VectorizedHOF
            self.all_pe_episode = VectorizedHOF(self.ch_bs2ue, System)
            
            # Store real UE positions
            ue_pos_complex = mat_data['UE'][0, 0]['Position'][0]
            self.ue_positions = np.stack([ue_pos_complex.real, ue_pos_complex.imag], axis=1)
            
            # Filters
            M = int(np.ceil(HO["Prep"]["PeriodicityRSRPMeasurement"] / Time["TimeStep"]))
            b = np.ones(HO["Prep"]["AverageRSRPMeasument_NL1"]) / HO["Prep"]["AverageRSRPMeasument_NL1"]
            L1 = lfilter(b, 1, self.ch_bs2ue[:, ::M], axis=1)
            self.pl3 = np.repeat(lfilter(HO["Prep"]["alphaIIRfilter"], [1, -1 + HO["Prep"]["alphaIIRfilter"]], L1, axis=1), M, axis=1)[:, :self.total_time]

            # Cache the data for next time
            _GLOBAL_UE_CACHE[filename] = {
                'total_time': self.total_time,
                'ch_bs2ue': self.ch_bs2ue,
                'all_mcs_episode': self.all_mcs_episode,
                'all_snir_episode': self.all_snir_episode,
                'all_pe_episode': self.all_pe_episode,
                'ue_positions': self.ue_positions,
                'pl3': self.pl3
            }

        # Initial State
        self.t = 0
        self.serving_sector = -1
        self.prev_prev_serving_sector = -1 # For Ping-Pong detection
        self.last_ho_time = -100 # For Ping-Pong detection
        self.prev_rsrp = self.pl3[:, 0].copy()
        self.serving_tenure = 0
        
        # Load Window Size (p) from config
        self.p_window = self.config.get('ho_state', {}).get('moving_average_window', 50)
        
        # Global History for all 21 sectors (Circular Buffer - Performance Optimization)
        self.mcs_history = np.zeros((self.p_window, NBS))
        self.snir_history = np.zeros((self.p_window, NBS))
        self.mcs_running_sum = np.zeros(NBS)
        self.snir_running_sum = np.full(NBS, -100.0 * self.p_window) # Initial SNIR is -100
        self.history_count = 0
        
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
                # Check if MCS > 0 in the pre-calculated matrix
                if self.all_mcs_episode[best, self.t] > 0:
                    self.serving_sector = best
                    # Initial state metrics for ALL sectors (Maintain running sum)
                    idx = self.history_count % self.p_window
                    self.mcs_running_sum += self.all_mcs_episode[:, self.t] - self.mcs_history[idx]
                    self.snir_running_sum += self.all_snir_episode[:, self.t] - self.snir_history[idx]
                    
                    self.mcs_history[idx] = self.all_mcs_episode[:, self.t]
                    self.snir_history[idx] = self.all_snir_episode[:, self.t]
                    self.history_count += 1
            self.t += 1
            
        return self._get_obs(), {}


    def _get_obs(self) -> np.ndarray:
        t = min(self.t, self.total_time - 1)
        curr_rsrp = self.pl3[:, t]
        
        # --- NORMALIZATION ---
        # 1. RSRP: Map [-120, -30] to [-1, 1]
        norm_rsrp = (curr_rsrp + 75) / 45
        
        # 2. Tenure: Scale by 1000
        norm_tenure = float(self.serving_tenure) / 1000.0
        
        # 3. Speed: Scale by 30 m/s
        norm_speed = self.ue_speeds[t] / 30.0

        # Calculate Moving Average over ALL sectors (window size p) using O(1) running sums
        if self.history_count > 0:
            count = min(self.history_count, self.p_window)
            avg_mcs = self.mcs_running_sum / count
            avg_snir = self.snir_running_sum / count
        else:
            avg_mcs = np.zeros(NBS)
            avg_snir = np.full(NBS, -100.0)

        # 4. MCS/SNIR: MCS [0, 9] -> [0, 1], SNIR [-10, 40] -> [-1, 1]
        norm_mcs = avg_mcs / 9.0
        norm_snir = (avg_snir - 15) / 25.0

        serving_one_hot = np.zeros(NBS)
        if self.serving_sector != -1:
            serving_one_hot[self.serving_sector] = 1.0

        # 5. Position: Scale by 500m
        norm_x = self.ue_positions[t, 0] / 500.0
        norm_y = self.ue_positions[t, 1] / 500.0
            
        obs = np.concatenate([
            [norm_speed],
            [norm_tenure],
            serving_one_hot,
            norm_rsrp,
            norm_mcs,
            norm_snir,
            [norm_x],
            [norm_y]
        ])
        return obs.astype(np.float32)


    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Perform 1 RL step (100ms of simulation).
        """
        if self.serving_sector == -1:
            # Baseline Recovery Strategy: Find strongest cell
            pbest = np.max(self.ch_bs2ue[:, self.t])
            best = np.argmax(self.ch_bs2ue[:, self.t])
            
            # If the strongest cell meets sensitivity, attempt connection
            if pbest + System["TxPower"] > ReceiverSensitivity:
                # Use pre-calculated MCS
                if self.all_mcs_episode[best, self.t] > 0:
                    self.serving_sector = best
                    self.sync["out_sync_count"] = 0
                    self.sync["in_sync_count"] = 0
                    self.sync["t310_running"] = False
                    self.sync["t310_counter"] = np.inf
                    self.serving_tenure = 0
                    
            self.HO_ind = 0.0
            self.PP_ind = 0.0
            self.HOF_ind = 0.0
        else:
            self._handle_handover_logic(action)

        r_thr, _ = self._simulate_radio_samples()
        reward = self._calculate_ltm_ho_reward(r_thr)
        done = self.t >= self.total_time - 1

        return self._get_obs(), float(reward), done, False, {}

    def _update_sync(self, snir: float) -> bool:
        """
        Maintains the 3GPP synchronization state machine (N310/N311/T310).
        Ensures 100% logic parity with the original simulation.
        """
        sync = self.sync
        if snir <= 0:
            sync["out_sync_count"] += 1
            sync["in_sync_count"] = 0
            if sync["out_sync_count"] >= sync["N310"] and not sync["t310_running"]:
                sync["t310_running"] = True
                sync["t310_counter"] = sync["T310"]

        elif snir > 2:
            sync["in_sync_count"] += 1
            sync["out_sync_count"] = 0
            if sync["in_sync_count"] >= sync["N311"]:
                sync["t310_running"] = False

        if sync["t310_running"]:
            sync["t310_counter"] -= 1
            if sync["t310_counter"] == 0:
                return True
        
        return False

    def _handle_handover_logic(self, action: int) -> None:
        """
        Detects HO, HOF, and Ping-Pong events. Updates serving sector and tenure.
        """
        prev_serving = self.serving_sector
        ho_occurred = (prev_serving != -1) and (action != prev_serving)
        
        self.prev_rsrp = self.pl3[:, min(self.t, self.total_time - 1)].copy()
        
        self.HO_ind = 0.0
        self.PP_ind = 0.0
        self.HOF_ind = 0.0
        
        if ho_occurred:
            # Check for Handover Failure (HOF) using pre-calculated Pe matrix
            pe = self.all_pe_episode[action, self.t]
            if np.random.rand() < pe:
                self.HOF_ind = 1.0
                self.serving_sector = -1 # Disconnected
                return
            
            # Check for Ping-Pong (PP)
            is_pp = (action == self.prev_prev_serving_sector) and (self.t - self.last_ho_time < 1.0 / Time["TimeStep"])
            if is_pp:
                self.PP_ind = 1.0
            
            self.HO_ind = 1.0
            self.prev_prev_serving_sector = prev_serving
            self.last_ho_time = self.t
            self.serving_sector = action
            self.serving_tenure = 0
        else:
            self.serving_tenure += 1

    def _simulate_radio_samples(self, step_duration: int = 10) -> tuple[float, bool]:
        """
        Simulates background radio samples using pre-calculated matrices.
        """
        mcs_sum = 0.0
        rlf_happened = False
        for _ in range(step_duration):
            if self.t >= self.total_time - 1:
                break
            
            # 1. Update Oracle History (Zero cost - array slice)
            idx = self.history_count % self.p_window
            self.mcs_running_sum += self.all_mcs_episode[:, self.t] - self.mcs_history[idx]
            self.snir_running_sum += self.all_snir_episode[:, self.t] - self.snir_history[idx]
            
            self.mcs_history[idx] = self.all_mcs_episode[:, self.t]
            self.snir_history[idx] = self.all_snir_episode[:, self.t]
            self.history_count += 1
            
            # 2. Evaluate Serving Cell
            if self.serving_sector != -1:
                m = self.all_mcs_episode[self.serving_sector, self.t]
                s = self.all_snir_episode[self.serving_sector, self.t]
                rlf = self._update_sync(s)
                mcs_sum += float(m)
                if rlf:
                    self.serving_sector = -1
                    rlf_happened = True
            else:
                mcs_sum += 0.0
            
            self.t += 1
            
        return mcs_sum / step_duration, rlf_happened

    def _calculate_ltm_ho_reward(self, r_thr: float) -> float:
        """
        Applies the LTM HO Multiplicative Reward Formula.
        """
        rew_cfg = self.config.get('ho_reward', {})

        ho_factor = rew_cfg.get('alpha_ho', 0.8) ** self.HO_ind
        pp_factor = rew_cfg.get('alpha_pp', 0.9) ** self.PP_ind
        hof_factor = rew_cfg.get('alpha_hof', 0.1) ** self.HOF_ind        
        # Reliability Penalty (Reverse Sigmoid)
        N_oos = self.sync["out_sync_count"]
        reliability_factor = 1.0 / (1 + np.exp(2 * (N_oos - 2)))
        
        return r_thr * (ho_factor * pp_factor * hof_factor) * reliability_factor
     
        return r_thr * (ho_factor * pp_factor * hof_factor) * reliability_factor
