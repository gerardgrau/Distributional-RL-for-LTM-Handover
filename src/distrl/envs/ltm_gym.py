import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import os
import glob
import gymnasium as gym
from gymnasium import spaces
from typing import Any
import pandas as pd

from src.distrl.envs.physics import (
    System, Time, HO, NBS, ReceiverSensitivity, 
    vectorized_oracle, vectorized_hof
)

ChannelDirectory = "data/ChannelGains"
PrecomputedDirectory = "data/Precomputed"

class LTMEnv(gym.Env):
    """
    Gymnasium wrapper for the LTM Handover simulation.
    """
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or {}
        
        # --- PHYSICS CONFIGURATION (Unified and Configurable) ---
        self.sys_cfg = System.copy()
        if 'system' in self.config:
            self.sys_cfg["TxPower"] = self.config['system'].get('tx_power', self.sys_cfg["TxPower"])
            self.sys_cfg["NoiseLevel"] = self.config['system'].get('noise_level', self.sys_cfg["NoiseLevel"])
            self.sys_cfg["Bandwidth"] = self.config['system'].get('bandwidth', 200e6)
        
        self.time_cfg = Time.copy()
        if 'simulation' in self.config:
            self.time_cfg["TotalSimTime"] = self.config['simulation'].get('total_sim_time', self.time_cfg["TotalSimTime"])
            self.time_cfg["TimeStep"] = self.config['simulation'].get('time_step', self.time_cfg["TimeStep"])
            
        self.ho_cfg = HO.copy()
        if 'ho_prep' in self.config:
            mapping = {
                'preparation_power_offset': 'PreparationPowerOffset',
                'preparation_time': 'PreparationTime',
                'exec_power_offset': 'ExecPowerOffset',
                'max_number_prepared_bs': 'MaxNumberPreparedBS'
            }
            for k, v in mapping.items():
                if k in self.config['ho_prep']:
                    self.ho_cfg['Prep'][v] = self.config['ho_prep'][k]

        self.receiver_sensitivity = self.config.get('system', {}).get('receiver_sensitivity', ReceiverSensitivity)
        self.data_dir = self.config.get('paths', {}).get('channel_data_directory', ChannelDirectory)

        # Observation Space: 88-dim Markovian vector (Paper Aligned)
        # [Speed(1), Tenure(1), OneHot(21), RSRP(21), MCS_All(21), SNIR_All(21), X(1), Y(1)]
        self.observation_space = spaces.Box(low=-5, high=5, shape=(88,), dtype=np.float32)
        self.action_space = spaces.Discrete(NBS)

        
        # Load all 1000 data paths (Refactored: No train/test split at Env level)
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "ChannelGainBSUE_User*.mat")))
        ue_count = self.config.get('simulation', {}).get('ue_number', len(all_files))
        self.files = all_files[:ue_count]
        self.current_ue_idx = 0


    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        
        filename = self.files[self.current_ue_idx % len(self.files)]
        self.current_ue_idx += 1
        
        user_id = os.path.basename(filename).split('_')[-1].split('.')[0]
        npz_filename = os.path.join(PrecomputedDirectory, f"{user_id}_precomputed.npz")

        if os.path.exists(npz_filename):
            # --- PERFORMANCE OPTIMIZATION: Instant binary load (Low RAM) ---
            with np.load(npz_filename) as data:
                self.total_time = int(data['total_time'])
                self.ch_bs2ue = data['ch_bs2ue']
                self.all_mcs_episode = data['all_mcs_episode']
                self.all_snir_episode = data['all_snir_episode']
                self.all_pe_episode = data['all_pe_episode']
                self.ue_positions = data['ue_positions']
                self.pl3 = data['pl3']
                # Try to load PL1 if available, otherwise reconstruct
                if 'pl1' in data:
                    self.pl1 = data['pl1']
                else:
                    M = int(np.ceil(self.ho_cfg["Prep"]["PeriodicityRSRPMeasurement"] / self.time_cfg["TimeStep"]))
                    b = np.ones(self.ho_cfg["Prep"]["AverageRSRPMeasument_NL1"]) / self.ho_cfg["Prep"]["AverageRSRPMeasument_NL1"]
                    self.pl1 = np.repeat(lfilter(b, 1, self.ch_bs2ue[:, ::M], axis=1), M, axis=1)[:, :self.total_time]
        else:
            # --- FALLBACK: Calculate on the fly (Legacy/First run) ---
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
            self.all_mcs_episode, self.all_snir_episode = vectorized_oracle(self.ch_bs2ue, self.sys_cfg)
            
            # --- PERFORMANCE OPTIMIZATION: Vectorized HOF ---
            self.all_pe_episode = vectorized_hof(self.ch_bs2ue, self.sys_cfg)
            
            # Store real UE positions
            ue_pos_complex = mat_data['UE'][0, 0]['Position'][0]
            self.ue_positions = np.stack([ue_pos_complex.real, ue_pos_complex.imag], axis=1)
            
            # Filters
            M = int(np.ceil(self.ho_cfg["Prep"]["PeriodicityRSRPMeasurement"] / self.time_cfg["TimeStep"]))
            b = np.ones(self.ho_cfg["Prep"]["AverageRSRPMeasument_NL1"]) / self.ho_cfg["Prep"]["AverageRSRPMeasument_NL1"]
            L1 = lfilter(b, 1, self.ch_bs2ue[:, ::M], axis=1)
            self.pl1 = np.repeat(L1, M, axis=1)[:, :self.total_time]
            self.pl3 = np.repeat(lfilter(self.ho_cfg["Prep"]["alphaIIRfilter"], [1, -1 + self.ho_cfg["Prep"]["alphaIIRfilter"]], L1, axis=1), M, axis=1)[:, :self.total_time]

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
        self.snir_running_sum = np.zeros(NBS)
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
        
        # --- HO DELAY STATE ---
        self.ho_in_progress = False
        self.ho_substep = 0
        self.target_sector = -1
        self.prev_serving_sector = -1

        # Performance Tracking (8 Metrics)
        self.metrics_mcs = np.zeros(self.total_time)
        self.metrics_rlf = np.zeros(self.total_time)
        self.metrics_ho = np.zeros(self.total_time)
        self.metrics_hof = np.zeros(self.total_time)
        self.metrics_pp = np.zeros(self.total_time)
        self.metrics_serving = np.full(self.total_time, -1, dtype=int)

        while self.serving_sector == -1 and self.t < self.total_time:
            pbest = np.max(self.ch_bs2ue[:, self.t])
            best = np.argmax(self.ch_bs2ue[:, self.t])
            if pbest + System["TxPower"] > ReceiverSensitivity:
                # Check if MCS > 0 in the pre-calculated matrix
                if self.all_mcs_episode[best, self.t] > 0:
                    self.serving_sector = best
                    self.metrics_mcs[self.t] = self.all_mcs_episode[best, self.t]
                    self.metrics_serving[self.t] = best
                    # Initial state metrics for ALL sectors (Maintain running sum)
                    self._update_oracle_history(self.all_mcs_episode[:, self.t], self.all_snir_episode[:, self.t])
            self.t += 1
            
        return self._get_obs(), {}

    def _update_oracle_history(self, mcs_values: np.ndarray, snir_values: np.ndarray) -> None:
        idx = self.history_count % self.p_window

        if self.history_count >= self.p_window:
            self.mcs_running_sum -= self.mcs_history[idx]
            self.snir_running_sum -= self.snir_history[idx]

        self.mcs_history[idx] = mcs_values
        self.snir_history[idx] = snir_values
        self.mcs_running_sum += mcs_values
        self.snir_running_sum += snir_values
        self.history_count += 1


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
            # We must still update sync state even when disconnected (Cell Search phase)
            # as per legacy_simulation.py parity.
            for _ in range(10): # 10 samples of 10ms
                if self.t >= self.total_time - 1:
                    break
                
                self._update_oracle_history(self.all_mcs_episode[:, self.t], self.all_snir_episode[:, self.t])
                
                pbest = np.max(self.ch_bs2ue[:, self.t])
                best = np.argmax(self.ch_bs2ue[:, self.t])
                
                s_best = self.all_snir_episode[best, self.t]
                m_best = self.all_mcs_episode[best, self.t]
                
                # Check for RLF even during search
                rlf = self._update_sync(s_best)
                self.metrics_rlf[self.t] = float(rlf)
                self.metrics_serving[self.t] = -1
                self.metrics_mcs[self.t] = 0.0
                
                if pbest + self.sys_cfg["TxPower"] > self.receiver_sensitivity:
                    if m_best > 0:
                        self.serving_sector = best
                        self.sync["out_sync_count"] = 0
                        self.sync["in_sync_count"] = 0
                        self.sync["t310_running"] = False
                        self.sync["t310_counter"] = np.inf
                        self.serving_tenure = 0
                        self.t += 1
                        break
                self.t += 1
                    
            self.HO_ind = 0.0
            self.PP_ind = 0.0
            self.HOF_ind = 0.0
            r_thr = 0.0
        else:
            self._handle_handover_logic(action)
            r_thr, _ = self._simulate_radio_samples()

        reward = self._calculate_ltm_ho_reward(r_thr)
        done = self.t >= self.total_time - 1

        info = {}
        if done:
            info["metrics"] = {
                "mcs": self.metrics_mcs,
                "rlf": self.metrics_rlf,
                "ho": self.metrics_ho,
                "hof": self.metrics_hof,
                "pp": self.metrics_pp,
                "serving": self.metrics_serving,
                "pl3": self.pl3
            }
        
        # Add current L1/L3 for the agent (not in observation to keep it 88-dim)
        t_curr = min(self.t, self.total_time - 1)
        info["rsrp_l1"] = self.pl1[:, t_curr]
        info["rsrp_l3"] = self.pl3[:, t_curr]

        return self._get_obs(), float(reward), done, False, info

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
            # --- HO ATTEMPT ---
            # Every HO attempt must be recorded in metrics_ho
            self.metrics_ho[self.t] = 1.0
            self.HO_ind = 1.0
            
            # Check for Handover Failure (HOF) using pre-calculated Pe matrix
            pe = self.all_pe_episode[action, self.t]
            if np.random.rand() < pe:
                self.HOF_ind = 1.0
                self.metrics_hof[self.t] = 1.0
                self.serving_sector = -1 # Disconnected
                return
            
            # Check for Ping-Pong (PP)
            is_pp = (action == self.prev_prev_serving_sector) and (self.t - self.last_ho_time < 1.0 / self.time_cfg["TimeStep"])
            if is_pp:
                self.PP_ind = 1.0
                self.metrics_pp[self.t] = 1.0
            
            self.prev_prev_serving_sector = prev_serving
            self.last_ho_time = self.t
            
            # --- START HO DELAY PROCEDURE ---
            # 5G/LTM HO is not instant. 
            # Per legacy_simulation.py: 20ms Command (on old cell) + 30ms Interruption.
            self.ho_in_progress = True
            self.ho_substep = 0
            self.prev_serving_sector = prev_serving
            self.target_sector = action
            
            self.serving_tenure = 0
        else:
            self.serving_tenure += 1

    def _simulate_radio_samples(self, step_duration: int = 10) -> tuple[float, bool]:
        """
        Simulates background radio samples using pre-calculated matrices.
        Accounts for HO delays and interruption.
        """
        mcs_sum = 0.0
        rlf_happened = False
        samples_count = 0
        
        for i in range(step_duration):
            if self.t >= self.total_time - 1:
                break
            
            # 1. Update Oracle History (Zero cost - array slice)
            self._update_oracle_history(self.all_mcs_episode[:, self.t], self.all_snir_episode[:, self.t])
            
            # 2. Determine currently active sector for this 10ms sample
            active_sector = self.serving_sector
            if self.ho_in_progress:
                if self.ho_substep < 2: # First 20ms: Still on OLD cell
                    active_sector = self.prev_serving_sector
                elif self.ho_substep < 5: # Next 30ms: Interruption (Outage)
                    active_sector = -1
                else: # 50ms+ : On NEW cell
                    active_sector = self.target_sector
                    if self.ho_substep == 5:
                        self.serving_sector = self.target_sector # Official switch
                
                self.ho_substep += 1
            
            # 3. Evaluate Active Sector
            if active_sector != -1:
                m = self.all_mcs_episode[active_sector, self.t]
                s = self.all_snir_episode[active_sector, self.t]
                rlf = self._update_sync(s)
                mcs_sum += float(m)
                
                # Tracking
                self.metrics_mcs[self.t] = m
                self.metrics_rlf[self.t] = float(rlf)
                self.metrics_serving[self.t] = active_sector

                if rlf:
                    self.serving_sector = -1
                    self.ho_in_progress = False
                    rlf_happened = True
                    active_sector = -1 # Disconnect
            else:
                mcs_sum += 0.0
                self.metrics_serving[self.t] = -1
                # Sync logic is suspended during interruption (matches legacy)
            
            self.t += 1
            samples_count += 1
            
        self.ho_in_progress = False # Reset for next RL step
        avg_mcs = mcs_sum / samples_count if samples_count > 0 else 0.0
        return avg_mcs, rlf_happened

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
