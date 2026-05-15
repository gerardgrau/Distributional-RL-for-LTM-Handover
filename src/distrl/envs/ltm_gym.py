import glob
import os
import re

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.distrl.envs.physics import (
    HO,
    NBS,
    ReceiverSensitivity,
    System,
    Time,
    Time_ContextRelease_11,
    Time_HOdecision_8,
    Time_LLHOCommand_9,
    Time_MeasReportL1_67,
    Time_MeasReportL3_1,
    Time_PingPong,
    Time_RA_10,
    Time_RRCConf3,
    Time_RRCReconf4_5,
    Time_RRCTransfer2,
    CheckHO_Failure,
    MCSEvaluation,
)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class LTMEnv(gym.Env):
    def __init__(self, config=None):
        super(LTMEnv, self).__init__()
        self.config = config or {}
        
        self.output_base = self.config.get('paths', {}).get('precomputed_data_directory', "data/Precomputed")
        self.subdir = self.config.get('paths', {}).get('dataset_subdir', "no_ris")
        self.data_dir = os.path.join(self.output_base, self.subdir)
        
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "User*_precomputed.npz")), key=natural_sort_key)
        ue_count = self.config.get('simulation', {}).get('ue_number', len(all_files))
        if ue_count == 0:
            ue_count = 1000 # Fallback
        self.files = all_files[:ue_count]
        self.current_ue_idx = 0
        
        self.NBS = NBS
        self.Max_iter = int(Time["TotalSimTime"] / Time["TimeStep"])
        
        # Action space: Target BS to HO to (or -1 for stay/none)
        self.action_space = spaces.Discrete(self.NBS)
        
        # Observation Space: 88-dim Markovian vector (Paper Aligned)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(88,), dtype=np.float32)

    def _get_obs_dict(self, t, state_type='NORMAL_STEP', HO_condition=None, PL1_report=None):
        state_map = {'NORMAL_STEP': 0, 'FIND_CELL': 1, 'HO_DECISION': 2}
        obs = {
            "state_type": state_map[state_type],
            "state_str": state_type,
            "t": np.array([t], dtype=np.int32),
            "ChBS2UE_t": self.ChBS2UE[:, t].astype(np.float32),
            "PL1_report": PL1_report if PL1_report is not None else np.zeros(self.NBS, dtype=np.float32),
            "HO_condition": HO_condition.astype(np.int8) if HO_condition is not None else np.zeros(self.NBS, dtype=np.int8)
        }
        return obs

    def _get_rl_obs(self) -> np.ndarray:
        t = min(self.t, self.Max_iter - 1)
        curr_rsrp = self.PL3[:, t]
        
        norm_rsrp = (curr_rsrp + 75) / 45
        norm_tenure = float(self.serving_tenure) / 1000.0
        norm_speed = self.ue_speeds[t] / 30.0

        # Calculate Moving Average over ALL sectors
        p_window = self.config.get('ho_state', {}).get('moving_average_window', 50)
        start_idx = max(0, t - p_window + 1)
        
        if t >= 0:
            avg_mcs = np.mean(self.all_mcs_episode[:, start_idx:t+1], axis=1)
            avg_snir = np.mean(self.all_snir_episode[:, start_idx:t+1], axis=1)
        else:
            avg_mcs = np.zeros(self.NBS)
            avg_snir = np.full(self.NBS, -100.0)

        # Normalization (Ainna-specific)
        norm_mcs = avg_mcs / 9.3 # Normalized by max capacity
        norm_snir = (avg_snir - 15) / 25.0

        serving_one_hot = np.zeros(self.NBS)
        if self.ServingBSSector[t] != -1:
            serving_one_hot[self.ServingBSSector[t]] = 1.0

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        filename = self.files[self.current_ue_idx % len(self.files)]
        self.current_ue_idx += 1
        
        # Fast loading from .npz — `with` ensures the file descriptor is
        # released even though we materialize the arrays we need.
        with np.load(filename) as data:
            self.ChBS2UE = data['ch_bs2ue']
            self.all_mcs_episode = data['all_mcs_episode']
            self.all_snir_episode = data['all_snir_episode']
            self.all_pe_episode = data['all_pe_episode']
            self.ue_positions = data['ue_positions']
            self.PL1 = data['pl1']
            self.PL3 = data['pl3']
        self.Max_iter = self.ChBS2UE.shape[1]
        
        self.ue_speeds = np.full(self.Max_iter, 10.0) 
        
        # Initialize trackers
        self.ServingBSSector = np.full(self.Max_iter, -1, dtype=int)
        self.MCS = np.zeros(self.Max_iter)
        self.HOF = np.zeros(self.Max_iter)
        self.RLF = np.zeros(self.Max_iter)
        self.HO_event = np.zeros(self.Max_iter)
        self.ping_pong = np.zeros(self.Max_iter)
        self.ReservedBSSectors = np.zeros((self.NBS, self.Max_iter))

        self.Sync = {
            "N310": 4, "N311": 2, "T310": 50,
            "out_sync_count": 0, "in_sync_count": 0,
            "t310_running": False, "t310_counter": np.inf
        }

        self.ListBSPrepared = np.zeros(self.NBS, dtype=int)
        self.TimerEntering = np.zeros(self.NBS)
        self.TimerLeaving = np.zeros(self.NBS)
        
        self.t = 0
        self.serving_tenure = 0
        self._sim_gen = self._simulation_generator()
        self._last_obs_dict = next(self._sim_gen)
        
        return self._get_rl_obs(), {"metrics": self._get_obs_dict(0)}

    def step(self, action, high_res_callback=None):
        reward = 0.0
        terminated = False
        truncated = False
        
        if high_res_callback:
            # High-res mode for baseline parity (100ms block = 10 ticks, callback each tick)
            for _ in range(10):
                try:
                    act = high_res_callback(self._last_obs_dict)
                    self._last_obs_dict = self._sim_gen.send(act)
                except StopIteration:
                    terminated = True
                    break
        else:
            # RL Mode (Strict 100ms Action Repeat with RLF Tripwire)
            mcs_accum = 0.0
            alpha_ho = 0.8
            alpha_pp = 0.9
            alpha_hof = 0.1
            has_ho = 0
            has_pp = 0
            has_hof = 0
            
            for _ in range(10):
                try:
                    current_state = self._last_obs_dict['state_type']
                    
                    if current_state == 1: # FIND_CELL (RLF Tripwire)
                        reward -= 100.0 # Massive penalty
                        # Force greedy recovery
                        ChBS2UE = self._last_obs_dict['ChBS2UE_t']
                        Pbest = np.max(ChBS2UE)
                        if Pbest + System["TxPower"] > ReceiverSensitivity:
                            act = np.argmax(ChBS2UE)
                        else:
                            act = -1
                        self._last_obs_dict = self._sim_gen.send(act)
                        
                    elif current_state == 2: # HO_DECISION
                        self._last_obs_dict = self._sim_gen.send(action)
                        action = -1 # Consume action
                        
                    else: # NORMAL_STEP
                        self._last_obs_dict = self._sim_gen.send(-1)

                    # Track metrics
                    t_idx = min(self._last_obs_dict['t'][0], self.Max_iter - 1)
                    mcs_accum += self.MCS[t_idx]
                    has_ho = max(has_ho, self.HO_event[t_idx])
                    has_pp = max(has_pp, self.ping_pong[t_idx])
                    has_hof = max(has_hof, self.HOF[t_idx])
                    
                except StopIteration:
                    terminated = True
                    break
                    
            if not terminated:
                # Ainna Reward
                avg_mcs = mcs_accum / 10.0
                multiplier = 1.0
                if has_ho: multiplier *= alpha_ho
                if has_pp: multiplier *= alpha_pp
                if has_hof: multiplier *= alpha_hof
                n_oos = self.Sync["out_sync_count"]
                reliability_factor = 1.0 / (1.0 + np.exp(2 * (n_oos - 2)))
                
                reward += avg_mcs * multiplier * reliability_factor

        info = {}
        if terminated:
            info["metrics"] = self._build_metrics_dict()
            
        return self._get_rl_obs(), reward, terminated, truncated, info

    def _simulation_generator(self):
        t = 0
        NextBSSector = -1
        former_BS = -1
        former_HO_time = -np.inf

        while t < (self.Max_iter - 10):
            self.t = t
            self.ReservedBSSectors[:, t] = self.ListBSPrepared

            # Mirror legacy line 321: `if ServingBSSector[t] >= 0 and not RLF[t]`.
            # Using `NextBSSector` here would evaluate one extra MCS at the
            # post-HO RACH tick (where ServingBSSector is still -1 from init),
            # consuming an extra RNG call and desyncing all subsequent
            # stochastic interference draws.
            if self.ServingBSSector[t] >= 0 and not self.RLF[t]:
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    NextBSSector = -1

            t += 1
            self.t = t

            # Legacy Line 317: while NextBSSector == -1:
            while NextBSSector == -1:
                action = yield self._get_obs_dict(t, 'FIND_CELL')
                if action != -1:
                    NextBSSector = action
                    self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(NextBSSector, self.ChBS2UE[:, t], System, self.Sync)
                    if self.MCS[t] == 0:
                        NextBSSector = -1
                    else:
                        # Reset
                        self.Sync.update({"out_sync_count": 0, "in_sync_count": 0, "t310_running": False, "t310_counter": np.inf})
                        self.serving_tenure = 0
                t += 1
                self.t = t

            # 5G HO logic
            self.ServingBSSector[t] = NextBSSector
            self.serving_tenure += 1
            self.ReservedBSSectors[:, t] = self.ListBSPrepared
            self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
            
            if self.RLF[t]:
                NextBSSector = -1
                continue

            # 1. L3 Measurement report
            # Legacy does not `break` here on RLF — it sets NextBSSector=-1 and
            # keeps iterating through Tf (writing Res and ServingBSSector each
            # tick). The post-loop `if RLF[t]: continue` then bails out.
            PL3_report = self.PL3[:, t]
            Tf = min(t + int(np.ceil(Time_MeasReportL3_1 / Time["TimeStep"])), self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    NextBSSector = -1
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # 2 & 3. Preparation (RRC transfer + config)
            Tf = min(t + int(np.ceil((Time_RRCTransfer2 + Time_RRCConf3) / Time["TimeStep"])), self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    NextBSSector = -1
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # Conditions
            In_condition = PL3_report > (PL3_report[self.ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
            self.TimerEntering = (self.TimerEntering + Time["TimeStep"]) * In_condition
            self.ListBSPrepared = np.logical_or(self.ListBSPrepared, (self.TimerEntering > HO["Prep"]["PreparationTime"]))

            Out_condition = PL3_report < (PL3_report[self.ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
            self.TimerLeaving = (self.TimerLeaving + Time["TimeStep"]) * Out_condition
            self.ListBSPrepared = np.logical_and(self.ListBSPrepared, ~(self.TimerLeaving > HO["Prep"]["PreparationTime"]))

            if np.sum(self.ListBSPrepared) > HO["Prep"]["MaxNumberPreparedBS"]:
                metric = (10 ** (PL3_report / 10)) * self.ListBSPrepared
                I_sorted = np.argsort(metric)[::-1]
                self.ListBSPrepared[I_sorted[HO["Prep"]["MaxNumberPreparedBS"]:]] = 0

            # 4 & 5. RRC reconfiguration
            Tf = min(t + int(np.ceil(Time_RRCReconf4_5 / Time["TimeStep"])), self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    NextBSSector = -1
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # EXECUTION
            PL1_report = self.PL1[:, t]
            HO_condition = np.logical_and(self.ListBSPrepared, PL1_report > (PL1_report[self.ServingBSSector[t]] + HO["Prep"]["ExecPowerOffset"]))

            Tf = min(t + int(np.ceil((Time_MeasReportL1_67 + Time_HOdecision_8) / Time["TimeStep"])), self.Max_iter-1)
            while t < Tf:
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                if self.RLF[t]:
                    NextBSSector = -1
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            if np.sum(HO_condition) > 0:
                action = yield self._get_obs_dict(t, 'HO_DECISION', HO_condition, PL1_report)
                if action != -1:
                    t0 = t
                    self.HO_event[t0] = 1
                    I = action
                    
                    Tf = min(t + int(np.ceil(Time_LLHOCommand_9 / Time["TimeStep"])), self.Max_iter-1)
                    while t < Tf:
                        self.ReservedBSSectors[:, t] = self.ListBSPrepared
                        self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                        if self.RLF[t]:
                            NextBSSector = -1
                        t += 1
                        self.t = t
                        self.serving_tenure += 1
                        self.ServingBSSector[t] = self.ServingBSSector[t - 1]
                    if self.RLF[t]: NextBSSector = -1; continue

                    t = min(t + int(np.ceil(Time_RA_10 / Time["TimeStep"])), self.Max_iter-1)
                    self.t = t
                    self.serving_tenure += 1
                    self.HOF[t] = CheckHO_Failure(I, self.ChBS2UE[:, t], System)

                    if self.HOF[t] == 0:
                        NextBSSector = I
                        t = min(t + int(np.ceil(Time_ContextRelease_11 / Time["TimeStep"])), self.Max_iter-1)
                        self.t = t
                        self.serving_tenure += 1
                        
                        # SYNC RESET ON HO
                        self.Sync.update({"out_sync_count": 0, "in_sync_count": 0, "t310_running": False, "t310_counter": np.inf})

                        self.ReservedBSSectors[:, t0:t + 1] = np.tile(self.ListBSPrepared.reshape(-1, 1), (1, t - t0 + 1))
                        self.ListBSPrepared[NextBSSector] = 0
                        self.serving_tenure = 0
                    else:
                        NextBSSector = -1

                    self.ping_pong[t0] = (NextBSSector == former_BS) and ((t0 - former_HO_time) < int(np.floor(Time_PingPong / Time["TimeStep"])))
                    former_BS = self.ServingBSSector[t0]
                    former_HO_time = t0
            else:
                # NORMAL STEP — mirror legacy fall-through.
                # Legacy doesn't yield here; it just lets the outer while loop
                # re-enter at the same t, which writes ReservedBSSectors[:, t]
                # and runs MCSEvaluation again as the new iter's first eval.
                # We yield for parity with the RL action protocol (the agent
                # receives an observation each cycle), but we do NOT advance t
                # nor pre-compute MCS — those happen at the top of the next
                # outer iter, exactly like legacy.
                action = yield self._get_obs_dict(t, 'NORMAL_STEP')

    def _build_metrics_dict(self):
        return {
            "mcs": self.MCS,
            "rlf": self.RLF,
            "ho": self.HO_event,
            "hof": self.HOF,
            "pp": self.ping_pong,
            "serving": self.ServingBSSector,
            "pl3": self.PL3,
            "reserved": self.ReservedBSSectors
        }
