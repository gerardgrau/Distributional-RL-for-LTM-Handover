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
    physics_hash,
)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


# Tick counts derived from the physics time constants. Hoisted to module
# scope so the simulation generator is not re-running int(np.ceil(...))
# on every iteration (called O(ticks) times per episode).
_TICK_MEAS_L3 = int(np.ceil(Time_MeasReportL3_1 / Time["TimeStep"]))
_TICK_RRC_PREP = int(np.ceil(
    (Time_RRCTransfer2 + Time_RRCConf3) / Time["TimeStep"]
))
_TICK_RRC_RECONF = int(np.ceil(Time_RRCReconf4_5 / Time["TimeStep"]))
_TICK_MEAS_L1_HODEC = int(np.ceil(
    (Time_MeasReportL1_67 + Time_HOdecision_8) / Time["TimeStep"]
))
_TICK_LL_HOCMD = int(np.ceil(Time_LLHOCommand_9 / Time["TimeStep"]))
_TICK_RA = int(np.ceil(Time_RA_10 / Time["TimeStep"]))
_TICK_CTX_RELEASE = int(np.ceil(Time_ContextRelease_11 / Time["TimeStep"]))
_TICK_PINGPONG = int(np.floor(Time_PingPong / Time["TimeStep"]))

# state_type string -> int code consumed by env.step. Hoisted out of
# _get_obs_dict so we are not rebuilding a 3-key Python dict on every
# yield (~30k per episode).
_STATE_MAP = {'NORMAL_STEP': 0, 'FIND_CELL': 1, 'HO_DECISION': 2}


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

        # Cache config knobs that are read on every step / obs call.
        self._p_window = int(self.config.get('ho_state', {}).get(
            'moving_average_window', 50,
        ))
        ho_reward_cfg = self.config.get('ho_reward', {})
        self._alpha_ho = float(ho_reward_cfg.get('alpha_ho', 0.8))
        self._alpha_pp = float(ho_reward_cfg.get('alpha_pp', 0.9))
        self._alpha_hof = float(ho_reward_cfg.get('alpha_hof', 0.1))

        # Pre-allocated 88-dim observation buffer. _get_rl_obs fills slots
        # in-place and returns .copy() to the caller. Replaces the prior
        # 7 small allocations + np.concatenate + .astype path.
        self._obs_buf = np.empty(88, dtype=np.float32)

        # Running-sum state for the per-sector moving averages of MCS / SNIR
        # (window = _p_window). Filled by _advance_ma; reset() invalidates
        # it. The previous implementation re-sliced and re-meaned the
        # episode arrays on every obs call — O(window * NBS) per tick.
        self._ma_last_t: int | None = None
        self._mcs_sum = np.zeros(self.NBS)
        self._snir_sum = np.zeros(self.NBS)

    def _get_obs_dict(self, t, state_type='NORMAL_STEP', HO_condition=None, PL1_report=None):
        """Build the per-tick observation dict yielded by the simulation
        generator.

        Lazy: only the fields actually consumed for `state_type` are
        included. `env.step` reads `state_type` and `t` for every yield,
        `ChBS2UE_t` only in FIND_CELL, and `PL1_report` / `HO_condition`
        only in HO_DECISION. `t` is passed through as a Python int — the
        previous np.array([t], dtype=np.int32) allocated 1 array per
        yield and was always unpacked back to an int by callers.

        NORMAL_STEP (the common case) returns a 3-field dict with no
        numpy allocations; FIND_CELL adds the 21-channel view; HO_DECISION
        passes the producer-side arrays straight through. The previous
        eager version allocated 4 numpy arrays per yield regardless of
        state_type.
        """
        obs = {
            "state_type": _STATE_MAP[state_type],
            "state_str": state_type,
            "t": t,
        }
        if state_type == 'FIND_CELL':
            # View into the precomputed channel matrix (no copy/cast — the
            # downstream max/argmax operate fine in the source dtype).
            obs["ChBS2UE_t"] = self.ChBS2UE[:, t]
        elif state_type == 'HO_DECISION':
            obs["PL1_report"] = PL1_report
            obs["HO_condition"] = HO_condition
        return obs

    def _advance_ma(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Update the O(1) MCS/SNIR running sums to time `t` and return the
        per-sector window averages.

        Window definition matches the original `np.mean(arr[:, start:t+1])`
        path: `start = max(0, t - p_window + 1)`. The legacy code re-sliced
        and re-summed both episode arrays on every call; this version
        maintains running totals and only touches the columns that enter
        or leave the window between calls.
        """
        p_window = self._p_window
        new_start = max(0, t - p_window + 1)
        # `n` is the actual window length (smaller than p_window while the
        # window still abuts t=0). Always >= 1 because we clamp t to the
        # valid range upstream.
        n = (t + 1) - new_start

        # First call after reset, or t walked backward (only possible if
        # reset() didn't clear the cache): rebuild sums from scratch.
        if self._ma_last_t is None or t < self._ma_last_t:
            self._mcs_sum = self.all_mcs_episode[:, new_start:t + 1].sum(axis=1)
            self._snir_sum = self.all_snir_episode[:, new_start:t + 1].sum(axis=1)
            self._ma_last_t = t
            return self._mcs_sum / n, self._snir_sum / n

        if t == self._ma_last_t:
            return self._mcs_sum / n, self._snir_sum / n

        # Add new columns (_ma_last_t + 1 .. t):
        add_lo = self._ma_last_t + 1
        self._mcs_sum += self.all_mcs_episode[:, add_lo:t + 1].sum(axis=1)
        self._snir_sum += self.all_snir_episode[:, add_lo:t + 1].sum(axis=1)
        # Drop columns leaving the window. old_start was the previous
        # window's left edge; new_start is the current one. Any columns
        # in [old_start, new_start) are no longer in the window.
        old_start = max(0, self._ma_last_t - p_window + 1)
        if new_start > old_start:
            self._mcs_sum -= self.all_mcs_episode[:, old_start:new_start].sum(axis=1)
            self._snir_sum -= self.all_snir_episode[:, old_start:new_start].sum(axis=1)
        self._ma_last_t = t
        return self._mcs_sum / n, self._snir_sum / n

    def _get_rl_obs(self) -> np.ndarray:
        t = min(self.t, self.Max_iter - 1)
        avg_mcs, avg_snir = self._advance_ma(t)

        nbs = self.NBS
        buf = self._obs_buf

        buf[0] = self.ue_speeds[t] / 30.0
        buf[1] = float(self.serving_tenure) / 1000.0

        # Serving one-hot at [2, 2+nbs)
        buf[2:2 + nbs] = 0.0
        s = self.ServingBSSector[t]
        if s != -1:
            buf[2 + s] = 1.0

        # RSRP at [2+nbs, 2+2*nbs)
        buf[2 + nbs:2 + 2 * nbs] = (self.PL3[:, t] + 75) / 45
        # MA(MCS) normalised by max capacity
        buf[2 + 2 * nbs:2 + 3 * nbs] = avg_mcs / 9.3
        # MA(SNIR) recentred and rescaled
        buf[2 + 3 * nbs:2 + 4 * nbs] = (avg_snir - 15) / 25.0

        buf[2 + 4 * nbs] = self.ue_positions[t, 0] / 500.0
        buf[2 + 4 * nbs + 1] = self.ue_positions[t, 1] / 500.0

        # Return a fresh copy: callers (main.py training loop) hold both
        # `state` and `next_state` simultaneously, so the buffer cannot be
        # shared between consecutive returns.
        return buf.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        filename = self.files[self.current_ue_idx % len(self.files)]
        self.current_ue_idx += 1
        
        # Fast loading from .npz — `with` ensures the file descriptor is
        # released even though we materialize the arrays we need.
        with np.load(filename) as data:
            cached_hash = str(data['physics_hash'].item()) if 'physics_hash' in data.files else None
            current_hash = physics_hash()
            if cached_hash != current_hash:
                raise RuntimeError(
                    f"Precomputed cache {filename} has physics_hash="
                    f"{cached_hash!r} but current physics is {current_hash!r}. "
                    "Rebuild with: "
                    "`./venv-RL/bin/python3 src/tools/preprocess_dataset.py`"
                )
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
        # Invalidate the running-MA cache for the new episode. The
        # underlying all_mcs_episode / all_snir_episode buffers were just
        # reloaded above, so any prior running sums are nonsense.
        self._ma_last_t = None

        self._sim_gen = self._simulation_generator()
        self._last_obs_dict = next(self._sim_gen)

        return self._get_rl_obs(), {"metrics": self._get_obs_dict(0)}

    def step(self, action, high_res_callback=None):
        reward = 0.0
        terminated = False
        truncated = False

        # Per-step config lookups were measurably hot. The values are
        # cached on the env in __init__; mutate via re-init if you ever
        # need to change them between episodes.
        alpha_ho = self._alpha_ho
        alpha_pp = self._alpha_pp
        alpha_hof = self._alpha_hof

        if high_res_callback:
            # High-res mode for baseline parity (100ms block = 10 ticks, callback each tick).
            # Computes the same pure-Ainna reward as RL mode (range-aggregated over
            # the actual tick range covered by these sends) so the LTM baseline gets
            # a meaningful reward signal comparable to DQN / QR-DQN.
            t_start = self._last_obs_dict['t']

            for _ in range(10):
                try:
                    act = high_res_callback(self._last_obs_dict)
                    self._last_obs_dict = self._sim_gen.send(act)
                except StopIteration:
                    terminated = True
                    break

            if not terminated:
                t_end = min(self._last_obs_dict['t'], self.Max_iter)
                if t_end > t_start:
                    mcs_slice = self.MCS[t_start:t_end]
                    avg_mcs = float(mcs_slice.mean())
                    has_ho = int(np.any(self.HO_event[t_start:t_end] > 0))
                    has_pp = int(np.any(self.ping_pong[t_start:t_end] > 0))
                    has_hof = int(np.any(self.HOF[t_start:t_end] > 0))
                else:
                    avg_mcs = 0.0
                    has_ho = has_pp = has_hof = 0

                multiplier = 1.0
                if has_ho: multiplier *= alpha_ho
                if has_pp: multiplier *= alpha_pp
                if has_hof: multiplier *= alpha_hof
                n_oos = self.Sync["out_sync_count"]
                reliability_factor = 1.0 / (1.0 + np.exp(2 * (n_oos - 2)))
                reward += avg_mcs * multiplier * reliability_factor
        else:
            # RL Mode — pure paper Ainna reward.
            #
            # MCS / HO_event / ping_pong / HOF are written by the simulation
            # generator at the START of each outer iteration, but the
            # generator yields AFTER the time-advance blocks have moved t
            # forward. So the freshly-yielded `self._last_obs_dict['t']`
            # points to a tick whose array slot is still the initialized
            # zero — it will be filled at the top of the NEXT outer iter,
            # AFTER this step() has returned. To get the correct values
            # for the reward we therefore aggregate over the actual range
            # [t_start, t_end) covered by this step's sends.
            t_start = self._last_obs_dict['t']

            for _ in range(10):
                try:
                    current_state = self._last_obs_dict['state_type']

                    if current_state == 1:  # FIND_CELL — force greedy recovery
                        ChBS2UE = self._last_obs_dict['ChBS2UE_t']
                        Pbest = np.max(ChBS2UE)
                        if Pbest + System["TxPower"] > ReceiverSensitivity:
                            act = np.argmax(ChBS2UE)
                        else:
                            act = -1
                        self._last_obs_dict = self._sim_gen.send(act)

                    elif current_state == 2:  # HO_DECISION
                        self._last_obs_dict = self._sim_gen.send(action)
                        action = -1  # Consume action

                    else:  # NORMAL_STEP
                        self._last_obs_dict = self._sim_gen.send(-1)

                except StopIteration:
                    terminated = True
                    break

            if not terminated:
                t_end = min(self._last_obs_dict['t'], self.Max_iter)
                if t_end > t_start:
                    mcs_slice = self.MCS[t_start:t_end]
                    avg_mcs = float(mcs_slice.mean())
                    has_ho = int(np.any(self.HO_event[t_start:t_end] > 0))
                    has_pp = int(np.any(self.ping_pong[t_start:t_end] > 0))
                    has_hof = int(np.any(self.HOF[t_start:t_end] > 0))
                else:
                    avg_mcs = 0.0
                    has_ho = has_pp = has_hof = 0

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
            PL3_report = self.PL3[:, t]
            Tf = min(t + _TICK_MEAS_L3, self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    break
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # 2 & 3. Preparation (RRC transfer + config)
            Tf = min(t + _TICK_RRC_PREP, self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    break
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
            Tf = min(t + _TICK_RRC_RECONF, self.Max_iter-1)
            while t < Tf:
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                if self.RLF[t]:
                    break
                t += 1
                self.t = t
                self.serving_tenure += 1
                self.ServingBSSector[t] = self.ServingBSSector[t - 1]
            if self.RLF[t]: NextBSSector = -1; continue

            # EXECUTION
            PL1_report = self.PL1[:, t]
            HO_condition = np.logical_and(self.ListBSPrepared, PL1_report > (PL1_report[self.ServingBSSector[t]] + HO["Prep"]["ExecPowerOffset"]))

            Tf = min(t + _TICK_MEAS_L1_HODEC, self.Max_iter-1)
            while t < Tf:
                self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                self.ReservedBSSectors[:, t] = self.ListBSPrepared
                if self.RLF[t]:
                    break
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
                    
                    Tf = min(t + _TICK_LL_HOCMD, self.Max_iter-1)
                    while t < Tf:
                        self.ReservedBSSectors[:, t] = self.ListBSPrepared
                        self.MCS[t], self.RLF[t], self.Sync = MCSEvaluation(self.ServingBSSector[t], self.ChBS2UE[:, t], System, self.Sync)
                        if self.RLF[t]:
                            break
                        t += 1
                        self.t = t
                        self.serving_tenure += 1
                        self.ServingBSSector[t] = self.ServingBSSector[t - 1]
                    if self.RLF[t]: NextBSSector = -1; continue

                    t = min(t + _TICK_RA, self.Max_iter-1)
                    self.t = t
                    self.serving_tenure += 1
                    self.HOF[t] = CheckHO_Failure(I, self.ChBS2UE[:, t], System)

                    if self.HOF[t] == 0:
                        NextBSSector = I
                        t = min(t + _TICK_CTX_RELEASE, self.Max_iter-1)
                        self.t = t
                        self.serving_tenure += 1

                        self.ReservedBSSectors[:, t0:t + 1] = np.tile(self.ListBSPrepared.reshape(-1, 1), (1, t - t0 + 1))
                        self.ListBSPrepared[NextBSSector] = 0
                        self.serving_tenure = 0
                    else:
                        NextBSSector = -1

                    self.ping_pong[t0] = (NextBSSector == former_BS) and ((t0 - former_HO_time) < _TICK_PINGPONG)
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
