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


# Tick counts for each HO sub-phase. Precomputed so the simulation
# generator avoids re-running int(np.ceil(...)) on every iteration.
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

        # Cache per-step config lookups.
        self._p_window = int(self.config.get('ho_state', {}).get(
            'moving_average_window', 50,
        ))
        ho_reward_cfg = self.config.get('ho_reward', {})
        self._alpha_ho = float(ho_reward_cfg.get('alpha_ho', 0.8))
        self._alpha_pp = float(ho_reward_cfg.get('alpha_pp', 0.9))
        self._alpha_hof = float(ho_reward_cfg.get('alpha_hof', 0.1))

        # Tier-0 "decline" probe. When True, an L1-triggered HO_DECISION is
        # only registered as a handover (HO event + execution + alpha_HO
        # penalty) if the agent picks a cell DIFFERENT from the current
        # serving one; picking the serving cell is an explicit stay, handled
        # exactly like a NORMAL_STEP. When False (default) the legacy
        # behaviour is preserved: any action at the trigger registers a HO,
        # so picking the serving cell is a counted self-handover. The LTM
        # baseline never selects its own serving cell, so this flag does not
        # affect baseline parity.
        self._ho_requires_change = bool(
            self.config.get('simulation', {}).get(
                'handover_requires_serving_change', False,
            )
        )

        # Reusable buffer for _get_rl_obs (returned as .copy() each call).
        self._obs_buf = np.empty(88, dtype=np.float32)

        # O(1) running sums for the per-sector MA used in the observation.
        self._ma_last_t: int | None = None
        self._mcs_sum = np.zeros(self.NBS)
        self._snir_sum = np.zeros(self.NBS)

        # Per-episode bookkeeping arrays — allocated once, zeroed in-place
        # by reset(). The reset path reallocates iff a loaded .npz has a
        # different Max_iter (not expected for the canonical dataset).
        self._allocate_episode_buffers(self.Max_iter)

    def _allocate_episode_buffers(self, max_iter: int) -> None:
        self.ServingBSSector = np.full(max_iter, -1, dtype=int)
        self.MCS = np.zeros(max_iter)
        self.HOF = np.zeros(max_iter)
        self.RLF = np.zeros(max_iter)
        self.HO_event = np.zeros(max_iter)
        self.ping_pong = np.zeros(max_iter)
        self.ReservedBSSectors = np.zeros((self.NBS, max_iter))
        self.ListBSPrepared = np.zeros(self.NBS, dtype=int)
        self.TimerEntering = np.zeros(self.NBS)
        self.TimerLeaving = np.zeros(self.NBS)

    def _reset_episode_buffers(self) -> None:
        self.ServingBSSector.fill(-1)
        self.MCS.fill(0)
        self.HOF.fill(0)
        self.RLF.fill(0)
        self.HO_event.fill(0)
        self.ping_pong.fill(0)
        self.ReservedBSSectors.fill(0)
        self.ListBSPrepared.fill(0)
        self.TimerEntering.fill(0)
        self.TimerLeaving.fill(0)

    def _get_obs_dict(self, t, state_type='NORMAL_STEP', HO_condition=None, PL1_report=None):
        """Per-yield observation dict consumed by env.step.

        Only the fields actually used for `state_type` are populated:
        NORMAL_STEP reads only `state_type` + `t`; FIND_CELL adds the
        channel snapshot; HO_DECISION adds the L1 report + HO mask.
        """
        obs = {
            "state_type": _STATE_MAP[state_type],
            "state_str": state_type,
            "t": t,
        }
        if state_type == 'FIND_CELL':
            obs["ChBS2UE_t"] = self.ChBS2UE[:, t]
        elif state_type == 'HO_DECISION':
            obs["PL1_report"] = PL1_report
            obs["HO_condition"] = HO_condition
        return obs

    def _advance_ma(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-sector window averages of MCS / SNIR up to time `t`.

        Maintains running sums over the trailing `_p_window` ticks
        (slid forward in-place between consecutive calls) so the cost
        per call is O(advance) rather than O(window).
        """
        p_window = self._p_window
        new_start = max(0, t - p_window + 1)
        n = (t + 1) - new_start

        if self._ma_last_t is None or t < self._ma_last_t:
            # First call after reset (or t walked back) — rebuild from scratch.
            self._mcs_sum = self.all_mcs_episode[:, new_start:t + 1].sum(axis=1)
            self._snir_sum = self.all_snir_episode[:, new_start:t + 1].sum(axis=1)
            self._ma_last_t = t
            return self._mcs_sum / n, self._snir_sum / n

        if t == self._ma_last_t:
            return self._mcs_sum / n, self._snir_sum / n

        add_lo = self._ma_last_t + 1
        self._mcs_sum += self.all_mcs_episode[:, add_lo:t + 1].sum(axis=1)
        self._snir_sum += self.all_snir_episode[:, add_lo:t + 1].sum(axis=1)
        old_start = max(0, self._ma_last_t - p_window + 1)
        if new_start > old_start:
            self._mcs_sum -= self.all_mcs_episode[:, old_start:new_start].sum(axis=1)
            self._snir_sum -= self.all_snir_episode[:, old_start:new_start].sum(axis=1)
        self._ma_last_t = t
        return self._mcs_sum / n, self._snir_sum / n

    def _get_rl_obs(self) -> np.ndarray:
        """88-dim Markovian observation: speed, tenure, one-hot serving,
        RSRP[NBS], MA(MCS)[NBS], MA(SNIR)[NBS], (x, y)."""
        t = min(self.t, self.Max_iter - 1)
        avg_mcs, avg_snir = self._advance_ma(t)

        nbs = self.NBS
        buf = self._obs_buf

        buf[0] = self.ue_speeds[t] / 30.0
        buf[1] = float(self.serving_tenure) / 1000.0

        buf[2:2 + nbs] = 0.0
        s = self.ServingBSSector[t]
        if s != -1:
            buf[2 + s] = 1.0

        buf[2 + nbs:2 + 2 * nbs] = (self.PL3[:, t] + 75) / 45
        buf[2 + 2 * nbs:2 + 3 * nbs] = avg_mcs / 9.3
        buf[2 + 3 * nbs:2 + 4 * nbs] = (avg_snir - 15) / 25.0

        buf[2 + 4 * nbs] = self.ue_positions[t, 0] / 500.0
        buf[2 + 4 * nbs + 1] = self.ue_positions[t, 1] / 500.0

        # Caller holds `state` and `next_state` simultaneously — return a
        # copy so consecutive env.step returns don't alias the buffer.
        return buf.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        filename = self.files[self.current_ue_idx % len(self.files)]
        self.current_ue_idx += 1

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

        loaded_max = self.ChBS2UE.shape[1]
        if loaded_max != self.Max_iter:
            self.Max_iter = loaded_max
            self._allocate_episode_buffers(loaded_max)
        else:
            self._reset_episode_buffers()

        self.ue_speeds = np.full(self.Max_iter, 10.0)
        self.Sync = {
            "N310": 4, "N311": 2, "T310": 50,
            "out_sync_count": 0, "in_sync_count": 0,
            "t310_running": False, "t310_counter": np.inf
        }
        self.t = 0
        self.serving_tenure = 0
        self._ma_last_t = None  # Invalidate the running-MA cache.

        self._sim_gen = self._simulation_generator()
        self._last_obs_dict = next(self._sim_gen)
        return self._get_rl_obs(), {"metrics": self._get_obs_dict(0)}

    def step(self, action, high_res_callback=None):
        reward = 0.0
        terminated = False
        truncated = False
        t_start = self._last_obs_dict['t']

        for _ in range(10):
            try:
                if high_res_callback is not None:
                    # Baseline-parity mode: callback fires on every tick.
                    act = high_res_callback(self._last_obs_dict)
                    self._last_obs_dict = self._sim_gen.send(act)
                else:
                    # RL mode: env handles FIND_CELL greedy recovery
                    # automatically; agent's action is consumed at the
                    # HO_DECISION yield; NORMAL_STEP yields ignore action.
                    current_state = self._last_obs_dict['state_type']
                    if current_state == 1:  # FIND_CELL
                        ChBS2UE = self._last_obs_dict['ChBS2UE_t']
                        Pbest = np.max(ChBS2UE)
                        if Pbest + System["TxPower"] > ReceiverSensitivity:
                            act = np.argmax(ChBS2UE)
                        else:
                            act = -1
                        self._last_obs_dict = self._sim_gen.send(act)
                    elif current_state == 2:  # HO_DECISION
                        self._last_obs_dict = self._sim_gen.send(action)
                        action = -1
                    else:  # NORMAL_STEP
                        self._last_obs_dict = self._sim_gen.send(-1)
            except StopIteration:
                terminated = True
                break

        if not terminated:
            # Reward aggregates over [t_start, t_end) rather than the
            # latest yielded tick — MCS/HO/PP/HOF for tick t are written
            # at the TOP of the NEXT outer iter, so sampling at the
            # just-yielded t would read a zero-initialised slot.
            t_end = min(self._last_obs_dict['t'], self.Max_iter)
            reward = self._compute_step_reward(t_start, t_end)

        info = {}
        if terminated:
            info["metrics"] = self._build_metrics_dict()
        return self._get_rl_obs(), reward, terminated, truncated, info

    def _compute_step_reward(self, t_start: int, t_end: int) -> float:
        """Multiplicative-Ainna reward over the simulator tick range
        [t_start, t_end)."""
        if t_end <= t_start:
            return 0.0
        avg_mcs = float(self.MCS[t_start:t_end].mean())
        multiplier = 1.0
        if np.any(self.HO_event[t_start:t_end] > 0):
            multiplier *= self._alpha_ho
        if np.any(self.ping_pong[t_start:t_end] > 0):
            multiplier *= self._alpha_pp
        if np.any(self.HOF[t_start:t_end] > 0):
            multiplier *= self._alpha_hof
        n_oos = self.Sync["out_sync_count"]
        reliability_factor = 1.0 / (1.0 + np.exp(2 * (n_oos - 2)))
        return avg_mcs * multiplier * reliability_factor

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
                # In the Tier-0 decline regime, selecting the current serving
                # cell is a stay: skip the HO entirely (no event, no penalty)
                # and fall through like a NORMAL_STEP cycle. Short-circuits to
                # the legacy path when the flag is off.
                _decline = self._ho_requires_change and action == self.ServingBSSector[t]
                if action != -1 and not _decline:
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
