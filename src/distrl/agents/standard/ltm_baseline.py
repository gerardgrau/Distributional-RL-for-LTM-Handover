import numpy as np
from typing import Any
from src.distrl.agents.base import BaseAgent

class LTMBaselineAgent(BaseAgent):
    """
    Agent that implements the hardcoded LTM handover algorithm from the legacy simulation.
    It picks the strongest prepared cell based on 5G preparation/execution logic.
    """
    def __init__(self, config: dict[str, Any], observation_space: Any, action_space: Any, device: str = "cpu") -> None:
        super().__init__(config, observation_space, action_space, device)
        
        self.nbs = action_space.n
        self.prep_offset = config.get("ho_prep", {}).get("preparation_power_offset", -3)
        self.exec_offset = config.get("ho_prep", {}).get("exec_power_offset", 3)
        self.prep_time_thresh = config.get("ho_prep", {}).get("preparation_time", 0.04)
        self.max_prep = config.get("ho_prep", {}).get("max_number_prepared_bs", 5)
        self.rl_step_time = 0.01 # 10ms resolution
        
        # Parity: 20ms measurement periodicity
        self.measurement_periodicity = 0.02 
        self.last_measurement_time = -1.0
        self.cached_rsrp_l1 = None
        self.cached_rsrp_l3 = None
        
        # Legacy Parity: Internal 10-step procedural counter
        self.pending_ho_target = -1
        
        self.reset()
        
    def reset(self) -> None:
        """
        Resets preparation timers and prepared cell list.
        """
        self.list_bs_prepared = np.zeros(self.nbs, dtype=bool)
        self.timer_entering = np.zeros(self.nbs)
        self.timer_leaving = np.zeros(self.nbs)
        self.last_measurement_time = -1.0
        self.cached_rsrp_l1 = None
        self.cached_rsrp_l3 = None
        self.pending_ho_target = -1

    def select_action(self, state: np.ndarray | None, info: dict[str, Any] | None = None, epsilon: float = 0.0) -> int:
        # 1. De-normalize state (RSRP)
        if info is not None and "rsrp_l3" in info:
            curr_rsrp_l3 = info["rsrp_l3"]
            curr_rsrp_l1 = info.get("rsrp_l1", curr_rsrp_l3)
        elif state is not None:
            rsrp_start = 2 + self.nbs
            rsrp_norm = state[rsrp_start : rsrp_start + self.nbs]
            curr_rsrp_l3 = rsrp_norm * 45 - 75
            curr_rsrp_l1 = curr_rsrp_l3
        else:
            raise ValueError("Baseline agent needs either state or info with rsrp_l3")

        if info is not None and "serving_sector" in info:
            serving_idx = info["serving_sector"]
        elif state is not None:
            serving_start = 2
            serving_one_hot = state[serving_start : serving_start + self.nbs]
            serving_idx = np.argmax(serving_one_hot) if np.max(serving_one_hot) > 0 else -1
        else:
            serving_idx = -1

        rsrp_l3 = curr_rsrp_l3
        rsrp_l1 = curr_rsrp_l1
        
        if serving_idx == -1:
            self.pending_ho_target = -1
            return int(np.argmax(rsrp_l3))
            
        # 2. Legacy Parity Procedural Logic
        cycle = info.get("legacy_cycle", 0) if info is not None else 0
        
        # Execution trigger happens at cycle == 0 (from previous cycle == 8 decision)
        if cycle == 0:
            if getattr(self, 'pending_ho_target', -1) != -1:
                target = self.pending_ho_target
                self.pending_ho_target = -1
                return target

        # Legacy caches PL3 at cycle == 1
        if cycle == 1 or self.cached_rsrp_l3 is None:
            self.cached_rsrp_l3 = rsrp_l3.copy()
            
        # Preparation check happens at cycle == 6
        if cycle == 6:
            # Use cached PL3 to match legacy exact evaluation
            in_condition = self.cached_rsrp_l3 > (self.cached_rsrp_l3[serving_idx] + self.prep_offset)
            out_condition = self.cached_rsrp_l3 < (self.cached_rsrp_l3[serving_idx] + self.prep_offset)
            
            # Increment by exactly 10ms
            self.timer_entering = (self.timer_entering + 0.01) * in_condition
            self.list_bs_prepared |= (self.timer_entering > self.prep_time_thresh)
            
            self.timer_leaving = (self.timer_leaving + 0.01) * out_condition
            self.list_bs_prepared &= ~(self.timer_leaving > self.prep_time_thresh)
            
            if np.sum(self.list_bs_prepared) > self.max_prep:
                metric = (10 ** (self.cached_rsrp_l3 / 10.0)) * self.list_bs_prepared
                I_sorted = np.argsort(metric)[::-1]
                self.list_bs_prepared[I_sorted[self.max_prep:]] = False

        # Decision check happens at cycle == 8
        if cycle == 8:
            ho_condition = self.list_bs_prepared & (rsrp_l1 > (rsrp_l1[serving_idx] + self.exec_offset))
            if np.any(ho_condition):
                metric = (10 ** (rsrp_l1 / 10.0)) * ho_condition
                self.pending_ho_target = int(np.argmax(metric))
            else:
                self.pending_ho_target = -1

        return int(serving_idx)

    def train_step(self, batch: Any) -> dict[str, float]:
        return {"loss": 0.0}
    def save(self, path: str) -> None: pass
    def load(self, path: str) -> None: pass
