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

    def select_action(self, state: np.ndarray | None, info: dict[str, Any] | None = None, epsilon: float = 0.0) -> int:
        # 1. De-normalize state (RSRP)
        # Prefer info rsrp if available
        if info is not None and "rsrp_l3" in info:
            curr_rsrp_l3 = info["rsrp_l3"]
            curr_rsrp_l1 = info.get("rsrp_l1", curr_rsrp_l3)
        elif state is not None:
            # observation layout: [speed(1), tenure(1), serving(nbs), rsrp(nbs), ...]
            rsrp_start = 2 + self.nbs
            rsrp_norm = state[rsrp_start : rsrp_start + self.nbs]
            curr_rsrp_l3 = rsrp_norm * 45 - 75
            curr_rsrp_l1 = curr_rsrp_l3
        else:
            raise ValueError("Baseline agent needs either state or info with rsrp_l3")

        # Determine serving_idx
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
            return int(np.argmax(rsrp_l3))
            
        # 2. Update Preparation Logic (L3 based)
        # Paper uses PL3_report > PL3_report[serving] + offset
        in_condition = rsrp_l3 > (rsrp_l3[serving_idx] + self.prep_offset)
        out_condition = rsrp_l3 < (rsrp_l3[serving_idx] + self.prep_offset)
        
        # Legacy code uses (Timer > 40ms), so it triggers at 50ms (tick 5).
        self.timer_entering = (self.timer_entering + self.rl_step_time) * in_condition
        self.list_bs_prepared |= (self.timer_entering > self.prep_time_thresh)
        
        self.timer_leaving = (self.timer_leaving + self.rl_step_time) * out_condition
        self.list_bs_prepared &= ~(self.timer_leaving > self.prep_time_thresh)
        
        # Limit prepared BS
        if np.sum(self.list_bs_prepared) > self.max_prep:
            metric = (10 ** (rsrp_l3 / 10.0)) * self.list_bs_prepared
            I_sorted = np.argsort(metric)[::-1]
            self.list_bs_prepared[I_sorted[self.max_prep:]] = False

        # 3. Execution Condition (L3 based for stability)
        ho_condition = self.list_bs_prepared & (rsrp_l3 > (rsrp_l3[serving_idx] + self.exec_offset))
        
        if np.any(ho_condition):
            metric = (10 ** (rsrp_l1 / 10.0)) * ho_condition
            target_idx = np.argmax(metric)
            self.list_bs_prepared[target_idx] = False
            return int(target_idx)
            
        return int(serving_idx)

    def train_step(self, batch: Any) -> dict[str, float]:
        # Baseline agent does not train
        return {"loss": 0.0}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
