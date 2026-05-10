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
        self.rl_step_time = 0.1 # 10 samples of 10ms
        
        # Internal state for the algorithm
        self.reset()
        
    def reset(self) -> None:
        """
        Resets preparation timers and prepared cell list.
        """
        self.list_bs_prepared = np.zeros(self.nbs, dtype=bool)
        self.timer_entering = np.zeros(self.nbs)
        self.timer_leaving = np.zeros(self.nbs)

    def select_action(self, state: np.ndarray, info: dict[str, Any] | None = None, epsilon: float = 0.0) -> int:
        # 1. De-normalize state (RSRP)
        # observation layout: [speed(1), tenure(1), serving(nbs), rsrp(nbs), ...]
        serving_start = 2
        rsrp_start = serving_start + self.nbs
        
        serving_one_hot = state[serving_start:rsrp_start]
        
        # Prefer info rsrp if available
        if info is not None and "rsrp_l3" in info:
            rsrp_l3 = info["rsrp_l3"]
            rsrp_l1 = info.get("rsrp_l1", rsrp_l3)
        else:
            rsrp_norm = state[rsrp_start : rsrp_start + self.nbs]
            # De-normalize RSRP: Map [-1, 1] back to [-120, -30] (approx)
            rsrp_l3 = rsrp_norm * 45 - 75
            rsrp_l1 = rsrp_l3
        
        serving_idx = np.argmax(serving_one_hot) if np.max(serving_one_hot) > 0 else -1
        
        if serving_idx == -1:
            # Recovery: pick strongest
            return int(np.argmax(rsrp_l3))
            
        # 2. Update Preparation Logic (L3 based)
        in_condition = rsrp_l3 > (rsrp_l3[serving_idx] + self.prep_offset)
        out_condition = rsrp_l3 < (rsrp_l3[serving_idx] + self.prep_offset)
        
        # Since RL step is 100ms and prep_time is 40ms, if condition is met now, it counts as prepared
        self.timer_entering = (self.timer_entering + self.rl_step_time) * in_condition
        self.list_bs_prepared |= (self.timer_entering >= self.prep_time_thresh)
        
        self.timer_leaving = (self.timer_leaving + self.rl_step_time) * out_condition
        self.list_bs_prepared &= ~(self.timer_leaving >= self.prep_time_thresh)
        
        # Limit prepared BS
        if np.sum(self.list_bs_prepared) > self.max_prep:
            # Use power-domain for sorting
            metric = (10 ** (rsrp_l3 / 10.0)) * self.list_bs_prepared
            I_sorted = np.argsort(metric)[::-1]
            self.list_bs_prepared[I_sorted[self.max_prep:]] = False

        # 3. Execution Condition (L1 based per 3GPP LTM)
        # HO if a prepared cell is stronger than serving + exec_offset
        ho_condition = self.list_bs_prepared & (rsrp_l1 > (rsrp_l1[serving_idx] + self.exec_offset))
        
        if np.any(ho_condition):
            # Pick best candidate
            metric = (10 ** (rsrp_l1 / 10.0)) * ho_condition
            target_idx = np.argmax(metric)
            
            # Reset local state upon HO
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
