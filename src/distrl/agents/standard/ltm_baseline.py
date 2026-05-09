import numpy as np
import torch
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
        self.time_step = config.get("simulation", {}).get("time_step", 0.01)
        self.rl_step_time = 0.1 # 10 samples of 10ms
        
        # Internal state for the algorithm
        self.list_bs_prepared = np.zeros(self.nbs, dtype=bool)
        self.timer_entering = np.zeros(self.nbs)
        self.timer_leaving = np.zeros(self.nbs)
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        # 1. De-normalize state (RSRP)
        # norm_rsrp is at indices 23 to 23+21-1 (43)
        norm_rsrp = state[23:44]
        rsrp = norm_rsrp * 45 - 75
        
        # serving_one_hot is at indices 2 to 2+21-1 (22)
        serving_one_hot = state[2:23]
        serving_idx = np.argmax(serving_one_hot) if np.max(serving_one_hot) > 0 else -1
        
        if serving_idx == -1:
            # Recovery: pick strongest
            return int(np.argmax(rsrp))
            
        # 2. Update Preparation Logic
        # (This is a simplified version at 100ms resolution)
        in_condition = rsrp > (rsrp[serving_idx] + self.prep_offset)
        out_condition = rsrp < (rsrp[serving_idx] + self.prep_offset)
        
        # Since RL step is 100ms and prep_time is 40ms, if condition is met now, it counts as prepared
        self.timer_entering = (self.timer_entering + self.rl_step_time) * in_condition
        self.list_bs_prepared |= (self.timer_entering >= self.prep_time_thresh)
        
        self.timer_leaving = (self.timer_leaving + self.rl_step_time) * out_condition
        self.list_bs_prepared &= ~(self.timer_leaving >= self.prep_time_thresh)
        
        # Limit prepared BS (MaxNumberPreparedBS = 5)
        if np.sum(self.list_bs_prepared) > 5:
            # Use power-domain for sorting
            metric = (10 ** (rsrp / 10.0)) * self.list_bs_prepared
            i_sorted = np.argsort(metric)[::-1]
            self.list_bs_prepared[i_sorted[5:]] = False

        # 3. Execution Condition
        # HO if a prepared cell is stronger than serving + exec_offset
        ho_condition = self.list_bs_prepared & (rsrp > (rsrp[serving_idx] + self.exec_offset))
        
        if np.any(ho_condition):
            # Pick best candidate
            metric = (10 ** (rsrp / 10.0)) * ho_condition
            target_idx = np.argmax(metric)
            
            # Reset local state upon HO
            # Note: the environment will handle the actual switch
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
