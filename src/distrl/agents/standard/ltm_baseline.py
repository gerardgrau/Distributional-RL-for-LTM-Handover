import numpy as np
from src.distrl.envs.physics import System, ReceiverSensitivity

class LTMBaselineAgent:
    def __init__(self, config=None, observation_space=None, action_space=None):
        pass

    def reset(self):
        pass

    def select_action(self, obs):
        state_type = obs['state_str']
        
        if state_type == 'FIND_CELL':
            # Greedy Selection strategy for broken connections
            ChBS2UE = obs['ChBS2UE_t']
            Pbest = np.max(ChBS2UE)
            Best = np.argmax(ChBS2UE)
            if Pbest + System["TxPower"] > ReceiverSensitivity:
                return Best
            return -1
            
        elif state_type == 'HO_DECISION':
            # Greedy execution on handovers conditionally met
            PL1_report = obs['PL1_report']
            HO_condition = obs['HO_condition']
            metric = (10 ** (PL1_report / 10)) * HO_condition
            if np.sum(metric) > 0:
                return np.argmax(metric)
            return -1
            
        else:
            return -1