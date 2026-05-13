import numpy as np
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.utils.config import Config

def trace_and_compare():
    with open("legacy_trace_ue1.pkl", "rb") as f:
        legacy_trace = pickle.load(f)
        
    Config.set_config_path("configs/config.yaml")
    config = Config.get()
    config['simulation']['ue_number'] = 1
    config['system']['tx_power'] = 25 # Standard
    
    # Same setup as compare_ue1.py
    Config.set_config_path("configs/config.yaml")
    config = Config.get()
    config['simulation']['ue_number'] = 1
    config['system']['tx_power'] = 25
    
    env = LTMEnv(config=config)
    agent = LTMBaselineAgent(config, env.observation_space, env.action_space)
    
    obs, info = env.reset()
    agent.reset()
    done = False
    
    while not done:
        obs, reward, done, truncated, info = env.step(0, high_res_callback=agent.select_action)
    gym_serving = env.metrics_serving
    gym_mcs = env.metrics_mcs
    gym_ho = env.metrics_ho
    gym_rlf = env.metrics_rlf
    
    leg_serving = legacy_trace["serving"]
    leg_mcs = legacy_trace["mcs"]
    leg_ho = legacy_trace["ho"]
    leg_rlf = legacy_trace["rlf"]
    leg_reserved = legacy_trace.get("reserved", np.zeros((21, len(leg_serving))))
    
    gym_reserved = info["metrics"]["reserved"]
    
    # Compare step by step
    for t in range(10, len(leg_serving)):
        div_serving = gym_serving[t] != leg_serving[t]
        div_reserved = not np.array_equal(gym_reserved[:, t], leg_reserved[:, t])
        
        if div_serving or div_reserved:
            print(f"Divergence at t={t}: Serving={div_serving}, Reserved={div_reserved}")
            
            print("\nDetailed trace around divergence:")
            print(f"{'t':<6} | {'Gym S':<6} | {'Leg S':<6} | {'Gym HO':<6} | {'Leg HO':<6} | {'Gym Prep':<8} | {'Leg Prep':<8}")
            for i in range(max(0, t-15), min(len(leg_serving), t+15)):
                g_prep = str(np.where(gym_reserved[:, i])[0].tolist())
                l_prep = str(np.where(leg_reserved[:, i])[0].tolist())
                print(f"{i:<6} | {gym_serving[i]:<6} | {leg_serving[i]:<6} | {gym_ho[i]:<6} | {leg_ho[i]:<6} | {g_prep:<8} | {l_prep:<8}")
            break

if __name__ == "__main__":
    trace_and_compare()
