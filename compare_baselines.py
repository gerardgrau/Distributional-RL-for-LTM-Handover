import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import src.distrl.envs.legacy_simulation as legacy
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.utils.metrics import calculate_8_metrics
from src.distrl.utils.config import Config
import re

legacy.UE_Number = 10

legacy_perf, _ = legacy.run_simulation()

Config.set_config_path("configs/config.yaml")
config = Config.get()
ue_count = 10
config['simulation']['ue_number'] = ue_count

env = LTMEnv(config=config)
agent = LTMBaselineAgent(config, env.observation_space, env.action_space)

gym_metrics = []

for i in range(1, ue_count + 1):
    filename = f"data/ChannelGains/ChannelGainBSUE_User{i}.mat"
    numeric_id = i
    np.random.seed(42 + numeric_id)
    
    # We must explicitly set the channel file for the environment
    # Since LTMEnv might not have a way to set file directly, let's rebuild it or override its file list.
    env.files = [filename]
    env.current_ue_idx = 0
    
    obs, info = env.reset()
    agent.reset()
    done = False
    
    while not done:
        obs, reward, done, truncated, info = env.step(0, high_res_callback=agent.select_action)
        
    if "metrics" in info:
        m = calculate_8_metrics(
            info["metrics"]["mcs"],
            info["metrics"]["rlf"],
            info["metrics"]["ho"],
            info["metrics"]["hof"],
            info["metrics"]["pp"],
            info["metrics"]["serving"],
            info["metrics"]["pl3"],
            config,
            reserved_history=info["metrics"].get("reserved")
        )
        gym_metrics.append(m)

print("\n=== PARITY COMPARISON (10 UEs) ===")
keys = ["Capacity", "Reliability", "RL_problems", "Number_HO", "Number_ping_pongs", "Number_cell_preparations", "Resource_reservation", "HOF"]
gym_keys = ["capacity_avg", "reliability_pct", "rlf_rate", "ho_rate", "pp_rate", "prep_rate", "res_reservation_pct", "hof_rate"]

for k_leg, k_gym in zip(keys, gym_keys):
    if k_leg == "Capacity":
        leg_val = np.mean([np.mean(p[k_leg]) for p in legacy_perf])
    else:
        leg_val = np.mean([p[k_leg] for p in legacy_perf])
        
    gym_val = np.mean([p[k_gym] for p in gym_metrics])
    
    print(f"{k_leg:25} | Legacy: {leg_val:.4f} | Gym: {gym_val:.4f} | Diff: {abs(leg_val - gym_val):.6f}")

