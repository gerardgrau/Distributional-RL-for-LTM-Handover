import numpy as np
import os
import sys
from typing import Any
import argparse
import time

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.utils.metrics import calculate_8_metrics

def verify_parity(ue_count: int, high_res: bool = False):
    Config.set_config_path("configs/config.yaml")
    config = Config.get()
    
    config['simulation']['ue_number'] = ue_count
    config['system']['tx_power'] = 25  # Force parity with legacy_simulation.py
    
    env = LTMEnv(config=config)
    agent = LTMBaselineAgent(config, env.observation_space, env.action_space)
    
    all_metrics = []
    mode_str = "10ms High-Res" if high_res else "100ms RL-Step"
    print(f"Running {mode_str} parity verification on {ue_count} UEs...")
    
    start_time = time.time()
    for i in range(ue_count):
        obs, info = env.reset()
        agent.reset()
        done = False
        
        while not done:
            if high_res:
                obs, reward, done, truncated, info = env.step(0, high_res_callback=agent.select_action)
            else:
                action = agent.select_action(obs, info=info)
                obs, reward, done, truncated, info = env.step(action)
            
        if "metrics" in info:
            m = calculate_8_metrics(
                info["metrics"]["mcs"],
                info["metrics"]["rlf"],
                info["metrics"]["ho"],
                info["metrics"]["hof"],
                info["metrics"]["pp"],
                info["metrics"]["serving"],
                info["metrics"]["pl3"],
                config
            )
            all_metrics.append(m)
            if (i+1) % 10 == 0:
                print(f"UE {i+1}/{ue_count} done. Mean Rel: {np.mean([x['reliability_pct'] for x in all_metrics]):.2f}%")

    total_duration = time.time() - start_time
    # Aggregate
    summary = {}
    for m in all_metrics:
        for k, v in m.items():
            if k not in summary: summary[k] = []
            summary[k].append(v)
            
    print(f"\n--- AGGREGATED METRICS ({mode_str} BASELINE) ---")
    for k, v in summary.items():
        if k in ["total_steps", "total_minutes"]: continue
        print(f"{k:20}: {np.mean(v):10.4f} (std: {np.std(v):.4f})")
        
    print(f"\nExecution Time: {total_duration:.2f}s ({total_duration/ue_count:.3f} s/UE)")
    print("\n--- PAPER LTM REFERENCE ---")
    print("capacity_avg       : 3.75")
    print("reliability_pct    : 95.00")
    print("rlf_rate           : 0.068")
    print("ho_rate            : 11.00")
    print("pp_rate            : 3.45")
    print("prep_rate          : 780.00")
    print("res_reservation_pct: 5.70")
    print("hof_rate           : 1.10")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ue_number", type=int, default=20)
    parser.add_argument("--high_res", action="store_true")
    args = parser.parse_args()
    verify_parity(args.ue_number, args.high_res)
