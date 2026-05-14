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
    
    import re
    from tqdm import tqdm
    start_time = time.time()
    for i in tqdm(range(ue_count), desc="Simulating UEs"):
        # Enforce exactly the same numpy seed per UE as the legacy script
        filename = env.files[env.current_ue_idx % len(env.files)]
        numeric_id = int(re.search(r'\d+', os.path.basename(filename)).group())
        np.random.seed(42 + numeric_id)
        
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
                config,
                reserved_history=info["metrics"].get("reserved")
            )
            all_metrics.append(m)

    total_duration = time.time() - start_time
    # Aggregate
    summary = {}
    for m in all_metrics:
        for k, v in m.items():
            if k not in summary: summary[k] = []
            summary[k].append(v)
            
    print(f"\n--- AGGREGATED METRICS ({mode_str} BASELINE) ---")
    
    import pandas as pd
    metric_order = [
        "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
        "reliability_pct", "prep_rate", "res_reservation_pct", "total_steps", "total_minutes"
    ]
    
    out_data = []
    for k in metric_order:
        if k in summary:
            v = summary[k]
            mean_val = np.mean(v)
            std_val = np.std(v)
            print(f"{k:20}: {mean_val:10.4f} (std: {std_val:.4f})")
            out_data.append({"metric": k, "mean": mean_val, "std": std_val})

    df = pd.DataFrame(out_data)
    # Add dummy reward to match old format
    df.loc[len(df.index)] = ['reward', 0.0, 0.0] 
    
    out_path = "results/final_metrics/baseline_summary.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved updated gym baseline metrics to {out_path}")
    
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
