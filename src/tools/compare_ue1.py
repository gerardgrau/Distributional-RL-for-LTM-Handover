import numpy as np
import os
import sys
import pickle
from scipy.io import loadmat
from scipy.signal import lfilter

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.utils.config import Config
from src.distrl.utils.metrics import calculate_8_metrics
import src.distrl.envs.legacy_simulation as legacy

def run_legacy_ue1():
    print("\n--- Running Legacy Simulation (UE 1) ---")
    # Backup original UE_Number
    orig_ue_num = legacy.UE_Number
    legacy.UE_Number = 1
    perf_all, metrics_all = legacy.run_simulation()
    legacy.UE_Number = orig_ue_num
    
    perf = perf_all[0]
    m = {
        "ho_rate": perf["Number_HO"],
        "hof_rate": perf["HOF"],
        "pp_rate": perf["Number_ping_pongs"],
        "capacity_avg": np.mean(perf["Capacity"]),
        "rlf_rate": perf["RL_problems"],
        "reliability_pct": perf["Reliability"],
        "res_reservation_pct": perf["Resource_reservation"],
    }
    # Derivació prep_rate (com a generate_legacy_summary.py)
    sum_reserved = (perf["Resource_reservation"] / 100.0) * 21 * 30000
    m["prep_rate"] = (sum_reserved / 10.0) / 5.0
    
    return m

def run_gym_ue1():
    print("\n--- Running Gym Environment (UE 1) ---")
    Config.set_config_path("configs/config.yaml")
    config = Config.get()
    config['simulation']['ue_number'] = 1
    config['system']['tx_power'] = 25 # Standard
    
    env = LTMEnv(config=config)
    agent = LTMBaselineAgent(config, env.observation_space, env.action_space)
    
    obs, info = env.reset()
    agent.reset()
    done = False
    
    while not done:
        # RL Step is 100ms, but we use the high_res_callback to simulate 10ms ticks
        obs, reward, done, truncated, info = env.step(0, high_res_callback=agent.select_action)
    
    print("\nGYM DEBUG SNIR (first 20 steps):")
    for t in range(20):
        s = env.metrics_serving[t]
        if s != -1:
            snir = env.all_snir_episode[s, t]
            ps = 10 ** ((env.ch_bs2ue[s, t] + 25) / 10.0)
        else:
            snir = -100
            ps = 0
        print(f"GYM_SNIR: t={t} s={s} snir={snir:.2f} ps={ps:.2e}")
        
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
    return m

def compare():
    legacy_m = run_legacy_ue1()
    gym_m = run_gym_ue1()
    
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'Legacy':<12} | {'Gym':<12} | {'Diff %':<10}")
    print("-"*50)
    
    metrics = ["ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate", "reliability_pct", "prep_rate", "res_reservation_pct"]
    for k in metrics:
        l_val = legacy_m[k]
        g_val = gym_m[k]
        diff = 0
        if abs(l_val) > 1e-9:
            diff = abs(l_val - g_val) / abs(l_val) * 100
        elif abs(g_val) > 1e-9:
            diff = 100.0
            
        print(f"{k:<20} | {l_val:12.4f} | {g_val:12.4f} | {diff:9.4f}%")
    print("="*50)

if __name__ == "__main__":
    compare()
