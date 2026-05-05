import torch
import numpy as np
import os
import sys
import pandas as pd
from typing import Any

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.utils.metrics import calculate_8_metrics

def evaluate(agent_type: str, model_path: str, num_episodes: int = 20):
    print(f"\n--- Starting Evaluation ---")
    print(f"Agent: {agent_type}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    
    # Load config from configs/config.yaml (default)
    config = Config.get()
    
    # Ensure we use 1000 UEs for full diversity if possible
    config['simulation']['ue_number'] = 1000
    
    env = LTMEnv(config=config)
    
    if agent_type.lower() == "dqn":
        agent = DQNAgent(config['agent'], env.observation_space, env.action_space)
    elif agent_type.lower() == "qrdqn":
        agent = QRDQNAgent(config['agent'], env.observation_space, env.action_space)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    print("Loading weights...")
    agent.load(model_path)
    
    all_metrics = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        last_info = {}
        episode_reward = 0
        
        while not done:
            # Epsilon = 0 for pure greedy evaluation
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, _, info = env.step(action)
            episode_reward += reward
            if done:
                last_info = info
        
        # Calculate the 8 metrics
        m8 = calculate_8_metrics(
            mcs_history=last_info["metrics"]["mcs"],
            rlf_history=last_info["metrics"]["rlf"],
            ho_history=last_info["metrics"]["ho"],
            hof_history=last_info["metrics"]["hof"],
            pp_history=last_info["metrics"]["pp"],
            serving_history=last_info["metrics"]["serving"],
            pl3_history=last_info["metrics"]["pl3"],
            config=config
        )
        m8['reward'] = episode_reward
        all_metrics.append(m8)
        print(f"  Episode {ep+1}/{num_episodes} | Reward: {episode_reward:.2f} | RLF: {m8['rlf_rate']:.2f}")

    df = pd.DataFrame(all_metrics)
    summary = df.mean().to_dict()
    std = df.std().to_dict()
    
    print(f"\n--- Evaluation Results (Mean ± Std) ---")
    print(f"Capacity (bps):      {summary['capacity_avg']:.2f} ± {std['capacity_avg']:.2f}")
    print(f"RLF Rate (per min):  {summary['rlf_rate']:.2f} ± {std['rlf_rate']:.2f}")
    print(f"HO Rate (per min):   {summary['ho_rate']:.2f} ± {std['ho_rate']:.2f}")
    print(f"PP Rate (per min):   {summary['pp_rate']:.2f} ± {std['pp_rate']:.2f}")
    print(f"Reliability (%):     {summary['reliability_pct']:.2f} ± {std['reliability_pct']:.2f}")
    print(f"Prep Rate (per min): {summary['prep_rate']:.2f} ± {std['prep_rate']:.2f}")
    print(f"Res Reservation (%): {summary['res_reservation_pct']:.2f} ± {std['res_reservation_pct']:.2f}")
    print(f"HOF Rate (per min):  {summary['hof_rate']:.2f} ± {std['hof_rate']:.2f}")
    print(f"Total Reward:        {summary['reward']:.2f} ± {std['reward']:.2f}")
    print(f"----------------------------------------\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="dqn or qrdqn")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    args = parser.parse_args()
    
    evaluate(args.agent, args.model, args.episodes)
