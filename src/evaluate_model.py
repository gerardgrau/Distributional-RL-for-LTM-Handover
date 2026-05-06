import torch
import numpy as np
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from typing import Any

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.utils.metrics import calculate_8_metrics

def evaluate(agent_type: str, model_path: str, config_path: str = "configs/config.yaml", num_episodes: int = 50, output_path: str = None):
    print(f"\n--- Starting Evaluation ---")
    print(f"Agent: {agent_type}")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Target Episodes: {num_episodes}")
    
    # Load custom config
    Config.set_config_path(config_path)
    config = Config.get()
    
    # Ensure we use 1000 UEs for full coverage as per the new protocol
    config['simulation']['ue_number'] = 1000
    
    # New Protocol: All users are used for both training and evaluation
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
    
    actual_episodes = min(num_episodes, len(env.files))
    
    for ep in tqdm(range(actual_episodes), desc="Evaluating"):
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

    df = pd.DataFrame(all_metrics)
    summary = {
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict()
    }
    
    print(f"\n--- Evaluation Results (Mean ± Std) ---")
    print(f"Capacity (bps):      {summary['mean']['capacity_avg']:.2f} ± {summary['std']['capacity_avg']:.2f}")
    print(f"RLF Rate (per min):  {summary['mean']['rlf_rate']:.2f} ± {summary['std']['rlf_rate']:.2f}")
    print(f"HO Rate (per min):   {summary['mean']['ho_rate']:.2f} ± {summary['std']['ho_rate']:.2f}")
    print(f"Reliability (%):     {summary['mean']['reliability_pct']:.2f} ± {summary['std']['reliability_pct']:.2f}")
    print(f"Total Reward:        {summary['mean']['reward']:.2f} ± {summary['std']['reward']:.2f}")
    print(f"----------------------------------------\n")

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        # 1. Save Summary JSON
        if output_path.endswith(".json"):
            json_path = output_path
            csv_path = output_path.replace(".json", ".csv")
        else:
            json_path = output_path + ".json"
            csv_path = output_path + ".csv"
            
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        # 2. Save Raw CSV
        df.to_csv(csv_path, index_label="eval_episode")
        
        print(f"Summary saved to {json_path}")
        print(f"Raw metrics saved to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="dqn or qrdqn")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--output", type=str, help="Path to save JSON output")
    args = parser.parse_args()
    
    evaluate(args.agent, args.model, args.config, args.episodes, args.output)
