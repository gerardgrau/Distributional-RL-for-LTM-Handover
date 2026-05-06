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
from src.distrl.utils.evaluation import run_evaluation

def evaluate(agent_type: str, model_path: str, config_path: str = "configs/config.yaml", episodes: int = 1000, output: str = None):
    print(f"\n--- Starting Standalone Evaluation ---")
    print(f"Agent:  {agent_type}")
    print(f"Model:  {model_path}")
    print(f"Config: {config_path}")
    
    # Load custom config
    Config.set_config_path(config_path)
    config = Config.get()
    
    # Ensure we use the correct number of UEs
    # Note: run_evaluation defaults to all available files if we don't constrain it.
    
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
    
    # Setup directory for results if output is provided
    experiment_dir = "."
    if output:
        experiment_dir = os.path.dirname(output) or "."
        os.makedirs(experiment_dir, exist_ok=True)
        
    # Use the unified evaluation utility
    run_evaluation(
        agent=agent,
        config=config,
        experiment_dir=experiment_dir,
        agent_type=agent_type,
        seed=42,
        save_results=True if output else False
    )

    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="dqn or qrdqn")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes (unused, always uses full dataset)")
    parser.add_argument("--output", type=str, help="Path to save evaluation summary CSV")
    args = parser.parse_args()
    
    evaluate(args.agent, args.model, args.config, args.episodes, args.output)
