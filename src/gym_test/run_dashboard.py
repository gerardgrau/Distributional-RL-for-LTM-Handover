import torch
import numpy as np
import os
import sys
import argparse

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.utils.dashboard import MobilityDashboard, get_mock_bs_positions

def run_agent_dashboard(agent_type="dqn", ue_idx=0, model_path=None, output_path=None):
    print(f"=== Generating Dashboard for Trained {agent_type.upper()} Agent (UE {ue_idx}) ===")
    
    env = LTMEnv()
    env.current_ue_idx = ue_idx
    config = Config.get()
    agent_cfg = config['agent']
    
    device = "cpu"
    if agent_type == "dqn":
        agent = DQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    else:
        agent = QRDQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
        
    if not model_path:
        model_path = f"results/models/{agent_type}_best.pth"
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return
            
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    state, _ = env.reset()
    done = False
    history = {'ue_pos': [], 'serving_bs': [], 'rsrp': []}
    
    print("Running episode...")
    # Track the real UE positions from the environment
    while not done:
        t_idx = min(env.t, env.total_time - 1)
        history['ue_pos'].append(env.ue_positions[t_idx].tolist())
        history['serving_bs'].append(env.serving_sector)
        rsrp_vals = state[23:44]
        history['rsrp'].append(rsrp_vals.tolist())
        
        action = agent.select_action(state, epsilon=0)
        state, reward, done, _, _ = env.step(action)
        
    print(f"Episode finished at t={env.t}. Generating animation...")
    
    bs_pos = get_mock_bs_positions()
    dash = MobilityDashboard(bs_pos)
    
    if not output_path:
        output_path = f"results/animations/{agent_type}_ue{ue_idx}_mobility.gif"
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dash.render_episode(history, save_path=output_path)
    print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "qrdqn"])
    parser.add_argument("--ue_idx", type=int, default=500, help="UE index (1 to 1000)")
    parser.add_argument("--model_path", type=str, help="Specific path to model weights")
    parser.add_argument("--output_path", type=str, help="Where to save the animation")
    args = parser.parse_args()
    
    run_agent_dashboard(args.agent, args.ue_idx - 1, args.model_path, args.output_path)
