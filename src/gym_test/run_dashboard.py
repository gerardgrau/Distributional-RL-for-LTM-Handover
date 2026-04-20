import torch
import numpy as np
import os
import sys

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.utils.dashboard import MobilityDashboard, get_mock_bs_positions

def run_agent_dashboard(agent_type="dqn", ue_idx=0):
    print(f"=== Generating Dashboard for Trained {agent_type.upper()} Agent (UE {ue_idx}) ===")
    
    # 1. Setup Env and Load Config
    env = LTMEnv()
    # Force the environment to a specific UE
    env.current_ue_idx = ue_idx
    
    config = Config.get()
    agent_cfg = config['agent']
    
    # 2. Setup Agent and Load best weights
    device = "cpu"
    if agent_type == "dqn":
        agent = DQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    else:
        agent = QRDQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
        
    model_path = f"results/models/{agent_type}_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return
            
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # 3. Run Episode and Collect History
    state, _ = env.reset()
    done = False
    
    history = {
        'ue_pos': [],
        'serving_bs': [],
        'rsrp': [],
    }
    
    print("Running episode...")
    while not done:
        # Record real position from the environment
        history['ue_pos'].append(env.ue_positions[env.t].tolist())
        history['serving_bs'].append(env.serving_sector)
        # Extract RSRP from the 67-dim state
        rsrp_vals = state[23:44]
        history['rsrp'].append(rsrp_vals.tolist())
        
        action = agent.select_action(state, epsilon=0)
        state, reward, done, _, _ = env.step(action)
        
    print(f"Episode finished at t={env.t}. Generating animation...")
    
    # 4. Generate Animation
    # Note: BS positions are still mock/circular. To improve this, we would need 
    # the actual BS coordinates used during the .mat generation.
    bs_pos = get_mock_bs_positions()
    dash = MobilityDashboard(bs_pos)
    
    save_path = f"results/animations/{agent_type}_ue{ue_idx}_mobility.gif"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dash.render_episode(history, save_path=save_path)
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "qrdqn"])
    parser.add_argument("--ue_idx", type=int, default=100, help="UE index (1 to 1000)")
    args = parser.parse_args()
    
    # Adjust for 1-based indexing in files
    run_agent_dashboard(args.agent, args.ue_idx - 1)
