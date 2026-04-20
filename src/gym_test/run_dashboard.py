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

def run_agent_dashboard(agent_type="dqn"):
    print(f"=== Generating Dashboard for Trained {agent_type.upper()} Agent ===")
    
    # 1. Setup Env and Load Config
    env = LTMEnv()
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
    state, _ = env.reset(seed=42)
    done = False
    
    history = {
        'ue_pos': [],
        'serving_bs': [],
        'rsrp': [],
    }
    
    print("Running episode...")
    while not done:
        history['ue_pos'].append([env.t * 2, np.sin(env.t/15.0) * 80])
        history['serving_bs'].append(env.serving_sector)
        # Extract RSRP from the 67-dim state
        rsrp_vals = state[23:44]
        history['rsrp'].append(rsrp_vals.tolist())
        
        action = agent.select_action(state, epsilon=0)
        state, reward, done, _, _ = env.step(action)
        
    print(f"Episode finished at t={env.t}. Generating animation...")
    
    # 4. Generate Animation
    bs_pos = get_mock_bs_positions()
    dash = MobilityDashboard(bs_pos)
    
    save_path = f"results/animations/{agent_type}_trained_mobility.gif"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dash.render_episode(history, save_path=save_path)
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "qrdqn"])
    args = parser.parse_args()
    run_agent_dashboard(args.agent)
