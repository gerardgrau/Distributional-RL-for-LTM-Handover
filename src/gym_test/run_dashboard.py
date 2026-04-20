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
    state, _ = env.reset(seed=42)
    done = False
    
    history = {
        'ue_pos': [],
        'serving_bs': [],
        'rsrp': [],
    }
    
    print("Running episode...")
    # Ensure we visualize enough steps even after RLF
    total_vis_steps = 1000 
    
    for _ in range(total_vis_steps):
        # Record real position and signal (if available)
        t_idx = min(env.t, env.total_time - 1)
        history['ue_pos'].append(env.ue_positions[t_idx].tolist())
        history['serving_bs'].append(env.serving_sector)
        
        # Extract RSRP from state (index 23 to 43)
        rsrp_vals = state[23:44]
        history['rsrp'].append(rsrp_vals.tolist())
        
        if not done:
            action = agent.select_action(state, epsilon=0)
            state, reward, done, _, _ = env.step(action)
        else:
            # CONNECTION LOST: UE keeps moving, but we don't call step()
            # because the connection is dead. We manually advance env.t.
            env.t += 10 # Move by 100ms to match step duration
            if env.t >= env.total_time: break
        
    print(f"Recorded {len(history['ue_pos'])} steps for animation. Generating...")
    
    # 4. Generate Animation
    bs_pos = get_mock_bs_positions()
    dash = MobilityDashboard(bs_pos)
    
    save_path = f"results/animations/{agent_type}_ue{ue_idx}_fixed.gif"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dash.render_episode(history, save_path=save_path)
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "qrdqn"])
    parser.add_argument("--ue_idx", type=int, default=500)
    args = parser.parse_args()
    run_agent_dashboard(args.agent, args.ue_idx - 1)
