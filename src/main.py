import torch
import numpy as np
import os
import sys

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.utils.replay_buffer import ReplayBuffer

def run_experiment():
    """
    Main entry point for running RL experiments.
    """
    print("=== Distributional RL for LTM Handover ===")
    
    # 1. Load Configuration
    config = Config.get()
    agent_cfg = config['agent']
    sim_cfg = config['simulation']
    
    print(f"Loading experiment: Agent={agent_cfg['type']}, UEs={sim_cfg['ue_number']}")
    
    # 2. Setup Environment
    env = LTMEnv()
    
    # 3. Setup Agent
    device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    
    if agent_cfg['type'].lower() == "dqn":
        agent = DQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    elif agent_cfg['type'].lower() == "qrdqn":
        agent = QRDQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_cfg['type']}")
    
    # 4. Setup Buffer
    buffer = ReplayBuffer(agent_cfg['buffer_size'], env.observation_space.shape)
    
    # 5. Training Loop
    print(f"Starting training on {device}...")
    
    num_episodes = agent_cfg.get('num_episodes', 20)
    epsilon = agent_cfg.get('epsilon_start', 1.0)
    eps_decay = agent_cfg.get('epsilon_decay', 2000)
    eps_end = agent_cfg.get('epsilon_end', 0.05)
    batch_size = agent_cfg.get('batch_size', 64)
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Step in environment
            next_state, reward, done, _, _ = env.step(action)
            
            # Store transition
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train agent
            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size, device=device)
                agent.train_step(batch)
                
            # Decay epsilon
            epsilon = max(eps_end, epsilon - (agent_cfg['epsilon_start'] - eps_end) / eps_decay)
            
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Episode {ep+1}/{num_episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")

    print("Experiment completed successfully.")
    
    # 6. Save Model
    save_path = f"results/models/{agent_cfg['type']}_final.pth"
    agent.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_experiment()
