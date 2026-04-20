import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.utils.replay_buffer import ReplayBuffer

def train_ltm_minimal():
    env = LTMEnv()
    
    config = {
        "hidden_dims": [64, 64],
        "lr": 1e-3,
        "gamma": 0.9,
        "tau": 0.01,
        "buffer_size": 5000,
        "batch_size": 64,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 2000
    }
    
    device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    agent = DQNAgent(config, env.observation_space, env.action_space, device=device)
    buffer = ReplayBuffer(config["buffer_size"], env.observation_space.shape)
    
    print(f"Starting DQN training on LTM-HO simulation ({device})...")
    
    num_episodes = 20
    epsilon = config["epsilon_start"]
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(buffer) > config["batch_size"]:
                agent.train_step(buffer.sample(config["batch_size"], device=device))
                
            # Decay epsilon per step
            epsilon = max(config["epsilon_end"], epsilon - (config["epsilon_start"] - config["epsilon_end"]) / config["epsilon_decay"])
            
        print(f"Episode {ep+1}/{num_episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")

    print("DQN LTM-HO validation successful.")

if __name__ == "__main__":
    train_ltm_minimal()
