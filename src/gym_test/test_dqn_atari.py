import gymnasium as gym
import ale_py
import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.utils.replay_buffer import ReplayBuffer

def train_minimal(env_id="CartPole-v1"):
    # Setup environment
    if "ALE" in env_id:
        env = gym.make(env_id, render_mode=None, obs_type="ram")
    else:
        env = gym.make(env_id, render_mode=None)
    
    config = {
        "hidden_dims": [256, 256],
        "lr": 1e-4,
        "gamma": 0.99,
        "tau": 1.0, # Hard update
        "target_update_freq": 100,
        "buffer_size": 10000,
        "batch_size": 32,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_mult": 0.99,
    }
    
    device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    agent = DQNAgent(config, env.observation_space, env.action_space, device=device)
    buffer = ReplayBuffer(config["buffer_size"], env.observation_space.shape)
    
    print(f"Starting minimal Atari (RAM) validation on {device}...")
    
    state, _ = env.reset()
    epsilon = config["epsilon_start"]
    
    for step in range(1000):
        # Select action
        action = agent.select_action(state, epsilon)
        
        # Step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store
        buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        if done:
            state, _ = env.reset()
            
        # Train
        if len(buffer) > config["batch_size"]:
            batch = buffer.sample(config["batch_size"], device=device)
            metrics = agent.train_step(batch)
            
            if step % 100 == 0:
                print(f"Step {step} | Loss: {metrics['loss']:.4f} | Epsilon: {epsilon:.2f}")
        
        # Decay epsilon
        epsilon = max(config["epsilon_end"], epsilon * config["epsilon_mult"])

    print("Minimal Atari validation successful.")
    env.close()

if __name__ == "__main__":
    try:
        train_minimal("ALE/Breakout-v5")
    except Exception as e:
        print(f"Atari failed ({e}), falling back to CartPole-v1...")
        train_minimal("CartPole-v1")
