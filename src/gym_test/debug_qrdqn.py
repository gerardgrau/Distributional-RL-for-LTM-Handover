import torch
import numpy as np
import gymnasium as gym
from src.distrl.agents.distributional.qrdqn import QRDQNAgent

def test_qrdqn_select():
    print("Testing QRDQN select_action...")
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(88,))
    act_space = gym.spaces.Discrete(21)
    config = {
        'hidden_dims': [128, 128],
        'num_quantiles': 50
    }
    agent = QRDQNAgent(config, obs_space, act_space, device="cpu")
    obs = obs_space.sample()
    print("Selecting action...")
    action = agent.select_action(obs, epsilon=0.0)
    print(f"Action selected: {action}")

if __name__ == "__main__":
    test_qrdqn_select()
