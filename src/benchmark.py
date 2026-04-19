import torch
import numpy as np
import os
import sys
import time
import csv
from datetime import datetime
from typing import Any

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.utils.replay_buffer import ReplayBuffer

class CSVLogger:
    def __init__(self, filepath: str, headers: list[str]):
        self.filepath = filepath
        self.headers = headers
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, row: list[Any]):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def run_seed(agent_type: str, env_name: str, seed: int, config: dict, experiment_dir: str):
    print(f"  -> Starting Seed {seed} for {agent_type}...")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup Env
    if env_name == "ltm":
        env = LTMEnv()
    else:
        import gymnasium as gym
        env = gym.make(env_name)
    
    # Setup Agent
    agent_cfg = config['agent']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if agent_type.lower() == "dqn":
        agent = DQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    elif agent_type.lower() == "qrdqn":
        agent = QRDQNAgent(agent_cfg, env.observation_space, env.action_space, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    buffer = ReplayBuffer(agent_cfg['buffer_size'], env.observation_space.shape)
    
    # Setup Logger
    log_file = os.path.join(experiment_dir, f"{agent_type}_{env_name.replace('/', '_')}_seed{seed}.csv")
    headers = ["episode", "reward", "loss", "steps", "wall_time"]
    logger = CSVLogger(log_file, headers)
    
    # Training Loop
    num_episodes = agent_cfg.get('num_episodes', 20)
    epsilon = agent_cfg.get('epsilon_start', 1.0)
    eps_decay = agent_cfg.get('epsilon_decay', 2000)
    eps_end = agent_cfg.get('epsilon_end', 0.05)
    batch_size = agent_cfg.get('batch_size', 64)
    
    start_time = time.time()
    total_steps = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        episode_steps = 0
        episode_loss = []
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if len(buffer) > batch_size:
                metrics = agent.train_step(buffer.sample(batch_size, device=device))
                episode_loss.append(metrics['loss'])
                
            epsilon = max(eps_end, epsilon - (agent_cfg['epsilon_start'] - eps_end) / eps_decay)
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        current_wall_time = time.time() - start_time
        
        logger.log([ep + 1, episode_reward, avg_loss, episode_steps, current_wall_time])
        
        if (ep + 1) % 10 == 0:
            print(f"    Seed {seed} | Ep {ep+1}/{num_episodes} | Reward: {episode_reward:.2f} | Time: {current_wall_time:.1f}s")

    env.close()

def run_benchmark():
    config = Config.get()
    bench_cfg = config['benchmark']
    agent_types = ["dqn", "qrdqn"] # We want to compare both
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(bench_cfg['results_dir'], f"benchmark_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"=== Starting Benchmark Suite: {experiment_dir} ===")
    
    env_name = bench_cfg['env_type'] # Typically 'ltm'
    
    for agent_type in agent_types:
        print(f"\nBenchmarking Agent: {agent_type}")
        for s in range(bench_cfg['num_seeds']):
            seed = 42 + s # Standard seed base
            run_seed(agent_type, env_name, seed, config, experiment_dir)

    print(f"\nBenchmark completed. Results saved in {experiment_dir}")

if __name__ == "__main__":
    run_benchmark()
