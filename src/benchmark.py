import torch
import numpy as np
import os
import sys
import time
import csv
import shutil
import subprocess
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

def run_seed(agent_type: str, env_name: str, seed: int, config: dict, experiment_dir: str, 
             run_idx: int, total_runs: int, bench_start_time: float) -> float:
    print(f"  -> Starting Run {run_idx+1}/{total_runs}: {agent_type} (Seed {seed})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if env_name == "ltm":
        env = LTMEnv(config=config)
    else:
        import gymnasium as gym
        env = gym.make(env_name)
    
    agent_config = config['agent']
    device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    
    if agent_type.lower() == "dqn":
        agent = DQNAgent(agent_config, env.observation_space, env.action_space, device=device)
    elif agent_type.lower() == "qrdqn":
        agent = QRDQNAgent(agent_config, env.observation_space, env.action_space, device=device)
    
    buffer = ReplayBuffer(agent_config['buffer_size'], env.observation_space.shape)
    
    log_file = os.path.join(experiment_dir, f"{agent_type}_{env_name.replace('/', '_')}_seed{seed}.csv")
    headers = ["episode", "reward", "loss", "steps", "wall_time"]
    logger = CSVLogger(log_file, headers)
    
    num_episodes = agent_config.get('num_episodes', 20)
    epsilon = agent_config.get('epsilon_start', 1.0)
    eps_end = agent_config.get('epsilon_end', 0.05)
    eps_mult = agent_config.get('epsilon_mult', 0.99)
    batch_size = agent_config.get('batch_size', 64)
    
    start_time = time.time()
    rewards_history = []
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if len(buffer) > batch_size:
                metrics = agent.train_step(buffer.sample(batch_size, device=device))
                episode_loss.append(metrics['loss'])
        
        # Per-episode epsilon decay
        epsilon = max(eps_end, epsilon * eps_mult)
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        logger.log([ep + 1, episode_reward, avg_loss, env.t, time.time() - start_time])
        rewards_history.append(episode_reward)
        
        # Log progress
        if (ep+1) % 100 == 0 or (ep+1 <= 50 and (ep+1) % 10 == 0):
            total_elapsed = time.time() - bench_start_time
            total_eps_done = (run_idx * num_episodes) + (ep + 1)
            total_eps_overall = total_runs * num_episodes

            avg_time_per_ep = total_elapsed / total_eps_done
            remaining_eps = total_eps_overall - total_eps_done
            eta_seconds = avg_time_per_ep * remaining_eps

            # Format ETA as HH:MM:SS
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            print(f"    Run {run_idx+1}/{total_runs} | Ep {ep+1}/{num_episodes} | Avg Reward: {np.mean(rewards_history[-10:]):.2f} | Time: {total_elapsed:.1f}s | ETA: {eta_str}")


    # SAVE MODEL FOR THIS SEED
    model_path = os.path.join(experiment_dir, f"{agent_type}_seed{seed}.pth")
    agent.save(model_path)
    env.close()
    return float(np.mean(rewards_history[-10:]))

def run_benchmark():
    config = Config.get()
    bench_cfg = config['benchmark']
    agent_types = ["dqn", "qrdqn"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(bench_cfg['results_dir'], f"benchmark_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"=== Starting Independent Benchmark: {experiment_dir} ===")
    
    bench_start_time = time.time()
    total_runs = len(agent_types) * bench_cfg['num_seeds']
    run_idx = 0
    
    for agent_type in agent_types:
        print(f"\nBenchmarking Agent: {agent_type}")
        best_reward = -np.inf
        best_seed_path = ""
        
        for s in range(bench_cfg['num_seeds']):
            seed = 42 + s
            final_reward = run_seed(agent_type, bench_cfg['env_type'], seed, config, experiment_dir,
                                    run_idx, total_runs, bench_start_time)
            run_idx += 1
            if final_reward > best_reward:
                best_reward = final_reward
                best_seed_path = os.path.join(experiment_dir, f"{agent_type}_seed{seed}.pth")
        
        # Save "Best" for this specific run
        shutil.copy(best_seed_path, os.path.join(experiment_dir, f"{agent_type}_best.pth"))

    # AUTO-PLOT
    print("\nGenerating performance plots...")
    subprocess.run([sys.executable, "src/distrl/utils/plot.py", "--results_dir", experiment_dir], env=os.environ.update({"PYTHONPATH": os.path.abspath("src")}))

    print(f"\nBenchmark completed. All artifacts saved in {experiment_dir}")

if __name__ == "__main__":
    run_benchmark()
