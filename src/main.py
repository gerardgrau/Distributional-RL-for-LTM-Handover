import torch
import numpy as np
import os
import sys
import time
import csv
import shutil
import json
from tqdm import tqdm
from datetime import datetime
from typing import Any

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.distrl.utils.config import Config
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.dqn import DQNAgent
from src.distrl.agents.distributional.qrdqn import QRDQNAgent
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.utils.replay_buffer import ReplayBuffer
from src.distrl.utils.plot import plot_learning_curves, plot_efficiency, plot_quantiles
from src.distrl.utils.metrics import calculate_8_metrics
from src.distrl.utils.evaluation import run_evaluation

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
             pbar: tqdm, device: str = "cpu", save_results: bool = True) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if env_name != "ltm":
        raise ValueError(f"Only 'ltm' environment is supported. Received: {env_name}")
    
    env = LTMEnv(config=config)
    
    agent_config = config['agent']
    
    if agent_type.lower() == "dqn":
        agent = DQNAgent(agent_config, env.observation_space, env.action_space, device=device)
    elif agent_type.lower() == "qrdqn":
        agent = QRDQNAgent(agent_config, env.observation_space, env.action_space, device=device)
    elif agent_type.lower() == "ltm_baseline":
        agent = LTMBaselineAgent(config, env.observation_space, env.action_space, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    buffer = ReplayBuffer(agent_config['buffer_size'], env.observation_space.shape)
    
    # Save CSV logs in experiment_dir/train/
    if save_results:
        log_file = os.path.join(experiment_dir, "train", f"{agent_type}_{env_name.replace('/', '_')}_seed{seed}.csv")
        headers = ["episode", "reward", "loss", "steps", "wall_time", 
                   "capacity", "rlf_rate", "ho_rate", "pp_rate", 
                   "reliability", "prep_rate", "res_reservation", "hof_rate"]
        logger = CSVLogger(log_file, headers)
    
    num_episodes = agent_config.get('num_episodes', 20)
    epsilon = agent_config.get('epsilon_start', 1.0)
    eps_end = agent_config.get('epsilon_end', 0.05)
    eps_mult = agent_config.get('epsilon_mult', 0.99)
    batch_size = agent_config.get('batch_size', 64)
    train_freq = max(1, int(agent_config.get('train_freq', 1)))
    
    start_time = time.time()
    rewards_history = []
    global_step = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        episode_loss = []
        done = False
        last_info = {}
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            global_step += 1
            
            if len(buffer) > batch_size and global_step % train_freq == 0:
                metrics = agent.train_step(buffer.sample(batch_size, device=device))
                episode_loss.append(metrics['loss'])
            if done:
                last_info = info
        
        # Calculate LTM metrics at end of episode (Tracking during training)
        metrics_8 = calculate_8_metrics(
            mcs_history=last_info["metrics"]["mcs"],
            rlf_history=last_info["metrics"]["rlf"],
            ho_history=last_info["metrics"]["ho"],
            hof_history=last_info["metrics"]["hof"],
            pp_history=last_info["metrics"]["pp"],
            serving_history=last_info["metrics"]["serving"],
            pl3_history=last_info["metrics"]["pl3"],
            config=config
        )
        
        epsilon = max(eps_end, epsilon * eps_mult)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        if save_results:
            logger.log([
                ep + 1, episode_reward, avg_loss, env.t, time.time() - start_time,
                metrics_8["capacity_avg"],
                metrics_8["rlf_rate"],
                metrics_8["ho_rate"],
                metrics_8["pp_rate"],
                metrics_8["reliability_pct"],
                metrics_8["prep_rate"],
                metrics_8["res_reservation_pct"],
                metrics_8["hof_rate"]
            ])
        rewards_history.append(episode_reward)
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            "agent": agent_type,
            "seed": seed,
            "reward": f"{np.mean(rewards_history[-10:]):.1f}"
        })

    # SAVE MODEL FOR THIS SEED in experiment_dir/models/
    if save_results:
        model_path = os.path.join(experiment_dir, "models", f"{agent_type}_seed{seed}.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save(model_path)
        
    # --- FORMAL EVALUATION PROTOCOL ---
    # Evaluate on ALL users after training finishes
    run_evaluation(agent, config, experiment_dir, agent_type, seed, save_results=save_results)

    env.close()
    return float(np.mean(rewards_history[-10:]))

def run_benchmark():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--no_save", action="store_true", help="Do not save any results, logs, or plots")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device (cpu, cuda, xpu)")
    parser.add_argument("--description", type=str, default="benchmark", help="Short description for the benchmark")
    parser.add_argument("--agents", type=str, default="dqn,qrdqn", help="Comma-separated list of agents to run")
    args = parser.parse_args()

    Config.set_config_path(args.config)
    config = Config.get()
    
    device = args.device
    
    bench_cfg = config['benchmark']
    agent_types = [a.strip().lower() for a in args.agents.split(",")]
    
    # --- PERFORMANCE OPTIMIZATION: Unique Naming Convention ---
    # Format: bmk_YYYY-MM-DD_num_description
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    desc_slug = args.description.replace(" ", "-")
    results_dir = bench_cfg['results_dir']
    
    num = 1
    import glob
    while True:
        # Check if any directory exists with this date and number, regardless of description
        pattern = os.path.join(results_dir, f"bmk_{date_str}_{num}_*")
        if not glob.glob(pattern):
            break
        num += 1
    
    experiment_dir = os.path.join(results_dir, f"bmk_{date_str}_{num}_{desc_slug}")
    
    if not args.no_save:
        # Pre-create subdirectories (Restructured)
        os.makedirs(os.path.join(experiment_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "figures"), exist_ok=True)
        
        # Copy config file to experiment directory for provenance
        shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))
        
        print(f"=== Starting Independent Benchmark: {experiment_dir} ===")
    else:
        print("=== Starting Profiling Run (No artifacts will be saved) ===")
    
    bench_start_time = time.time()
    total_runs = len(agent_types) * bench_cfg['num_seeds']
    num_episodes = config['agent'].get('num_episodes', 20)
    total_eps_overall = total_runs * num_episodes
    
    with tqdm(total=total_eps_overall, desc="Benchmark Overall") as pbar:
        for agent_type in agent_types:
            best_reward = -np.inf
            best_seed_path = ""
            
            for s in range(bench_cfg['num_seeds']):
                seed = 42 + s
                final_reward = run_seed(agent_type, bench_cfg['env_type'], seed, config, experiment_dir,
                                        pbar, device=device, save_results=not args.no_save)
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_seed_path = os.path.join(experiment_dir, "models", f"{agent_type}_seed{seed}.pth")
            
            if not args.no_save:
                # Save "Best" for this specific run in models/
                if best_seed_path and os.path.exists(best_seed_path):
                    best_dst = os.path.join(experiment_dir, "models", f"{agent_type}_best.pth")
                    shutil.copy(best_seed_path, best_dst)
                else:
                    print(f"  -> No model file found for {agent_type} to mark as best.")

            # Generate Quantile Plot if it's QRDQN
            if agent_type.lower() == "qrdqn":
                if not args.no_save:
                    print(f"  -> Generating distributional insight plot for {agent_type}...")
                    # We need to reload the best agent to sample a state
                    env = LTMEnv(config=config)
                    agent = QRDQNAgent(config['agent'], env.observation_space, env.action_space)

                    # Determine path to best model
                    best_dst = os.path.join(experiment_dir, "models", f"{agent_type}_best.pth")
                    if os.path.exists(best_dst):
                        agent.load(best_dst)

                        state, _ = env.reset()
                        # Sample a few steps to get a meaningful state
                        for _ in range(50):
                            state, _, d, _, _ = env.step(agent.select_action(state, 0))
                            if d: break

                        q_save_path = os.path.join(experiment_dir, "figures", "quantile_distribution.png")
                        plot_quantiles(agent, state, save_path=q_save_path)
                    
                    env.close()

    if not args.no_save:
        # AUTO-PLOT in figures/
        print("\nGenerating performance plots...")
        fig_dir = os.path.join(experiment_dir, "figures")
        csv_dir = os.path.join(experiment_dir, "train")
        plot_learning_curves(csv_dir, save_path=os.path.join(fig_dir, "learning_curves.png"))
        plot_efficiency(csv_dir, metric="reward", save_path=os.path.join(fig_dir, "reward_vs_time.png"))
        plot_efficiency(csv_dir, metric="loss", save_path=os.path.join(fig_dir, "loss_vs_time.png"))

        import json
        metadata = {
            "timestamp": timestamp,
            "total_execution_time_seconds": time.time() - bench_start_time,
            "device": device,
            "config": config
        }
        
        # We always save metadata for benchmarking, even if logs/models are disabled
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"\nBenchmark completed. All artifacts saved in {experiment_dir}")
    else:
        print("\nProfiling run completed. No artifacts saved.")

if __name__ == "__main__":
    run_benchmark()
