import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import glob
from typing import Optional, Any

def plot_learning_curves(results_dir: str, save_path: Optional[str] = None):
    """
    Parses all CSV files in results_dir, aggregates by agent, 
    and plots Reward vs Episode with confidence intervals.
    """
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        print(f"No CSV files found in {results_dir}")
        return

    data_frames = []
    for f in files:
        df = pd.read_csv(f)
        filename = os.path.basename(f)
        agent_type = filename.split('_')[0]
        df['agent'] = agent_type
        data_frames.append(df)

    all_data = pd.concat(data_frames, ignore_index=True)
    agents = all_data['agent'].unique()

    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        agent_data = all_data[all_data['agent'] == agent]
        grouped = agent_data.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
        
        plt.plot(grouped['episode'], grouped['mean'], label=f'{agent.upper()}')
        plt.fill_between(
            grouped['episode'], 
            grouped['mean'] - grouped['std'], 
            grouped['mean'] + grouped['std'], 
            alpha=0.2
        )

    plt.title("Learning Curves: Reward vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_efficiency(results_dir: str, metric: str = "reward", save_path: Optional[str] = None):
    """
    Plots a metric (reward or loss) against wall-clock time to compare efficiency.
    """
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    all_data = pd.concat([pd.read_csv(f).assign(agent=os.path.basename(f).split('_')[0]) for f in files])
    agents = all_data['agent'].unique()

    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        agent_data = all_data[all_data['agent'] == agent]
        for seed_file in glob.glob(os.path.join(results_dir, f"{agent}_*.csv")):
            df = pd.read_csv(seed_file)
            plt.plot(df['wall_time'], df[metric], alpha=0.3, color='gray' if agent == 'dqn' else 'blue')
        
        grouped = agent_data.groupby('episode')[['wall_time', metric]].mean().reset_index()
        plt.plot(grouped['wall_time'], grouped[metric], label=f'{agent.upper()}', linewidth=2)

    plt.title(f"Efficiency: {metric.capitalize()} vs Wall-clock Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_quantiles(agent: Any, state: np.ndarray, action_names: Optional[list[str]] = None, save_path: Optional[str] = None):
    """
    Plots the return distribution (quantiles) for each action in a given state.
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        quantiles = agent.q_net(state_tensor).cpu().numpy()[0]
    
    num_actions = quantiles.shape[0]
    plt.figure(figsize=(12, 6))
    
    for i in range(num_actions):
        label = action_names[i] if action_names else f"Action {i}"
        plt.hist(quantiles[i], bins=30, alpha=0.5, label=label, density=True)
        
    plt.title("Action-Value Distributions (Quantiles)")
    plt.xlabel("Return (Q-Value)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Quantile plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, help="Directory with benchmark CSVs")
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "quantiles"])
    args = parser.parse_args()

    results_root = "results"
    
    if args.mode == "benchmark":
        target_dir = args.results_dir
        if not target_dir:
            subdirs = [os.path.join(results_root, d) for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
            if not subdirs:
                print("No benchmark results found.")
                exit()
            target_dir = max(subdirs, key=os.path.getmtime)
            
        print(f"Generating benchmark plots for: {target_dir}")
        plot_learning_curves(target_dir, save_path=os.path.join(target_dir, "learning_curves.png"))
        plot_efficiency(target_dir, metric="reward", save_path=os.path.join(target_dir, "reward_vs_time.png"))
        plot_efficiency(target_dir, metric="loss", save_path=os.path.join(target_dir, "loss_vs_time.png"))
    
    elif args.mode == "quantiles":
        print("Quantile plotting requires an active agent instance.")
