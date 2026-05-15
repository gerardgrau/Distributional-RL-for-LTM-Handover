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
    and plots Reward vs Episode with confidence intervals and a 50-episode moving average.
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
        
        # Calculate Moving Average (Centered)
        window = 51
        grouped['moving_avg'] = grouped['mean'].rolling(window=window, min_periods=1, center=True).mean()
        
        # Plot Raw Mean (Transparent)
        line = plt.plot(grouped['episode'], grouped['mean'], alpha=0.3)[0]
        color = line.get_color()
        
        # Plot Moving Average (Solid)
        plt.plot(grouped['episode'], grouped['moving_avg'], color=color, label=f'{agent.upper()} (MA {window})', linewidth=2)
        
        # Plot Confidence Interval
        plt.fill_between(
            grouped['episode'], 
            grouped['mean'] - grouped['std'], 
            np.maximum(0, grouped['mean'] + grouped['std']),  # Ensure upper bound is not negative
            color=color,
            alpha=0.1
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
    Plots a metric (reward or loss) against wall-clock time with a 50-unit moving average.
    """
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    all_data = pd.concat([pd.read_csv(f).assign(agent=os.path.basename(f).split('_')[0]) for f in files])
    agents = all_data['agent'].unique()

    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        agent_data = all_data[all_data['agent'] == agent]
        
        # Individual seeds (Very transparent)
        for seed_file in glob.glob(os.path.join(results_dir, f"{agent}_*.csv")):
            df = pd.read_csv(seed_file)
            plt.plot(df['wall_time'], df[metric], alpha=0.1, color='gray' if agent == 'dqn' else 'blue')
        
        # Aggregate trend
        grouped = agent_data.groupby('episode')[['wall_time', metric]].mean().reset_index()
        
        # Calculate Moving Average (Centered)
        window = 51
        grouped['moving_avg'] = grouped[metric].rolling(window=window, min_periods=1, center=True).mean()
        
        # Plot Aggregate Trend (Transparent)
        line = plt.plot(grouped['wall_time'], grouped[metric], alpha=0.3)[0]
        color = line.get_color()
        
        # Plot Moving Average (Solid)
        plt.plot(grouped['wall_time'], grouped['moving_avg'], color=color, label=f'{agent.upper()} (MA {window})', linewidth=2)

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

def plot_metrics_grid(
    results_dir: str,
    save_path: Optional[str] = None,
    window: int = 51,
) -> None:
    """3x3 grid of per-episode learning curves across all 8 scientific
    metrics plus the training reward, aggregated across seeds per agent.

    Layout (left-to-right, top-to-bottom):
        reward, capacity, reliability,
        ho_rate, pp_rate, hof_rate,
        rlf_rate, prep_rate, res_reservation
    """
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        print(f"No CSV files found in {results_dir}")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df['agent'] = os.path.basename(f).split('_')[0]
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)
    agents = all_data['agent'].unique()

    panels = [
        ("reward",          "Episode reward",        "reward (sum of per-step Ainna)"),
        ("capacity",        "Capacity",              "bps/Hz"),
        ("reliability",     "Reliability",           "%"),
        ("ho_rate",         "HO rate",               "events/min"),
        ("pp_rate",         "Ping-pong rate",        "events/min"),
        ("hof_rate",        "HOF rate",              "events/min"),
        ("rlf_rate",        "RLF rate",              "events/min"),
        ("prep_rate",       "Cell preparation rate", "preps/min"),
        ("res_reservation", "Resource reservation",  "%"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    fig.suptitle("Training metrics by episode", fontsize=14, fontweight='bold')

    for ax, (col, title, ylabel) in zip(axes.flat, panels):
        if col not in all_data.columns:
            ax.set_title(f"{title} (missing)")
            ax.set_axis_off()
            continue
        for agent in agents:
            sub = all_data[all_data['agent'] == agent]
            grouped = sub.groupby('episode')[col].agg(['mean', 'std']).reset_index()
            grouped['ma'] = grouped['mean'].rolling(
                window=window, min_periods=1, center=True
            ).mean()

            raw_line, = ax.plot(grouped['episode'], grouped['mean'], alpha=0.2)
            color = raw_line.get_color()
            ax.plot(
                grouped['episode'], grouped['ma'],
                color=color, linewidth=2.0,
                label=f"{agent.upper()} (MA {window})",
            )
            if grouped['std'].notna().any():
                ax.fill_between(
                    grouped['episode'],
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=color, alpha=0.1,
                )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=':', alpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel("Episode")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc='lower center',
            ncol=max(1, len(handles)), bbox_to_anchor=(0.5, -0.01),
            fontsize='medium',
        )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"Metrics grid saved to {save_path}")
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

    results_root = "results/benchmarks"
    
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
        plot_metrics_grid(target_dir, save_path=os.path.join(target_dir, "metrics_grid.png"))
    
    elif args.mode == "quantiles":
        print("Quantile plotting requires an active agent instance.")
