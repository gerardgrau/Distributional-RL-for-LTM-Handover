import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_summary(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns] 
    df = df.set_index('metric')
    df.index = [i.strip() for i in df.index]
    return df['mean']

def generate_plots():
    results_dir = "results/final_metrics"
    
    # --- 1. Define the Fixed Order and Mapping ---
    # File Key -> Nice Label
    mapping = {
        "paper_ltm": "LTM",
        "paper_lmmse": "LMMSE",
        "paper_ltm_cmab": "LTM-CMAB",
        "paper_lmmse_cmab": "LMMSE-CMAB",
        "legacy_baseline_summary": "Baseline (Legacy)",
        "baseline_summary": "Baseline (Ours)",
        "dqn_summary": "DQN (Ours)",
        "qrdqn_summary": "QR-DQN (Ours)"
    }
    
    desired_order = [
        "LTM", "LMMSE", "LTM-CMAB", "LMMSE-CMAB", 
        "Baseline (Legacy)", "Baseline (Ours)", "DQN (Ours)", "QR-DQN (Ours)"
    ]
    
    all_data = {}
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    for f in csv_files:
        file_key = os.path.basename(f).replace(".csv", "")
        # Handle specific naming like dqn_2k-ep_summary
        if "qrdqn_2k-ep" in file_key:
            file_key = "qrdqn_summary"
        elif "dqn_2k-ep" in file_key:
            file_key = "dqn_summary"
            
        if file_key in mapping:
            name = mapping[file_key]
            series = load_summary(f)
            if series is not None:
                all_data[name] = series

    # Convert to DataFrame and REINDEX to enforce the requested order
    data = pd.DataFrame(all_data).transpose()
    # Filter only available agents that are in our desired order
    data = data.reindex([a for a in desired_order if a in data.index])
    agents = data.index.tolist()

    metrics_to_plot = [
        "ho_rate", "hof_rate", "pp_rate",
        "capacity_avg", "rlf_rate", "reliability_pct",
        "prep_rate", "res_reservation_pct", "reward"
    ]

    nice_names = {
        "reward": "Total Reward",
        "capacity_avg": "Capacity (bps/Hz)",
        "rlf_rate": "RLF Rate (/min)",
        "hof_rate": "HOF Rate (/min)",
        "ho_rate": "HO Rate (/min)",
        "pp_rate": "PP Rate (/min)",
        "reliability_pct": "Reliability (%)",
        "prep_rate": "Preparations (/min)",
        "res_reservation_pct": "Res. Reservation (%)"
    }

    # Consistent color palette for the agents
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))

    # --- 2. CONSOLIDATED BAR SUBPLOTS ---
    print(f"Generating 3x3 bar plots for agents in order: {agents}")
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        if metric in data.columns:
            valid_series = data[metric].dropna()
            current_agents = valid_series.index
            current_colors = [colors[agents.index(a)] for a in current_agents]
            
            bars = ax.bar(current_agents, valid_series, color=current_colors, alpha=0.8, edgecolor='black')
            
            ax.set_title(nice_names[metric], fontsize=15, fontweight='bold', pad=10)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.tick_params(axis='x', rotation=35, labelsize=9)
            
            for bar in bars:
                yval = bar.get_height()
                fmt = '.3f' if yval < 1 else '.2f'
                ax.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.01), f'{yval:{fmt}}', 
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.set_visible(False)

    plt.suptitle("LTM-HO Comparative Analysis: RL Agents vs. State-of-the-Art Baselines", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    bar_path = os.path.join(results_dir, "master_bar_plots.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"  Saved {bar_path}")

    # --- 3. CONSOLIDATED RADAR CHART ---
    print("Generating master radial chart...")
    radar_labels = ["Capacity", "Reliability", "RLF (Inv)", "HOF (Inv)", "HO Stability", "PP Stability"]
    radar_df = pd.DataFrame(index=agents, columns=radar_labels)
    
    for agent in agents:
        radar_df.loc[agent, "Capacity"] = data.loc[agent, "capacity_avg"]
        radar_df.loc[agent, "Reliability"] = data.loc[agent, "reliability_pct"]
        radar_df.loc[agent, "RLF (Inv)"] = 1.0 / (1.0 + data.loc[agent, "rlf_rate"])
        radar_df.loc[agent, "HOF (Inv)"] = 1.0 / (1.0 + data.loc[agent, "hof_rate"])
        radar_df.loc[agent, "HO Stability"] = 1.0 / (1.0 + data.loc[agent, "ho_rate"] / 20.0)
        radar_df.loc[agent, "PP Stability"] = 1.0 / (1.0 + data.loc[agent, "pp_rate"])

    for col in radar_df.columns:
        c_min, c_max = radar_df[col].min(), radar_df[col].max()
        if c_max > c_min:
            radar_df[col] = (radar_df[col] - c_min) / (c_max - c_min) * 0.7 + 0.3
        else:
            radar_df[col] = 1.0

    angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    for i, agent in enumerate(agents):
        values = radar_df.loc[agent].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=3, label=agent, alpha=0.9)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=13, fontweight='bold')
    
    plt.title("Master Performance Profile: Global Comparison", fontsize=20, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)
    
    radar_path = os.path.join(results_dir, "master_radial_plot.png")
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {radar_path}")

if __name__ == "__main__":
    generate_plots()
