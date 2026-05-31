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
        "qrdqn_riskneutral_summary": "QR-DQN-RN (Ours)",
        "qrdqn_ra_k1_summary": "QR-DQN-RA-1q (Ours)",
        "qrdqn_summary": "QR-DQN-RA (Ours)"
    }

    desired_order = [
        "LTM", "LMMSE", "LTM-CMAB", "LMMSE-CMAB",
        "Baseline (Legacy)", "Baseline (Ours)", "DQN (Ours)",
        "QR-DQN-RN (Ours)", "QR-DQN-RA-1q (Ours)", "QR-DQN-RA (Ours)"
    ]

    # Paper references get a striped hatch so they are visually distinct from
    # the measurements produced by this project.
    paper_refs = {"LTM", "LMMSE", "LTM-CMAB", "LMMSE-CMAB"}
    
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

    # Convert to DataFrame and REINDEX to enforce the requested order.
    # Keep ALL agents from `desired_order` (filling missing rows with NaN)
    # so the 8-slot x-axis layout stays consistent across every subplot,
    # even when a CSV is missing for some agent.
    data = pd.DataFrame(all_data).transpose()
    data = data.reindex(desired_order)
    agents = desired_order

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

    # Broken-y-axis configuration: (ymin, ymax). NaN ymax = autoscale upper.
    broken_ylim = {
        "reward": (800.0, None),
        "capacity_avg": (2.0, None),
        "reliability_pct": (80.0, 100.0),
    }

    def _draw_axis_break(ax_) -> None:
        """Flag a broken (non-zero-anchored) y-axis with a visible indicator.

        Draws the universal "≈" symbol at the bottom-left of the axis and a
        thicker double-slash zigzag on the y-axis spine just above the
        bottom tick — together these are unambiguous "this axis does not
        start at zero".
        """
        # ≈ symbol just below the axis bottom-left
        ax_.text(
            -0.06, -0.04, "≈",
            transform=ax_.transAxes,
            fontsize=20, fontweight='bold',
            ha='center', va='center',
            clip_on=False,
        )
        # Two short diagonal segments on the y-axis spine for the zigzag
        d_x = 0.025
        d_y = 0.022
        kw = dict(transform=ax_.transAxes, color='k',
                  clip_on=False, linewidth=2.0)
        ax_.plot([-d_x, d_x], [0.005, 0.005 + 1.6 * d_y], **kw)
        ax_.plot([-d_x, d_x], [0.005 + 2 * d_y, 0.005 + 3.6 * d_y], **kw)

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        if metric not in data.columns:
            ax.set_visible(False)
            continue

        # Keep all 8 agents on the x-axis (NaN -> invisible bar but reserved
        # slot). Use numeric positions so matplotlib doesn't drop empty
        # categories and the bar widths stay consistent across panels.
        series = data[metric].reindex(agents)
        current_colors = [colors[i] for i in range(len(agents))]
        current_hatches = ['///' if a in paper_refs else '' for a in agents]
        x_pos = np.arange(len(agents))

        bars = ax.bar(
            x_pos, series.values,
            color=current_colors, alpha=0.8,
            edgecolor='black',
        )
        # Apply hatches per-bar — older matplotlib versions don't accept
        # `hatch=` as a list, so set it on each bar individually.
        for bar, hatch in zip(bars, current_hatches):
            if hatch:
                bar.set_hatch(hatch)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents)
        ax.set_xlim(-0.6, len(agents) - 0.4)
        ax.set_title(nice_names[metric], fontsize=15, fontweight='bold', pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', rotation=35, labelsize=9)

        for bar, yval in zip(bars, series.values):
            if yval is None or pd.isna(yval):
                continue
            fmt = '.3f' if yval < 1 else '.2f'
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + abs(yval) * 0.01,
                f'{yval:{fmt}}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

        # Broken y-axis: clip the visible range and mark the break visually
        if metric in broken_ylim:
            ymin, ymax = broken_ylim[metric]
            ax.set_ylim(bottom=ymin, top=ymax)
            _draw_axis_break(ax)

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
