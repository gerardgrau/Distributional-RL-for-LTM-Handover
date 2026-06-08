import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from matplotlib.patches import Patch

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
    # File Key -> Nice Label. The roster mirrors the paper's final table
    # (tab:final): the four learned agents, then the two references. "LTM" is
    # our own hand-tuned baseline run in our simulator (the paper's "LTM"
    # column); "CMAB" is the published contextual bandit of [cmab_ho].
    mapping = {
        "dqn_summary": "DQN",
        "qrdqn_riskneutral_summary": "RN",
        "qrdqn_softstep_summary": "Soft-step",
        "qrdqn_summary": "RA-1q",
        "ltm_ours_summary": "LTM",
        "paper_ltm_cmab": "CMAB",
    }

    # Colour encodes the algorithm family; hatch encodes the data source. The
    # four learned agents share a red family, lightest (DQN) to darkest (RA-1q),
    # so the risk gradient reads off the shade. LTM is our own baseline measured
    # in our simulator (blue, solid); CMAB is reproduced from the literature
    # (grey, hatched), and its figures come from a different simulator.
    # Order: references first (prior art), then the learned agents by increasing
    # sophistication, so the eye ends on RA-1q (the risk-aware contribution).
    desired_order = [
        "LTM",                               # our LTM baseline (blue)
        "CMAB",                              # published bandit (grey)
        "DQN", "RN", "Soft-step", "RA-1q",   # learned agents (red, light->dark)
    ]

    agent_color = {
        "DQN": "#fcae91",        # red, lightest
        "RN": "#fb6a4a",
        "Soft-step": "#de2d26",
        "RA-1q": "#a50f15",      # red, darkest (most risk-averse)
        "LTM": "#3182bd",        # blue, our baseline
        "CMAB": "#969696",       # grey, published
    }
    # Literature-sourced bars get a diagonal hatch; our own measurements are
    # solid. Only CMAB is external now -- the LTM bar is our own simulator.
    paper_sourced = {"CMAB"}
    agent_hatch = {a: ("///" if a in paper_sourced else "") for a in desired_order}
    
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
    # so the x-axis layout stays consistent across every subplot,
    # even when a CSV is missing for some agent.
    data = pd.DataFrame(all_data).transpose()
    data = data.reindex(desired_order)
    agents = desired_order

    # Canonical KPI order, used across every table and plot in the paper:
    # benefits (higher is better) -> mobility outcomes by increasing severity
    # -> resource cost. See notes/paper-revision-todo.md.
    metrics_to_plot = [
        "reward", "capacity_avg", "reliability_pct",
        "ho_rate", "pp_rate", "hof_rate",
        "rlf_rate", "prep_rate", "res_reservation_pct"
    ]

    nice_names = {
        "reward": "Reward / Decision",
        "capacity_avg": "Capacity (bps/Hz)",
        "rlf_rate": "RLF Rate (/min)",
        "hof_rate": "HOF Rate (/min)",
        "ho_rate": "HO Rate (/min)",
        "pp_rate": "PP Rate (/min)",
        "reliability_pct": "Reliability (%)",
        "prep_rate": "Preparations (/min)",
        "res_reservation_pct": "Res. Reservation (%)"
    }

    # Group-coded colours, shared by the bar and radar charts.
    colors = [agent_color[a] for a in agents]

    # --- 2. CONSOLIDATED BAR SUBPLOTS ---
    print(f"Generating 3x3 bar plots for agents in order: {agents}")
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    axes = axes.flatten()

    # Every panel starts at zero -- no broken axes. (The reward row in the
    # final_metrics CSVs is the mean reward PER DECISION, not the total.)
    broken_ylim: dict[str, tuple[float, float | None]] = {}

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

    # Per-agent colours / hatches and uniform x-positions (computed once;
    # identical across every subplot). Colour already separates the groups, so
    # no extra spacing between them is needed.
    current_colors = [agent_color[a] for a in agents]
    current_hatches = [agent_hatch[a] for a in agents]
    x_pos = np.arange(len(agents))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        if metric not in data.columns:
            ax.set_visible(False)
            continue

        # Keep all agents on the x-axis (NaN -> invisible bar but reserved
        # slot). Use numeric positions so matplotlib doesn't drop empty
        # categories and the bar widths stay consistent across panels.
        series = data[metric].reindex(agents)

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

        # (The reward panel's data-source note lives in the figure caption,
        # not inside the axes.)

        # Broken y-axis: clip the visible range and mark the break visually
        if metric in broken_ylim:
            ymin, ymax = broken_ylim[metric]
            ax.set_ylim(bottom=ymin, top=ymax)
            _draw_axis_break(ax)

    # Figure-level legend: colour encodes the algorithm family, the hatched
    # swatch explains that paper-sourced bars are striped.
    # CMAB is the only literature-sourced (hatched) bar, so its grey-hatched
    # swatch doubles as the "published" key -- no separate hatch entry needed.
    legend_handles = [
        Patch(facecolor="#3182bd", edgecolor='black', alpha=0.8,
              label="LTM baseline (our simulator)"),
        Patch(facecolor="#969696", edgecolor='black', alpha=0.8, hatch='///',
              label="CMAB (published)"),
        Patch(facecolor="#de2d26", edgecolor='black', alpha=0.8,
              label="Learned RL agents (ours)"),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=3,
        fontsize=15, frameon=True, bbox_to_anchor=(0.5, 0.012),
    )

    plt.suptitle("LTM-HO Comparative Analysis: RL Agents vs. State-of-the-Art Baselines", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    bar_path = os.path.join(results_dir, "master_bar_plots.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"  Saved {bar_path}")

    # --- 3. CONSOLIDATED RADAR CHART ---
    print("Generating master radial chart...")
    # Canonical order (radar omits reward / prep / reservation):
    # benefits -> mobility outcomes by increasing severity.
    radar_labels = ["Capacity", "Reliability", "HO Stability", "PP Stability", "HOF (Inv)", "RLF (Inv)"]
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

def generate_single_column_bars():
    """Column-width variant of the master bar plot (single IEEE text column).

    Re-tuned for ~3.4 in print width: a 5x2 portrait grid rendered at near
    print size (so fonts stay legible under `width=\\columnwidth`), no per-bar
    value labels (exact values live in tab:final), and a compact legend placed
    in the spare 10th panel. All panels start at zero.
    """
    results_dir = "results/final_metrics"
    mapping = {
        "ltm_ours_summary": "LTM",
        "paper_ltm_cmab": "CMAB",
        "dqn_summary": "DQN",
        "qrdqn_riskneutral_summary": "RN",
        "qrdqn_softstep_summary": "Soft-step",
        "qrdqn_summary": "RA-1q",
    }
    order = ["LTM", "CMAB", "DQN", "RN", "Soft-step", "RA-1q"]
    color = {"DQN": "#fcae91", "RN": "#fb6a4a", "Soft-step": "#de2d26",
             "RA-1q": "#a50f15", "LTM": "#3182bd", "CMAB": "#969696"}
    hatches = ["///" if a == "CMAB" else "" for a in order]

    all_data = {}
    for f in glob.glob(os.path.join(results_dir, "*.csv")):
        key = os.path.basename(f).replace(".csv", "")
        if key in mapping:
            s = load_summary(f)
            if s is not None:
                all_data[mapping[key]] = s
    data = pd.DataFrame(all_data).transpose().reindex(order)

    metrics = ["reward", "capacity_avg", "reliability_pct",
               "ho_rate", "pp_rate", "hof_rate",
               "rlf_rate", "prep_rate", "res_reservation_pct"]
    nice = {
        "reward": "Reward / Decision", "capacity_avg": "Capacity (bps/Hz)",
        "reliability_pct": "Reliability (%)", "ho_rate": "HO (/min)",
        "pp_rate": "PP (/min)", "hof_rate": "HOF (/min)",
        "rlf_rate": "RLF (/min)", "prep_rate": "Preparations (/min)",
        "res_reservation_pct": "Res. Reservation (%)",
    }

    x = np.arange(len(order))
    colors = [color[a] for a in order]

    # Rendered at ~print size so width=\columnwidth needs no down-scaling.
    fig, axes = plt.subplots(5, 2, figsize=(3.5, 5.4))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = data[metric].reindex(order).values
        bars = ax.bar(x, vals, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        ax.set_title(nice[metric], fontsize=6.5, fontweight="bold", pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5, pad=1)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.4)
        ax.set_xlim(-0.6, len(order) - 0.4)

    # Compact legend in the spare 10th panel.
    legend_handles = [
        Patch(facecolor="#3182bd", edgecolor="black", alpha=0.85,
              label="LTM baseline (our simulator)"),
        Patch(facecolor="#969696", edgecolor="black", alpha=0.85, hatch="///",
              label="CMAB (published)"),
        Patch(facecolor="#de2d26", edgecolor="black", alpha=0.85,
              label="Learned RL agents (ours)"),
    ]
    axes[9].axis("off")
    axes[9].legend(handles=legend_handles, loc="center", fontsize=6.0,
                   frameon=True, handlelength=1.4, borderpad=0.6)

    fig.tight_layout(h_pad=0.3, w_pad=0.8)
    out = os.path.join(results_dir, "master_bar_plots_1col.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


if __name__ == "__main__":
    generate_plots()
    generate_single_column_bars()
