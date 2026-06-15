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


# Canonical KPI order, used across every table and plot in the paper:
# benefits (higher is better) -> mobility outcomes by increasing severity
# -> resource cost.
METRICS_TO_PLOT = [
    "reward", "capacity_avg", "reliability_pct",
    "ho_rate", "pp_rate", "hof_rate",
    "rlf_rate", "prep_rate", "res_reservation_pct",
]
NICE_NAMES = {
    "reward": "Reward / Decision",
    "capacity_avg": "Capacity (bps/Hz)",
    "reliability_pct": "Reliability (%)",
    "ho_rate": "HO Rate (/min)",
    "pp_rate": "PP Rate (/min)",
    "hof_rate": "HOF Rate (/min)",
    "rlf_rate": "RLF Rate (/min)",
    "prep_rate": "Preparations (/min)",
    "res_reservation_pct": "Res. Reservation (%)",
}


def _load_data(mapping):
    """Load every summary CSV whose file-key is in `mapping`, keyed by label.

    Returns a DataFrame indexed by the nice label, columns = metrics.
    """
    results_dir = "results/final_metrics"
    all_data = {}
    for f in glob.glob(os.path.join(results_dir, "*.csv")):
        key = os.path.basename(f).replace(".csv", "")
        # Handle specific naming like dqn_2k-ep_summary.
        if "qrdqn_2k-ep" in key:
            key = "qrdqn_summary"
        elif "dqn_2k-ep" in key:
            key = "dqn_summary"
        if key in mapping:
            series = load_summary(f)
            if series is not None:
                all_data[mapping[key]] = series
    return pd.DataFrame(all_data).transpose()


def _draw_bar_grid(data, agents, agent_color, agent_hatch, title, out_name,
                   legend_handles, bar_width=0.8, metrics=None,
                   grid=(3, 3), figsize=(22, 18)):
    """Grid of KPI bar charts for the given agent order.

    Every panel starts at zero (no broken axes); missing values (e.g. CMAB has
    no reward) leave a reserved-but-empty slot so the x-axis stays aligned
    across panels.

    Args:
        metrics: KPI keys to plot (defaults to all nine, in canonical order).
        grid: (nrows, ncols) of the subplot layout.
        figsize: overall figure size.
    """
    if metrics is None:
        metrics = METRICS_TO_PLOT
    fig, axes = plt.subplots(grid[0], grid[1], figsize=figsize)
    axes = axes.flatten()
    colors = [agent_color[a] for a in agents]
    hatches = [agent_hatch[a] for a in agents]
    x_pos = np.arange(len(agents))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if metric not in data.columns:
            ax.set_visible(False)
            continue
        series = data[metric].reindex(agents)
        bars = ax.bar(x_pos, series.values, width=bar_width,
                      color=colors, alpha=0.8, edgecolor='black')
        # Per-bar hatch (older matplotlib rejects a list for `hatch=`).
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents)
        ax.set_xlim(-0.6, len(agents) - 0.4)
        ax.set_title(NICE_NAMES[metric], fontsize=15, fontweight='bold', pad=10)
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

    # Hide any leftover panels (when metrics < grid cells).
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.legend(
        handles=legend_handles, loc='lower center', ncol=len(legend_handles),
        fontsize=15, frameon=True, bbox_to_anchor=(0.5, 0.012),
    )
    plt.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    plots_dir = "results/final_metrics/plots"
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")

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
    
    # Load the summaries and enforce the requested order. Keep ALL agents from
    # `desired_order` (filling missing rows with NaN) so the x-axis layout
    # stays consistent across every subplot, even when a CSV is missing.
    data = _load_data(mapping).reindex(desired_order)
    agents = desired_order
    # Group-coded colours, shared by the bar and radar charts.
    colors = [agent_color[a] for a in agents]

    # --- 2. CONSOLIDATED BAR SUBPLOTS ---
    # Figure-level legend: colour encodes the algorithm family, the hatched
    # swatch explains that paper-sourced bars are striped. CMAB is the only
    # literature-sourced (hatched) bar, so its grey-hatched swatch doubles as
    # the "published" key -- no separate hatch entry needed.
    legend_handles = [
        Patch(facecolor="#3182bd", edgecolor='black', alpha=0.8,
              label="LTM baseline (our simulator)"),
        Patch(facecolor="#969696", edgecolor='black', alpha=0.8, hatch='///',
              label="CMAB (published)"),
        Patch(facecolor="#de2d26", edgecolor='black', alpha=0.8,
              label="Learned RL agents (ours)"),
    ]
    print(f"Generating 3x3 bar plots for agents in order: {agents}")
    _draw_bar_grid(
        data, agents, agent_color, agent_hatch,
        "LTM-HO Comparative Analysis: RL Agents vs. State-of-the-Art Baselines",
        "master_bar_plots.png", legend_handles,
    )

    # --- 3. CONSOLIDATED RADAR CHART ---
    plots_dir = "results/final_metrics/plots"
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
    
    radar_path = os.path.join(plots_dir, "master_radial_plot.png")
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {radar_path}")

def generate_ltm_cmab():
    """2x4 KPI bar grid for ONLY the two prior-art baselines (LTM vs CMAB).

    Wide 2-row x 4-col layout (the 8 KPIs except reward) so it fits a single
    slide. Same colours as the full master plot: our LTM baseline (blue) and
    the published CMAB bandit (grey, hatched).
    """
    mapping = {"ltm_ours_summary": "LTM", "paper_ltm_cmab": "CMAB"}
    order = ["LTM", "CMAB"]
    agent_color = {"LTM": "#3182bd", "CMAB": "#969696"}
    agent_hatch = {"LTM": "", "CMAB": "///"}
    data = _load_data(mapping).reindex(order)

    legend_handles = [
        Patch(facecolor="#3182bd", edgecolor='black', alpha=0.8,
              label="LTM baseline (our simulator)"),
        Patch(facecolor="#969696", edgecolor='black', alpha=0.8, hatch='///',
              label="CMAB (published)"),
    ]
    # 2x4 layout (8 KPIs, reward dropped) so it sits nicely on one slide.
    metrics = ["capacity_avg", "reliability_pct", "ho_rate", "pp_rate",
               "hof_rate", "rlf_rate", "prep_rate", "res_reservation_pct"]
    print(f"Generating LTM-vs-CMAB 2x4 bar plots: {order}")
    _draw_bar_grid(
        data, order, agent_color, agent_hatch,
        "Prior-Art Baselines: LTM vs CMAB",
        "master_bar_plots_ltm_cmab.png", legend_handles, bar_width=0.5,
        metrics=metrics, grid=(2, 4), figsize=(24, 11),
    )


def generate_distributional(include_rn1q=True,
                            out_name="master_bar_plots_distributional.png"):
    """3x3 KPI bar grid for the SECTION-4 result: does the distribution help?

    With ``include_rn1q`` (default) the five models isolate the source of the
    gain (paper Table II + the two references): the LTM and CMAB baselines,
    scalar DQN, a single-quantile risk-neutral head (RN-1q, N=1 median) that
    reproduces DQN, and the full 25-quantile risk-neutral head (RN). The story:
    DQN ~ RN-1q < RN, all at equal reward -> the gain comes from the *full
    distribution*, not the loss. Set ``include_rn1q=False`` for the leaner
    four-model version (just DQN vs RN against the two references).

    Colours reuse the final master plot exactly (LTM blue, CMAB grey-hatched,
    DQN and RN in the red family), with RN-1q assigned a new, distinct colour
    (purple) since it is not part of the master roster.
    """
    mapping = {
        "ltm_ours_summary": "LTM",
        "paper_ltm_cmab": "CMAB",
        "dqn_summary": "DQN",
        "qrdqn_rn_1q_summary": "RN-1q",
        "qrdqn_riskneutral_summary": "RN",
    }
    order = ["LTM", "CMAB", "DQN", "RN-1q", "RN"]
    agent_color = {
        "LTM": "#3182bd",        # blue, our baseline (master)
        "CMAB": "#969696",       # grey, published (master)
        "DQN": "#fcae91",        # red, lightest (master)
        "RN-1q": "#984ea3",      # purple, NEW (not in the master roster)
        "RN": "#fb6a4a",         # red (master)
    }
    legend_by_agent = {
        "LTM": Patch(facecolor="#3182bd", edgecolor='black', alpha=0.8,
                     label="LTM baseline (our simulator)"),
        "CMAB": Patch(facecolor="#969696", edgecolor='black', alpha=0.8,
                      hatch='///', label="CMAB (published)"),
        "DQN": Patch(facecolor="#fcae91", edgecolor='black', alpha=0.8,
                     label="DQN"),
        "RN-1q": Patch(facecolor="#984ea3", edgecolor='black', alpha=0.8,
                       label="RN-1q (single quantile)"),
        "RN": Patch(facecolor="#fb6a4a", edgecolor='black', alpha=0.8,
                    label="RN (full distribution)"),
    }
    if not include_rn1q:
        order = [a for a in order if a != "RN-1q"]
        mapping = {k: v for k, v in mapping.items() if v != "RN-1q"}

    agent_hatch = {a: ("///" if a == "CMAB" else "") for a in order}
    data = _load_data(mapping).reindex(order)
    legend_handles = [legend_by_agent[a] for a in order]
    print(f"Generating 3x3 distributional bar plots: {order}")
    _draw_bar_grid(
        data, order, agent_color, agent_hatch,
        "Distributional Learning on the Handover Task: DQN vs QR-DQN",
        out_name, legend_handles, bar_width=0.7,
    )


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
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, "master_bar_plots_1col.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


if __name__ == "__main__":
    generate_plots()
    generate_ltm_cmab()
    generate_distributional()
    generate_distributional(include_rn1q=False,
                            out_name="master_bar_plots_distributional_no1q.png")
    generate_single_column_bars()
