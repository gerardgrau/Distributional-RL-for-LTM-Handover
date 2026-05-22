"""High-resolution parity check: LTMBaselineAgent in LTMEnv vs paper LTM.

Runs the gym env in high-res callback mode (one yield per simulation
tick) for `--ue_number` UEs, aggregates the 8 LTM metrics, and writes
the result to `results/final_metrics/baseline_summary.csv`. Seeds match
`legacy_simulation.py` so the run is bit-comparable.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.config import Config
from src.distrl.utils.metrics import calculate_8_metrics


METRIC_ORDER = [
    "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
    "reliability_pct", "prep_rate", "res_reservation_pct", "reward",
]
# Diagnostic-only fields surfaced during the run but excluded from the saved
# CSV so its schema matches `legacy_baseline_summary.csv`.
EXTRA_METRICS = ["total_steps", "total_minutes"]


def verify_parity(ue_count: int) -> None:
    Config.set_config_path("configs/config.yaml")
    config = Config.get()
    config["simulation"]["ue_number"] = ue_count

    env = LTMEnv(config=config)
    agent = LTMBaselineAgent(config, env.observation_space, env.action_space)

    all_metrics = []
    print(f"Running 10ms high-res parity verification on {ue_count} UEs...")

    start_time = time.time()
    for _ in tqdm(range(ue_count), desc="Simulating UEs"):
        filename = env.files[env.current_ue_idx % len(env.files)]
        numeric_id = int(re.search(r"\d+", os.path.basename(filename)).group())
        np.random.seed(42 + numeric_id)

        _, _ = env.reset()
        agent.reset()
        done = False
        info: dict = {}
        episode_reward = 0.0
        while not done:
            _, r, done, _, info = env.step(
                0, high_res_callback=agent.select_action
            )
            episode_reward += float(r)

        if "metrics" in info:
            m = calculate_8_metrics(
                info["metrics"]["mcs"],
                info["metrics"]["rlf"],
                info["metrics"]["ho"],
                info["metrics"]["hof"],
                info["metrics"]["pp"],
                info["metrics"]["serving"],
                info["metrics"]["pl3"],
                config,
                reserved_history=info["metrics"].get("reserved"),
            )
            m["reward"] = episode_reward
            all_metrics.append(m)

    elapsed = time.time() - start_time

    summary: dict[str, list[float]] = {}
    for m in all_metrics:
        for k, v in m.items():
            summary.setdefault(k, []).append(v)

    print(f"\n--- AGGREGATED METRICS (10ms High-Res BASELINE) ---")
    rows = []
    for k in METRIC_ORDER:
        if k not in summary:
            continue
        mean_val = float(np.mean(summary[k]))
        std_val = float(np.std(summary[k]))
        print(f"{k:20}: {mean_val:10.4f} (std: {std_val:.4f})")
        rows.append({"metric": k, "mean": mean_val, "std": std_val})

    for k in EXTRA_METRICS:
        if k not in summary:
            continue
        print(f"{k:20}: {float(np.mean(summary[k])):10.4f} (std: {float(np.std(summary[k])):.4f})")

    out_path = "results/final_metrics/baseline_summary.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved gym baseline metrics to {out_path}")
    print(f"Execution time: {elapsed:.2f}s ({elapsed / ue_count:.3f} s/UE)")

    print("\n--- PAPER LTM REFERENCE ---")
    print("capacity_avg       : 3.75")
    print("reliability_pct    : 95.00")
    print("rlf_rate           : 0.068")
    print("ho_rate            : 11.00")
    print("pp_rate            : 3.45")
    print("prep_rate          : 780.00")
    print("res_reservation_pct: 5.70")
    print("hof_rate           : 1.10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ue_number", type=int, default=20)
    # --high_res is the only supported mode and is left here for
    # backward-compat with the previous CLI; it's a no-op.
    parser.add_argument("--high_res", action="store_true")
    args = parser.parse_args()
    verify_parity(args.ue_number)
