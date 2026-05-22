#!/usr/bin/env bash
# Train the canonical QR-DQN + DQN baselines under the corrected physics
# (M*Ps fix applied to MCSEvaluation, CheckHO_Failure, AND calculate_snir_matrix).
# physics_hash bumped to v2_M_signal -> old caches rejected.
#
# Each agent: 3 seeds x 2000 ep on XPU, using the previously-locked HPs
# (lr=1e-4, gamma=0.9, hidden=[128,128], num_quantiles=50, kappa=1.0,
# midpoint/mean for QR-DQN).
#
# After training: copy the per-seed pooled CSVs into results/final_metrics
# (overwriting the placeholder single-seed numbers used in the master plot
# this afternoon) and regenerate master_bar_plots.png / master_radial_plot.png.

cd /home/gerard/Documents/UPC/42_4t-Q2/I2R/Distributional-RL-for-LTM-Handover
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

LOGDIR="logs/overnight_2026-05-19"
mkdir -p "$LOGDIR"
QLOG="$LOGDIR/queue.log"

PY=./venv-RL/bin/python3

log() {
    echo "$(date -Iseconds) $*" | tee -a "$QLOG"
}

run_step() {
    local desc=$1 config=$2 agents=$3
    log "START $desc (config=$config, agents=$agents)"
    local t0=$(date +%s)
    "$PY" src/main.py \
        --config "$config" \
        --device xpu \
        --description "$desc" \
        --agents "$agents" \
        > "$LOGDIR/$desc.log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    log "END $desc (rc=$rc, duration=$((t1-t0))s)"
}

log "BASELINES under corrected physics (v2_M_signal). 3 seeds x 2000 ep each."

# 1. QR-DQN baseline (the heavier one — quantile loss + 50-quantile head)
run_step v2_qrdqn_baseline configs/hp_search/kappa_10.yaml qrdqn

# 2. DQN baseline (vanilla DQN with QR-DQN champion HPs)
run_step v2_dqn_baseline configs/hp_search/dqn_baseline_lr1e-4.yaml dqn

# 3. Aggregate and refresh master plot. Pool 3 seeds per agent into a single
#    summary CSV (mean over 3 x 1000 = 3000 UE rows) following the format the
#    master plot tool expects.
log "POST-TRAINING: aggregating into master CSVs"

QRDQN_DIR=$(ls -dt results/benchmarks/bmk_*_v2_qrdqn_baseline 2>/dev/null | head -1)
DQN_DIR=$(ls -dt results/benchmarks/bmk_*_v2_dqn_baseline 2>/dev/null | head -1)
log "  qrdqn dir: $QRDQN_DIR"
log "  dqn dir:   $DQN_DIR"

"$PY" - <<EOF
import os, glob
import numpy as np
import pandas as pd

def pool_seeds(bmk_dir: str, agent: str, out_csv: str) -> None:
    raws = sorted(glob.glob(os.path.join(bmk_dir, "eval", f"{agent}_raw_seed*.csv")))
    if not raws:
        print(f"WARN: no raw seed CSVs found under {bmk_dir}/eval; skipping {out_csv}")
        return
    dfs = [pd.read_csv(p) for p in raws]
    pooled = pd.concat(dfs, ignore_index=True)
    metrics = [
        "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
        "reliability_pct", "prep_rate", "res_reservation_pct", "reward",
        "total_steps", "total_minutes",
    ]
    rows = []
    for m in metrics:
        if m not in pooled.columns:
            continue
        rows.append({
            "metric": m,
            "mean": float(pooled[m].mean()),
            "std": float(pooled[m].std(ddof=0)),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(raws)} seeds x {len(pooled)//len(raws)} UEs)")

pool_seeds("$QRDQN_DIR", "qrdqn", "results/final_metrics/qrdqn_summary.csv")
pool_seeds("$DQN_DIR", "dqn", "results/final_metrics/dqn_summary.csv")
EOF

log "POST-TRAINING: regenerating master plot"
"$PY" src/tools/generate_final_plots.py >> "$LOGDIR/post_training.log" 2>&1

log "Baselines done."
