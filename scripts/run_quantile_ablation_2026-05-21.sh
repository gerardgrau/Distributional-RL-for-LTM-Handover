#!/usr/bin/env bash
# Quantile-count ablation under corrected (v2_M_signal) physics.
# Sweep N in {10, 100, 200}; the champion baseline (N=50) is already
# pooled in results/final_metrics/qrdqn_summary.csv (kappa_10 / 3 seeds
# x 2000 ep) and acts as the in-series N=50 reference.
#
# Budget: 1 seed x 1000 ep per variant. The first attempt this evening
# (22:12) used 2 seeds but the measured rate under v2 physics is ~5s/ep,
# so 2 seeds x 3 variants projected to finish ~06:30 -- past the 06:00
# deadline. User pre-authorized dropping seeds for time ("Si veus que
# triga massa, redueix les seeds"); the N=50 reference is already pooled
# from 3 seeds so the 1-seed sweep is enough to read the trend.
# Predicted: 1 seed x 1000 ep ~= 1h 23min per variant; 3 variants ~= 4h
# 10min total. From 23:00 -> done ~03:10.

cd /home/gerard/Documents/UPC/42_4t-Q2/I2R/Distributional-RL-for-LTM-Handover
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

DATESTAMP="2026-05-21"
LOGDIR="logs/overnight_${DATESTAMP}"
mkdir -p "$LOGDIR"
QLOG="$LOGDIR/queue.log"

PY=./venv-RL/bin/python3

log() {
    echo "$(date -Iseconds) $*" | tee -a "$QLOG"
}

run_step() {
    local desc=$1 config=$2
    log "START $desc (config=$config)"
    local t0=$(date +%s)
    "$PY" src/main.py \
        --config "$config" \
        --device xpu \
        --description "$desc" \
        --agents qrdqn \
        > "$LOGDIR/$desc.log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    log "END $desc (rc=$rc, duration=$((t1-t0))s)"
}

log "QUANTILE ABLATION v2 (1 seed x 1000 ep, N in {10, 100, 200})"

for N in 10 100 200; do
    desc="v2_ablation_N${N}"
    tmp="configs/hp_search/_tmp_${desc}.yaml"
    sed -e "s/^  num_quantiles: 50/  num_quantiles: ${N}/" \
        -e "s/^  num_episodes: 2000/  num_episodes: 1000/" \
        -e "s/^  num_seeds: 3/  num_seeds: 1/" \
        configs/hp_search/kappa_10.yaml > "$tmp"
    run_step "$desc" "$tmp"
    rm -f "$tmp"
done

# ---- Post-training aggregator: pool the eval CSVs into a single
# wide-format summary at results/final_metrics/quantile_ablation_v2.csv.
# Columns: N, then mean+/-std for each of the 8 metrics + reward.
log "POST-ABLATION: aggregating quantile-N sweep into a summary CSV"

"$PY" - <<'EOF'
import glob
import os
import pandas as pd

ABLATION_DIRS = {
    10:  sorted(glob.glob("results/benchmarks/bmk_*_v2_ablation_N10"))[-1:],
    50:  sorted(glob.glob("results/benchmarks/bmk_*_v2_qrdqn_baseline"))[-1:],
    100: sorted(glob.glob("results/benchmarks/bmk_*_v2_ablation_N100"))[-1:],
    200: sorted(glob.glob("results/benchmarks/bmk_*_v2_ablation_N200"))[-1:],
}

METRICS = [
    "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
    "reliability_pct", "prep_rate", "res_reservation_pct", "reward",
]

rows = []
for N, dirs in ABLATION_DIRS.items():
    if not dirs:
        print(f"WARN: no benchmark dir for N={N}; skipping")
        continue
    bdir = dirs[0]
    raws = sorted(glob.glob(os.path.join(bdir, "eval", "qrdqn_raw_seed*.csv")))
    if not raws:
        print(f"WARN: no qrdqn_raw_seed*.csv under {bdir}/eval; skipping N={N}")
        continue
    pooled = pd.concat([pd.read_csv(p) for p in raws], ignore_index=True)
    row = {"N": N, "n_seeds": len(raws), "bmk_dir": os.path.basename(bdir)}
    for m in METRICS:
        if m in pooled.columns:
            row[f"{m}_mean"] = float(pooled[m].mean())
            row[f"{m}_std"]  = float(pooled[m].std(ddof=0))
    rows.append(row)
    print(f"  N={N:3d}: {len(raws)} seeds x {len(pooled)//len(raws)} UEs from {bdir}")

if rows:
    df = pd.DataFrame(rows).sort_values("N")
    out = "results/final_metrics/quantile_ablation_v2.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    cols = ["N", "n_seeds", "reward_mean", "capacity_avg_mean",
            "hof_rate_mean", "rlf_rate_mean", "reliability_pct_mean"]
    print(df[cols].to_string(index=False))
EOF

log "Quantile ablation done."
