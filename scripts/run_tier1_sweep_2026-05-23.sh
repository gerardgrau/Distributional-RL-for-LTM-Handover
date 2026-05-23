#!/usr/bin/env bash
# Tier 1 baseline-solidification sweep (2026-05-23) under v2 physics.
# Each variant: 1 seed x 1000 ep (per feedback_episode_budgets).
#
# Three knobs left untested by the 2026-05-22 sweep:
#   - grad_clip=10        (Dabney's original QR-DQN choice; not in code until today)
#   - adam_eps=1.5e-4     (Rainbow/IQN setting; PyTorch default was used so far)
#   - gamma=0.95          (longer effective horizon; pre-v2 result was inconclusive)
#
# After all three finish, aggregate into
# results/final_metrics/tier1_sweep_v2.csv with the 3-seed champion row at
# the top for direct comparison.

cd /home/gerard/Documents/UPC/42_4t-Q2/I2R/Distributional-RL-for-LTM-Handover
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

DATESTAMP="2026-05-23"
LOGDIR="logs/tier1_${DATESTAMP}"
mkdir -p "$LOGDIR"
QLOG="$LOGDIR/queue.log"
PY=./venv-RL/bin/python3

log() { echo "$(date -Iseconds) $*" | tee -a "$QLOG"; }

run_step() {
    local desc=$1 config=$2
    if compgen -G "results/benchmarks/bmk_*_${desc}" > /dev/null; then
        log "SKIP $desc (already done: $(ls -d results/benchmarks/bmk_*_${desc} 2>/dev/null | head -1))"
        return 0
    fi
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

log "==== TIER 1 SWEEP (3 variants, 1 seed x 1000 ep each) ===="

run_step v2_tier1_grad_clip_10  configs/hp_refresh/grad_clip_10.yaml
run_step v2_tier1_adam_eps_15e-4 configs/hp_refresh/adam_eps_15e-4.yaml
run_step v2_tier1_gamma_095     configs/hp_refresh/gamma_095.yaml

log "POST: aggregating tier1 sweep results"

"$PY" - <<'EOF'
import glob, os
import pandas as pd

VARIANTS = [
    ("champion (grad_clip=off, adam_eps=1e-8, gamma=0.9)",
     "results/benchmarks/bmk_*_v2_qrdqn_baseline"),
    ("grad_clip=10",
     "results/benchmarks/bmk_*_v2_tier1_grad_clip_10"),
    ("adam_eps=1.5e-4",
     "results/benchmarks/bmk_*_v2_tier1_adam_eps_15e-4"),
    ("gamma=0.95",
     "results/benchmarks/bmk_*_v2_tier1_gamma_095"),
]
METRICS = [
    "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
    "reliability_pct", "prep_rate", "res_reservation_pct", "reward",
]

rows = []
for label, pattern in VARIANTS:
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f"WARN: no dir for {label} (pattern {pattern})")
        continue
    bdir = dirs[-1]
    raws = sorted(glob.glob(os.path.join(bdir, "eval", "qrdqn_raw_seed*.csv")))
    if not raws:
        print(f"WARN: no qrdqn_raw_seed*.csv under {bdir}/eval")
        continue
    pooled = pd.concat([pd.read_csv(p) for p in raws], ignore_index=True)
    row = {"variant": label, "n_seeds": len(raws),
           "bmk_dir": os.path.basename(bdir)}
    for m in METRICS:
        if m in pooled.columns:
            row[f"{m}_mean"] = float(pooled[m].mean())
            row[f"{m}_std"]  = float(pooled[m].std(ddof=0))
    rows.append(row)

if rows:
    df = pd.DataFrame(rows)
    out_path = "results/final_metrics/tier1_sweep_v2.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    cols = ["variant", "n_seeds", "reward_mean", "capacity_avg_mean",
            "hof_rate_mean", "rlf_rate_mean", "reliability_pct_mean"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
EOF

log "Tier 1 sweep done."
