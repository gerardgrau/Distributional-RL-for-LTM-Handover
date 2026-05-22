#!/usr/bin/env bash
# Combined HP sweep under v2 (corrected SNIR) physics. Three phases:
#   Phase 1: Quadrature modes      (5 variants, ~8h)
#   Phase 2: HP refresh (lr/tau)   (5 variants, ~8h)
#   Phase 3: Misc (batch/hidden)   (3 variants, ~5h)
#
# Each variant: 1 seed x 1000 ep (per feedback_episode_budgets and the
# pattern set by the quantile ablation: 1 seed is enough to read trends
# now that the N=50 champion is fully pooled from 3 seeds x 2000 ep).
# Predicted ~85-105 min per variant on XPU under v2 physics.
#
# Skipped on purpose (would duplicate existing work):
#   - configs/quantile_study/qmode_midpoint.yaml -- this IS the champion
#     (midpoint, N=50). Use bmk_*_v2_qrdqn_baseline as the reference.
#   - configs/hp_refresh/baseline_refresh.yaml   -- also a midpoint+N=50
#     champion duplicate.
#
# After all phases, one aggregator pools eval CSVs into three master
# CSVs at results/final_metrics/{quadrature,hp_refresh,misc_hp}_sweep_v2.csv,
# each including the champion row for direct comparison.

cd /home/gerard/Documents/UPC/42_4t-Q2/I2R/Distributional-RL-for-LTM-Handover
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

DATESTAMP="2026-05-22"
LOGDIR="logs/sweep_${DATESTAMP}"
mkdir -p "$LOGDIR"
QLOG="$LOGDIR/queue.log"
PY=./venv-RL/bin/python3

log() { echo "$(date -Iseconds) $*" | tee -a "$QLOG"; }

# Generate a temp config from a base, forcing num_episodes=1000 and
# num_seeds=1 (defensive; quantile_study/* are already that way, but
# hp_refresh/* are at 500 ep).
mktmp_from() {
    local base=$1 out=$2
    sed -e "s/^  num_episodes:.*/  num_episodes: 1000/" \
        -e "s/^  num_seeds:.*/  num_seeds: 1/" \
        "$base" > "$out"
}

run_step() {
    local desc=$1 config=$2 agents=${3:-qrdqn}
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
        --agents "$agents" \
        > "$LOGDIR/$desc.log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    log "END $desc (rc=$rc, duration=$((t1-t0))s)"
}

# ===========================================================
# Phase 1: Quadrature study  (skip midpoint = champion)
# ===========================================================
log "==== PHASE 1: QUADRATURE STUDY ===="
P1=(
    "configs/quantile_study/qmode_gauss_legendre.yaml:v2_qmode_gl"
    "configs/quantile_study/qmode_trapezoidal.yaml:v2_qmode_trap"
    "configs/quantile_study/qmode_simpson.yaml:v2_qmode_simpson"
    "configs/quantile_study/qmode_cvar_full.yaml:v2_qmode_cvar_full"
    "configs/quantile_study/qmode_cvar_truncated.yaml:v2_qmode_cvar_trunc"
)
for pair in "${P1[@]}"; do
    base="${pair%%:*}"; desc="${pair##*:}"
    tmp="${base%.yaml}_tmp_1seed.yaml"
    mktmp_from "$base" "$tmp"
    run_step "$desc" "$tmp"
    rm -f "$tmp"
done

# ===========================================================
# Phase 2: HP refresh under v2  (skip baseline_refresh = champion)
# ===========================================================
log "==== PHASE 2: HP REFRESH ===="
P2=(
    "configs/hp_refresh/lr_3e-4.yaml:v2_hpr_lr_3e-4"
    "configs/hp_refresh/tau_0005.yaml:v2_hpr_tau_0005"
    "configs/hp_refresh/tau_005.yaml:v2_hpr_tau_005"
    "configs/hp_refresh/hard_update.yaml:v2_hpr_hard"
)
for pair in "${P2[@]}"; do
    base="${pair%%:*}"; desc="${pair##*:}"
    tmp="${base%.yaml}_tmp_1seed.yaml"
    mktmp_from "$base" "$tmp"
    run_step "$desc" "$tmp"
    rm -f "$tmp"
done

# ===========================================================
# Phase 3: Misc HP (batch_size, hidden_dims) -- generated from
# kappa_10 by overriding one field at a time, plus the 1seed/1000ep
# normalization.
# ===========================================================
log "==== PHASE 3: MISC HP ===="

build_misc() {
    local field_sed=$1 out=$2
    sed -e "$field_sed" \
        -e "s/^  num_episodes: 2000/  num_episodes: 1000/" \
        -e "s/^  num_seeds: 3/  num_seeds: 1/" \
        configs/hp_search/kappa_10.yaml > "$out"
}

# 3a: batch_size 128
tmp=configs/hp_search/_tmp_v2_batch_128.yaml
build_misc "s/^  batch_size: 256/  batch_size: 128/" "$tmp"
run_step v2_batch_128 "$tmp"
rm -f "$tmp"

# 3b: batch_size 512
tmp=configs/hp_search/_tmp_v2_batch_512.yaml
build_misc "s/^  batch_size: 256/  batch_size: 512/" "$tmp"
run_step v2_batch_512 "$tmp"
rm -f "$tmp"

# 3c: hidden_dims [256, 256]
tmp=configs/hp_search/_tmp_v2_hidden_256.yaml
build_misc "s/^  hidden_dims: \[128, 128\]/  hidden_dims: [256, 256]/" "$tmp"
run_step v2_hidden_256 "$tmp"
rm -f "$tmp"

# ===========================================================
# Aggregate per-phase CSVs at results/final_metrics/<phase>_sweep_v2.csv.
# Each table includes the champion row pulled from bmk_*_v2_qrdqn_baseline
# so the variant-vs-champion comparison is direct.
# ===========================================================
log "POST: aggregating sweep results"

"$PY" - <<'EOF'
import glob, os
import pandas as pd

PHASES = {
    "quadrature_sweep_v2.csv": [
        ("midpoint (champion)",   "results/benchmarks/bmk_*_v2_qrdqn_baseline"),
        ("gauss_legendre",        "results/benchmarks/bmk_*_v2_qmode_gl"),
        ("trapezoidal",           "results/benchmarks/bmk_*_v2_qmode_trap"),
        ("simpson",               "results/benchmarks/bmk_*_v2_qmode_simpson"),
        ("cvar_full(0.1)",        "results/benchmarks/bmk_*_v2_qmode_cvar_full"),
        ("cvar_truncated(0.1)",   "results/benchmarks/bmk_*_v2_qmode_cvar_trunc"),
    ],
    "hp_refresh_sweep_v2.csv": [
        ("champion lr=1e-4 tau=0.01",  "results/benchmarks/bmk_*_v2_qrdqn_baseline"),
        ("lr=3e-4",                     "results/benchmarks/bmk_*_v2_hpr_lr_3e-4"),
        ("tau=0.005",                   "results/benchmarks/bmk_*_v2_hpr_tau_0005"),
        ("tau=0.05",                    "results/benchmarks/bmk_*_v2_hpr_tau_005"),
        ("tau=1.0 (hard)",              "results/benchmarks/bmk_*_v2_hpr_hard"),
    ],
    "misc_hp_sweep_v2.csv": [
        ("champion batch=256 hidden=[128,128]", "results/benchmarks/bmk_*_v2_qrdqn_baseline"),
        ("batch_size=128",                       "results/benchmarks/bmk_*_v2_batch_128"),
        ("batch_size=512",                       "results/benchmarks/bmk_*_v2_batch_512"),
        ("hidden_dims=[256,256]",                "results/benchmarks/bmk_*_v2_hidden_256"),
    ],
}

METRICS = [
    "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
    "reliability_pct", "prep_rate", "res_reservation_pct", "reward",
]

for out_file, variants in PHASES.items():
    rows = []
    for label, pattern in variants:
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            print(f"WARN ({out_file}): no dir for {label} (pattern {pattern})")
            continue
        bdir = dirs[-1]
        raws = sorted(glob.glob(os.path.join(bdir, "eval", "qrdqn_raw_seed*.csv")))
        if not raws:
            print(f"WARN ({out_file}): no qrdqn_raw_seed*.csv under {bdir}/eval")
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
        out_path = f"results/final_metrics/{out_file}"
        df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")
        cols = ["variant", "n_seeds", "reward_mean", "capacity_avg_mean",
                "hof_rate_mean", "rlf_rate_mean", "reliability_pct_mean"]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
EOF

log "Sweep done."
