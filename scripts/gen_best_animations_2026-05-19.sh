#!/usr/bin/env bash
# Generate the 3 dashboard animations for the best models under the
# corrected (2026-05-18 tutor fix) physics.
#
# - baseline (LTM hardcoded heuristic) -- no training
# - DQN champion: bmk_2026-05-19_2_dqn_baseline_lr1e-4/models/dqn_best.pth
# - QR-DQN champion: bmk_2026-05-19_1_kappa_10/models/qrdqn_best.pth
#
# All on UE500 (same UE the 2026-05-18 meeting animations used, so
# side-by-side comparison shows the physics-fix delta cleanly).

cd /home/gerard/Documents/UPC/42_4t-Q2/I2R/Distributional-RL-for-LTM-Handover
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

OUTDIR="results/animations/2026-05-19_best"
mkdir -p "$OUTDIR"

LOGDIR="logs/overnight_2026-05-19"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/animations.log"

PY=./venv-RL/bin/python3
UE=500

log() {
    echo "$(date -Iseconds) $*" | tee -a "$LOG"
}

log "Animation generation start (UE${UE})"

log "  [1/3] baseline (LTM hardcoded heuristic)"
"$PY" src/scripts/run_dashboard.py \
    --agent ltm_baseline \
    --ue_idx $UE \
    --config configs/hp_search/kappa_10.yaml \
    --output_path "$OUTDIR/01_ltm_baseline_ue${UE}.mp4" \
    >> "$LOG" 2>&1

log "  [2/3] DQN champion (lr=1e-4)"
"$PY" src/scripts/run_dashboard.py \
    --agent dqn \
    --ue_idx $UE \
    --config configs/hp_search/dqn_baseline_lr1e-4.yaml \
    --model_path results/benchmarks/bmk_2026-05-19_2_dqn_baseline_lr1e-4/models/dqn_best.pth \
    --output_path "$OUTDIR/02_dqn_ue${UE}.mp4" \
    >> "$LOG" 2>&1

log "  [3/3] QR-DQN champion (kappa_10, midpoint, mean)"
"$PY" src/scripts/run_dashboard.py \
    --agent qrdqn \
    --ue_idx $UE \
    --config configs/hp_search/kappa_10.yaml \
    --model_path results/benchmarks/bmk_2026-05-19_1_kappa_10/models/qrdqn_best.pth \
    --output_path "$OUTDIR/03_qrdqn_ue${UE}.mp4" \
    >> "$LOG" 2>&1

log "Animations done. Output: $OUTDIR/"
ls -la "$OUTDIR" >> "$LOG"
