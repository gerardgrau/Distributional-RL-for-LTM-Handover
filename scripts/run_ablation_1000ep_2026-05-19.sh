#!/usr/bin/env bash
# Quantile-count ablation, restarted at 1000 episodes (2026-05-19 morning).
# User feedback: "1000 episodes provides basically the exact same output
# as training over 2000 steps" for HP-comparison purposes -- saves ~3h 30min
# vs the original 2000-ep plan. Champion (kappa_10, num_quantiles=50) was
# already trained at 2000 ep and is the in-series N=50 reference -- not
# re-run here.
#
# Each step builds a temp config from kappa_10.yaml with num_quantiles and
# num_episodes overridden via sed.

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

log "ABLATION RESTART (1000 ep, N in {10, 100, 200})"

for N in 10 100 200; do
    desc="ablation_N${N}"
    tmp="configs/hp_search/_tmp_${desc}.yaml"
    sed -e "s/^  num_quantiles: 50/  num_quantiles: ${N}/" \
        -e "s/^  num_episodes: 2000/  num_episodes: 1000/" \
        configs/hp_search/kappa_10.yaml > "$tmp"
    run_step "$desc" "$tmp"
    rm -f "$tmp"
done

log "Ablation done."
