#!/usr/bin/env bash
# Overnight HP-search queue (2026-05-19).
# Priority order:
#   1. QR-DQN champion kappa=1.0 — re-validates the previous champion HPs
#      under the corrected physics. Becomes the solid QR-DQN baseline.
#   2. DQN baseline lr=1e-4     — solid DQN baseline (DQN never properly
#      HP-searched before tonight; using QR-DQN champion HPs).
#   3. QR-DQN kappa=0.5         — kappa search variant A.
#   4. QR-DQN kappa=2.0         — kappa search variant B.
#   5. (stretch) num_quantiles ablation N in {10, 100, 200}.
#
# Each step writes its own bmk_YYYY-MM-DD_*_<desc>/ folder under
# results/benchmarks/. Per-step stdout/stderr goes to logs/.

# Do NOT set -u (PYTHONPATH may be unset) or -e (one failed step
# should not block the rest of the queue).

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
    local desc="$1"
    local config="$2"
    local agents="$3"
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

log "Queue starting on $(hostname). PWD=$(pwd)"

# 1. QR-DQN champion kappa=1.0 (solid baseline re-validation)
run_step kappa_10 configs/hp_search/kappa_10.yaml qrdqn

# 2. DQN baseline (solid baseline)
run_step dqn_baseline_lr1e-4 configs/hp_search/dqn_baseline_lr1e-4.yaml dqn

# 3. QR-DQN kappa=0.5 (kappa search variant A)
run_step kappa_05 configs/hp_search/kappa_05.yaml qrdqn

# 4. QR-DQN kappa=2.0 (kappa search variant B)
run_step kappa_20 configs/hp_search/kappa_20.yaml qrdqn

# 5. Stretch: num_quantiles ablation. Build temp configs from kappa_10
#    (the champion) and run N in {10, 100, 200}.
log "STRETCH quantile ablation starting"
for N in 10 100 200; do
    desc="ablation_N${N}"
    tmp="configs/hp_search/_tmp_${desc}.yaml"
    sed "s/^  num_quantiles: 50/  num_quantiles: ${N}/" \
        configs/hp_search/kappa_10.yaml > "$tmp"
    run_step "$desc" "$tmp" qrdqn
    rm -f "$tmp"
done

log "Queue done."
