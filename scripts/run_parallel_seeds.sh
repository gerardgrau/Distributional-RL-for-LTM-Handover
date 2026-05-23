#!/usr/bin/env bash
# Run 3 QR-DQN seeds in roughly the wall time of one CPU seed:
#
#   t=0      seed 42 on XPU  +  seed 43 on CPU  (both training)
#   t≈T_xpu  seed 42 done    +  seed 44 on XPU  (starts on freed XPU)
#                              + seed 43 still training on CPU
#   t≈T_cpu  seed 43 done    +  seed 44 done
#
# Assumes XPU train+eval finishes well before CPU; if that flips
# (e.g. tiny num_episodes) the second XPU launch effectively serialises
# behind the first and there's no win.
#
# Resource contention to be aware of:
#   - main.py caps PyTorch intra-op threads at 2 (XPU) / 4 (CPU).
#   - Each process's eval phase spins up 10 single-threaded workers
#     (see src/distrl/utils/evaluation.py). When eval phases overlap
#     with another process's training, ~14 logical threads compete for
#     ~12 cores — works, but per-step times will rise a bit.
#
# Usage:
#   scripts/run_parallel_seeds.sh                # default config, qrdqn, seeds 42/43/44
#   scripts/run_parallel_seeds.sh CONFIG DESC AGENTS S1 S2 S3
#     CONFIG  YAML path (default configs/config.yaml)
#     DESC    description tag for the bmk dirs (default parallel)
#     AGENTS  comma-list (default qrdqn)
#     S1..S3  seed integers (defaults 42 43 44).
#             S1, S3 run on XPU; S2 on CPU.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

CONFIG="${1:-configs/config.yaml}"
DESC="${2:-parallel}"
AGENTS="${3:-qrdqn}"
S1="${4:-42}"
S2="${5:-43}"
S3="${6:-44}"

DATESTAMP="$(date +%Y-%m-%d)"
LOGDIR="logs/parallel_${DATESTAMP}"
mkdir -p "$LOGDIR"
PY=./venv-RL/bin/python3

launch() {
    local desc=$1 device=$2 seed=$3 log=$4
    echo "$(date +%H:%M:%S) launching $desc on $device (seed=$seed)" | tee -a "$LOGDIR/queue.log"
    "$PY" src/main.py \
        --config "$CONFIG" --device "$device" \
        --description "${DESC}_${desc}" \
        --agents "$AGENTS" --seeds "$seed" \
        > "$log" 2>&1
    echo "$(date +%H:%M:%S) finished $desc" | tee -a "$LOGDIR/queue.log"
}

launch "xpu_s${S1}" xpu "$S1" "$LOGDIR/xpu_s${S1}.log" &
PID_XPU1=$!
launch "cpu_s${S2}" cpu "$S2" "$LOGDIR/cpu_s${S2}.log" &
PID_CPU=$!

# Wait only for the XPU process; once it returns we can reuse the device.
wait "$PID_XPU1"
launch "xpu_s${S3}" xpu "$S3" "$LOGDIR/xpu_s${S3}.log" &
PID_XPU2=$!

# Block on the remaining two.
wait "$PID_CPU"
wait "$PID_XPU2"

echo "$(date +%H:%M:%S) all seeds done. logs under $LOGDIR/"
