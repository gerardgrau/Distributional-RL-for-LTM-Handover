#!/usr/bin/env bash
# Run 3 QR-DQN seeds in roughly max(T_cpu, T_xpu + T_cpu_solo) wall time:
#
#   t=0          seed 42 on XPU  +  seed 43 on CPU       (both training)
#   t≈T_xpu      seed 42 done    +  seed 44 on CPU       (starts when XPU frees up)
#                                    seed 43 still training on CPU
#   t≈T_cpu_43   seed 43 done    +  seed 44 still on CPU (solo, no contention)
#   t≈T_xpu+T_cpu_solo
#                seed 44 done
#
# Why 1 XPU + 2 CPU and not 2 XPU + 1 CPU:
#   At the current defaults (train_freq=4, num_quantiles=25, [128,128]
#   trunk) the per-train-step work is so small that XPU's kernel-launch
#   overhead dominates and CPU at 4 threads beats XPU per-process
#   (~0.49 vs ~0.62 s/ep, isolated benchmark). The orchestrator still
#   uses one XPU seed so both devices get used during the concurrent
#   phase — pure 3-CPU parallel would over-subscribe cores during eval.
#
# Resource contention to be aware of:
#   - main.py caps PyTorch intra-op threads at 2 (XPU) / 4 (CPU).
#   - Each process's eval phase spins up 10 single-threaded workers
#     (see src/distrl/utils/evaluation.py). When eval phases overlap
#     with another process's training, ~14 logical threads compete for
#     ~12 cores — works, but per-step times will rise a bit. This is
#     already baked into the measured ~32% wall-time saving vs 3x XPU.
#
# Usage:
#   scripts/run_parallel_seeds.sh                # default config, qrdqn, seeds 42/43/44
#   scripts/run_parallel_seeds.sh CONFIG DESC AGENTS S1 S2 S3
#     CONFIG  YAML path (default configs/config.yaml)
#     DESC    description tag for the bmk dirs (default parallel)
#     AGENTS  comma-list (default qrdqn)
#     S1..S3  seed integers (defaults 42 43 44).
#             S1 runs on XPU; S2 and S3 run on CPU.

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
PID_XPU=$!
launch "cpu_s${S2}" cpu "$S2" "$LOGDIR/cpu_s${S2}.log" &
PID_CPU1=$!

# Wait for the XPU process first; once it returns we have the freed
# core budget for another CPU seed.
wait "$PID_XPU"
launch "cpu_s${S3}" cpu "$S3" "$LOGDIR/cpu_s${S3}.log" &
PID_CPU2=$!

# Block on the remaining two.
wait "$PID_CPU1"
wait "$PID_CPU2"

echo "$(date +%H:%M:%S) all seeds done. logs under $LOGDIR/"
