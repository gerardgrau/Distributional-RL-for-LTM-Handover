"""Frozen-weight evaluation over the full UE dataset.

`run_evaluation` is called at the end of every training seed (main.py)
and also by the standalone `src/evaluate_model.py` tool. It iterates
over all UEs, runs one episode each with epsilon=0, and aggregates the
8 LTM metrics + per-episode reward.

Two execution modes:

  - Sequential (default for small `num_eval` or when num_workers == 1):
    runs all episodes inside the calling process. Used for the baseline
    agent (which is stateless and reaches per-episode cost in the
    millisecond range) and for tiny datasets where multiprocessing
    startup cost would dominate.

  - Parallel (default for >= 48 episodes, learned agents): spawns a
    ProcessPoolExecutor with `num_workers` CPU workers. Each worker
    reloads the saved model from disk, takes a contiguous chunk of UE
    indices, and returns per-episode metrics. We force CPU workers
    because the model is small (~150k params) and the per-call
    XPU/CUDA round-trip costs more than CPU forward at batch=1.

Determinism: both modes seed `np.random.seed(eval_base_seed + ue_idx)`
at the start of each UE episode (eval_base_seed = 1000 + training
seed). This makes sequential and parallel produce bit-identical
per-episode metrics, and makes eval results independent of the
training-trajectory RNG leftover (the previous behaviour was
trajectory-dependent which precluded reproducible comparisons).
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Force 'spawn' for the worker pool. Linux's default fork() copies the
# parent process verbatim — including PyTorch's intra-op thread pool,
# whose threads do not survive fork(). Calling PyTorch ops in a
# fork-ed worker then deadlocks. Spawn starts a fresh interpreter that
# initialises its own torch state cleanly.
_SPAWN_CTX = multiprocessing.get_context("spawn")

from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.metrics import calculate_8_metrics

# Per-episode metric dict shape — convenient to type the worker return.
_METRIC_KEYS = (
    "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
    "reliability_pct", "prep_rate", "res_reservation_pct",
    "total_steps", "total_minutes", "reward",
)


def _build_agent_for_eval(
    agent_type: str,
    observation_space: Any,
    action_space: Any,
    agent_config: dict,
    device: str = "cpu",
) -> Any:
    """Construct a fresh agent from config (used by parallel workers)."""
    if agent_type == "qrdqn":
        from src.distrl.agents.distributional.qrdqn import QRDQNAgent
        return QRDQNAgent(agent_config, observation_space, action_space, device=device)
    if agent_type == "dqn":
        from src.distrl.agents.standard.dqn import DQNAgent
        return DQNAgent(agent_config, observation_space, action_space, device=device)
    if agent_type == "ltm_baseline":
        from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
        return LTMBaselineAgent(agent_config, observation_space, action_space, device=device)
    raise ValueError(f"Unknown agent type: {agent_type!r}")


def _run_episode(
    env: LTMEnv,
    agent: Any,
    config: dict,
    is_baseline: bool,
) -> dict[str, Any]:
    """One eval episode → 8 metrics + reward."""
    state, _ = env.reset()
    agent.reset()
    done = False
    last_info: dict = {}
    episode_reward = 0.0
    while not done:
        if is_baseline:
            state, reward, done, _, info = env.step(
                0, high_res_callback=agent.select_action,
            )
        else:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, _, info = env.step(action)
        episode_reward += reward
        if done:
            last_info = info

    m = calculate_8_metrics(
        mcs_history=last_info["metrics"]["mcs"],
        rlf_history=last_info["metrics"]["rlf"],
        ho_history=last_info["metrics"]["ho"],
        hof_history=last_info["metrics"]["hof"],
        pp_history=last_info["metrics"]["pp"],
        serving_history=last_info["metrics"]["serving"],
        pl3_history=last_info["metrics"]["pl3"],
        config=config,
    )
    m["reward"] = episode_reward
    return m


def _run_chunk_in_worker(
    model_path: str | None,
    config: dict,
    agent_type: str,
    ue_start: int,
    ue_end: int,
    eval_base_seed: int,
) -> list[dict[str, Any]]:
    """Worker entry point. Process UE indices [ue_start, ue_end).

    Each worker constructs its own env and agent, loads the saved model
    (when model_path is provided — baseline has none), and runs the
    assigned chunk with per-UE seeding.
    """
    # Pin each worker to a single intra-op thread. PyTorch/NumPy/OpenMP
    # otherwise default to ~cpu_count threads per process, and with N
    # workers each spinning N threads we oversubscribe the box (load
    # average explodes, throughput drops). Single-threaded workers are
    # what we want for the parallel-of-many-small-models pattern.
    import torch as _torch
    _torch.set_num_threads(1)

    env = LTMEnv(config=config)
    env.current_ue_idx = ue_start

    agent = _build_agent_for_eval(
        agent_type, env.observation_space, env.action_space,
        config["agent"], device="cpu",
    )
    if model_path is not None:
        agent.load(model_path)

    is_baseline = agent_type == "ltm_baseline"

    results: list[dict[str, Any]] = []
    for i in range(ue_end - ue_start):
        ue_idx = ue_start + i
        # Per-UE seed — matches between sequential and parallel paths.
        np.random.seed(eval_base_seed + ue_idx)
        results.append(_run_episode(env, agent, config, is_baseline))
    env.close()
    return results


def _run_sequential_eval(
    agent: Any,
    config: dict,
    agent_type: str,
    num_eval: int,
    eval_base_seed: int,
    progress_desc: str,
) -> list[dict[str, Any]]:
    """Sequential path — runs everything in the calling process."""
    env = LTMEnv(config=config)
    is_baseline = agent_type == "ltm_baseline"
    results: list[dict[str, Any]] = []
    for ue_idx in tqdm(range(num_eval), desc=progress_desc, leave=False):
        np.random.seed(eval_base_seed + ue_idx)
        results.append(_run_episode(env, agent, config, is_baseline))
    env.close()
    return results


def _run_parallel_eval(
    agent: Any,
    config: dict,
    agent_type: str,
    num_eval: int,
    num_workers: int,
    eval_base_seed: int,
    progress_desc: str,
    model_path: str | None,
) -> list[dict[str, Any]]:
    """Parallel path — chunks UE indices across CPU workers."""
    # Ensure a model file exists. If the caller (main.py) already saved
    # one we'll reuse that; otherwise dump a temp file just for workers.
    cleanup_tmp = False
    if model_path is None and agent_type != "ltm_baseline":
        tmp = tempfile.NamedTemporaryFile(
            prefix=f"eval_{agent_type}_", suffix=".pth", delete=False,
        )
        tmp.close()
        agent.save(tmp.name)
        model_path = tmp.name
        cleanup_tmp = True

    # Contiguous chunks: worker w gets UEs [w*chunk_size, (w+1)*chunk_size).
    chunk_size = (num_eval + num_workers - 1) // num_workers
    chunks = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min(start + chunk_size, num_eval)
        if start < end:
            chunks.append((start, end))

    # Pin BLAS / OpenMP to a single thread per spawned worker. These
    # libraries read their thread count from env vars at import time;
    # spawn-mode children inherit the parent's environ, so setting
    # these BEFORE creating the executor is what reaches the workers.
    # Saving/restoring is cosmetic — only matters if the parent later
    # forks another worker pool with different env.
    _thread_env_keys = (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
    _saved_env = {k: os.environ.get(k) for k in _thread_env_keys}
    for k in _thread_env_keys:
        os.environ[k] = "1"

    all_results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=_SPAWN_CTX,
        ) as exe:
            futures = [
                exe.submit(
                    _run_chunk_in_worker,
                    model_path, config, agent_type,
                    start, end, eval_base_seed,
                )
                for (start, end) in chunks
            ]
            with tqdm(total=num_eval, desc=progress_desc, leave=False) as pbar:
                # Collect in submission order so per-UE metrics stay
                # aligned with UE index (the raw CSV is per-episode).
                for fut, (start, end) in zip(futures, chunks):
                    chunk_results = fut.result()
                    all_results.extend(chunk_results)
                    pbar.update(end - start)
    finally:
        # Restore parent env (cosmetic; doesn't affect already-loaded
        # libraries in this process).
        for k, v in _saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if cleanup_tmp and model_path is not None:
            try:
                os.remove(model_path)
            except OSError:
                pass

    return all_results


def run_evaluation(
    agent: Any,
    config: dict,
    experiment_dir: str,
    agent_type: str,
    seed: int,
    save_results: bool = True,
    output_prefix: str | None = None,
    num_workers: int | None = None,
) -> dict[str, float]:
    """
    Evaluates the frozen agent on all trajectories available in the environment
    (based on the dataset and config) with epsilon=0.

    Args:
        num_workers: how many CPU worker processes for the parallel path.
            None (default) auto-picks: 1 for baseline / small datasets,
            min(6, cpu_count) otherwise.
    """
    print(f"    -> Starting Formal Evaluation Phase (Seed {seed})...")

    # Cheap probe of the trajectory count without holding the env open.
    probe_env = LTMEnv(config=config)
    num_eval = len(probe_env.files)
    probe_env.close()
    if num_eval == 0:
        print("    -> Skipping Evaluation: No trajectories found.")
        return {}

    is_baseline = agent_type.lower() == "ltm_baseline"
    agent_type_lower = agent_type.lower()
    eval_base_seed = 1000 + int(seed)
    progress_desc = f"    Eval {agent_type}"

    # Worker auto-selection. Baseline + small datasets stay sequential
    # (multiproc startup would dominate). For everything else we cap at
    # 10 workers: a 200-UE sweep on a 12-core box showed throughput
    # peaking at 10 workers (7.14x speedup vs sequential) and regressing
    # at 12 (spawn / OS background contention). cpu_count - 2 leaves
    # headroom for the OS + the main process.
    if num_workers is None:
        if is_baseline or num_eval < 48:
            num_workers = 1
        else:
            num_workers = min(10, max(1, (os.cpu_count() or 1) - 2))

    if num_workers > 1:
        # Reuse the already-saved checkpoint if main.py saved one before
        # calling us; otherwise the parallel helper writes a temp file.
        model_path: str | None = None
        if save_results and not is_baseline:
            candidate = os.path.join(
                experiment_dir, "models", f"{agent_type_lower}_seed{seed}.pth",
            )
            if os.path.exists(candidate):
                model_path = candidate
        all_eval_metrics = _run_parallel_eval(
            agent, config, agent_type_lower, num_eval, num_workers,
            eval_base_seed, progress_desc, model_path,
        )
    else:
        all_eval_metrics = _run_sequential_eval(
            agent, config, agent_type_lower, num_eval,
            eval_base_seed, progress_desc,
        )

    # Aggregate results
    df = pd.DataFrame(all_eval_metrics)
    summary = {
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict()
    }

    # Save to files
    if save_results:
        if output_prefix:
            # Predictable naming: always append suffixes to the user prefix
            summary_csv = f"{output_prefix}_summary.csv"
            raw_csv = f"{output_prefix}_raw.csv"
        else:
            # Default behavior for benchmarks (organized in folders)
            eval_dir = os.path.join(experiment_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            summary_csv = os.path.join(eval_dir, f"{agent_type}_summary_seed{seed}.csv")
            raw_csv = os.path.join(eval_dir, f"{agent_type}_raw_seed{seed}.csv")

        # 1. Save Summary CSV (Metric, Mean, Std)
        metric_order = [
            "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
            "reliability_pct", "prep_rate", "res_reservation_pct",
            "reward", "total_steps", "total_minutes"
        ]

        summary_df = pd.DataFrame({
            "metric": summary["mean"].keys(),
            "mean": summary["mean"].values(),
            "std": summary["std"].values()
        })

        # Sort according to standard order, keeping any extra metrics at the end
        summary_df['metric'] = pd.Categorical(
            summary_df['metric'],
            categories=metric_order + [m for m in summary_df['metric'] if m not in metric_order],
            ordered=True
        )
        summary_df = summary_df.sort_values('metric')

        summary_df.to_csv(summary_csv, index=False)

        # 2. Save Raw CSV (Per-episode metrics)
        df.to_csv(raw_csv, index_label="eval_episode")

        print(f"    -> Summary saved to {summary_csv}")
        print(f"    -> Raw metrics saved to {raw_csv}")

    print(f"    -> Evaluation Complete. HO Rate: {summary['mean']['ho_rate']:.2f} ± {summary['std']['ho_rate']:.2f}")
    return summary['mean']
