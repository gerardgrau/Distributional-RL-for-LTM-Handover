"""Atari benchmark runner for QR-DQN variants.

Trains a (QR-)DQN agent on a single Atari game with Nature-DQN preprocessing
(grayscale, 84x84, action-repeat 4, frame-stack 4) and the standard reward-
clipping convention. Same agent code as the LTM benchmark — just CNN trunk
and a frame budget instead of episode budget.

Results land in ``results/atari/<game>_<description>_<timestamp>/``:
    config.yaml            copy of the active config
    train.csv              per-episode reward + loss + epsilon log
    model.pth              final agent checkpoint
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from tqdm import tqdm

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.distrl.agents.distributional.qrdqn import QRDQNAgent  # noqa: E402
from src.distrl.agents.replay_buffer import ReplayBuffer  # noqa: E402
from src.distrl.agents.standard.dqn import DQNAgent  # noqa: E402

gym.register_envs(ale_py)


def make_env(game: str, seed: int | None = None):
    env = gym.make(f"ALE/{game}-v5", frameskip=1)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    if seed is not None:
        env.reset(seed=seed)
    return env


def build_agent(
    agent_type: str,
    config: dict[str, Any],
    env: gym.Env,
    device: str,
):
    agent_config = dict(config["agent"])
    agent_config["trunk_type"] = "cnn"
    if agent_type == "qrdqn":
        return QRDQNAgent(
            agent_config, env.observation_space, env.action_space, device=device,
        )
    if agent_type == "dqn":
        return DQNAgent(
            agent_config, env.observation_space, env.action_space, device=device,
        )
    raise ValueError(f"Unknown agent_type: {agent_type!r}")


def evaluate(agent, eval_env, episodes: int, epsilon_eval: float = 0.001) -> float:
    """Greedy evaluation over `episodes` complete episodes. Returns mean total
    reward. epsilon_eval allows tiny exploration to break loops in determ. envs."""
    returns: list[float] = []
    for _ in range(episodes):
        state, _ = eval_env.reset()
        total = 0.0
        done = False
        while not done:
            a = agent.select_action(state, epsilon=epsilon_eval)
            state, r, term, trunc, _ = eval_env.step(a)
            total += float(r)
            done = bool(term or trunc)
        returns.append(total)
    return float(np.mean(returns))


def train_loop(
    agent,
    env,
    eval_env,
    config: dict[str, Any],
    results_dir: str,
    frame_budget: int,
    eval_every: int,
    eval_episodes: int,
    device: str,
    save: bool,
) -> None:
    agent_cfg = config["agent"]
    buf = ReplayBuffer(
        max_size=int(agent_cfg.get("buffer_size", 100_000)),
        state_shape=env.observation_space.shape,
        state_dtype=torch.uint8,
    )

    eps_start = float(agent_cfg.get("epsilon_start", 1.0))
    eps_end = float(agent_cfg.get("epsilon_end", 0.01))
    eps_decay_frames = int(
        agent_cfg.get("epsilon_decay_frames", min(1_000_000, frame_budget // 2))
    )
    batch_size = int(agent_cfg.get("batch_size", 32))
    train_freq = int(agent_cfg.get("train_freq", 4))
    learn_start = int(agent_cfg.get("learn_start", 5_000))
    clip_reward = bool(agent_cfg.get("clip_reward", True))

    train_csv = os.path.join(results_dir, "train.csv")
    if save:
        with open(train_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "episode", "episode_reward", "mean_loss", "epsilon",
                "wall_time_s",
            ])
    eval_csv = os.path.join(results_dir, "eval.csv")
    if save and eval_every:
        with open(eval_csv, "w", newline="") as f:
            csv.writer(f).writerow(["frame", "mean_return"])

    state, _ = env.reset()
    episode_reward = 0.0
    episode_idx = 0
    episode_losses: list[float] = []
    t0 = time.time()

    pbar = tqdm(total=frame_budget, desc=f"frames", smoothing=0.05)
    last_pbar = 0

    for frame in range(frame_budget):
        # Linearly anneal epsilon over eps_decay_frames
        progress = min(1.0, frame / max(1, eps_decay_frames))
        epsilon = eps_start + (eps_end - eps_start) * progress

        action = agent.select_action(state, epsilon=epsilon)
        next_state, reward, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        stored_reward = float(np.sign(reward)) if clip_reward else float(reward)
        buf.push(state, action, stored_reward, next_state, done)
        episode_reward += float(reward)
        state = next_state

        if len(buf) >= max(learn_start, batch_size) and frame % train_freq == 0:
            metrics = agent.train_step(buf.sample(batch_size, device=device))
            episode_losses.append(metrics["loss"])

        if frame - last_pbar >= 1000:
            pbar.update(frame - last_pbar)
            pbar.set_postfix({
                "ep": episode_idx,
                "R": f"{episode_reward:.0f}",
                "eps": f"{epsilon:.3f}",
            })
            last_pbar = frame

        if done:
            mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            if save:
                with open(train_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        frame, episode_idx, episode_reward, mean_loss, epsilon,
                        time.time() - t0,
                    ])
            episode_idx += 1
            episode_reward = 0.0
            episode_losses = []
            state, _ = env.reset()

        if eval_every and frame > 0 and frame % eval_every == 0:
            mean_eval = evaluate(agent, eval_env, eval_episodes)
            if save:
                with open(eval_csv, "a", newline="") as f:
                    csv.writer(f).writerow([frame, mean_eval])
            tqdm.write(f"  [eval @ frame {frame}] mean return over {eval_episodes} eps = {mean_eval:.2f}")

    pbar.update(frame_budget - last_pbar)
    pbar.close()

    if save:
        agent.save(os.path.join(results_dir, "model.pth"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--game", default="Breakout")
    parser.add_argument(
        "--frames", type=int, default=500_000,
        help="Total frames to train (action-repeat counts).",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--description", default="atari")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument(
        "--eval_every", type=int, default=0,
        help="Frames between intermediate greedy evals (0 = end-only).",
    )
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument(
        "--agent", default="qrdqn", choices=["qrdqn", "dqn"],
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(args.game, seed=args.seed)
    eval_env = make_env(args.game, seed=args.seed + 10_000)

    agent = build_agent(args.agent, config, env, args.device)

    save = not args.no_save
    if save:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(
            "results", "atari", f"{args.game}_{args.description}_{ts}",
        )
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    else:
        results_dir = "/tmp/atari_smoke"
        os.makedirs(results_dir, exist_ok=True)

    print(f"=== Atari Benchmark ===")
    print(f"Game:     {args.game}")
    print(f"Agent:    {args.agent}")
    print(f"Device:   {args.device}")
    print(f"Frames:   {args.frames}")
    print(f"Out dir:  {results_dir}")
    print(f"Config:   {args.config}")

    train_loop(
        agent=agent,
        env=env,
        eval_env=eval_env,
        config=config,
        results_dir=results_dir,
        frame_budget=args.frames,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        device=args.device,
        save=save,
    )

    # End-of-run eval (always, for the final summary).
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    print(f"\nFinal eval mean return over {args.eval_episodes} eps = {final_eval:.2f}")
    if save:
        with open(os.path.join(results_dir, "final_eval.txt"), "w") as f:
            f.write(f"mean_return={final_eval}\nepisodes={args.eval_episodes}\n")

    env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
