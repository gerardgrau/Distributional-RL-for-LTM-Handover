"""Overlay the training-reward curves of the canonical no-gate final agents.

The per-run ``learning_curves.png`` figures show one agent at a time. For a
presentation we want a single axis comparing how each final policy *learns*
over the full 2,000-episode, five-seed schedule. This script plots each agent
as a 50-episode rolling mean with a shaded across-seed std band.

Two figures are written:
- ``finals_learning_overlay.png``     -- the four-agent paper roster
                                         (DQN, RN, Soft-step, RA-1q).
- ``finals_learning_overlay_all.png`` -- the same plus RA (k=7) and hard CVaR.

Run from the repo root with ``PYTHONPATH`` exported (see CLAUDE.md).
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (label, benchmark folder, agent filename prefix, colour).
# Colours are the reserved per-algorithm palette shared with the head-to-head
# figures (LTM black, DQN brown, RN red, Soft-step blue, RA-1q green, RA amber).
_FINALS: list[tuple[str, str, str, str]] = [
    ("DQN (risk-neutral scalar)",
     "results/benchmarks/bmk_2026-06-04_10_nogate_final_dqn", "dqn", "#8c564b"),
    ("RN (risk-neutral QR-DQN)",
     "results/benchmarks/bmk_2026-06-04_11_nogate_final_rn", "qrdqn", "#e41a1c"),
    ("Soft-step (spectral CVaR)",
     "results/benchmarks/bmk_2026-06-05_13_nogate_final_softcvar_r20", "qrdqn", "#1f77b4"),
    ("RA-1q (single-quantile VaR)",
     "results/benchmarks/bmk_2026-06-04_14_nogate_final_ra_1q", "qrdqn", "#2ca02c"),
]

# Extra learned finals shown only in the "all" figure.
_EXTRA: list[tuple[str, str, str, str]] = [
    ("RA (hard CVaR, k=7)",
     "results/benchmarks/bmk_2026-06-04_12_nogate_final_ra", "qrdqn", "#d9a000"),
    ("Hard CVaR (alpha=0.25)",
     "results/benchmarks/bmk_2026-06-07_3_nogate_final_cvarfull_a025", "qrdqn", "#9467bd"),
]


def load_train_curves(
    expdir: str, agent: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (episode, mean_reward, std_reward) over seeds for one agent."""
    pattern = os.path.join(expdir, "train", f"{agent}_*_seed*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return np.array([]), np.array([]), np.array([])
    seed_curves = []
    for p in paths:
        df = pd.read_csv(p)
        seed_curves.append(df[["episode", "reward"]].set_index("episode"))
    combined = pd.concat(seed_curves, axis=1)
    mean = combined.mean(axis=1).values
    std = combined.std(axis=1).values
    ep = combined.index.values
    return ep, mean, std


def smooth(y: np.ndarray, w: int = 50) -> np.ndarray:
    """Centered rolling mean to denoise the per-episode reward."""
    if len(y) < w:
        return y
    return pd.Series(y).rolling(w, min_periods=1, center=True).mean().values


def _render(
    variants: list[tuple[str, str, str, str]], title: str, out: str
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, bmk, agent, color in variants:
        ep, mean, std = load_train_curves(bmk, agent)
        if len(ep) == 0:
            print(f"  WARNING missing train curves: {bmk} ({agent})")
            continue
        m = smooth(mean, 50)
        s = smooth(std, 50)
        ax.plot(ep, m, color=color, label=label, linewidth=1.8)
        ax.fill_between(ep, m - s, m + s, color=color, alpha=0.12)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward (50-ep rolling mean, +/- std over 5 seeds)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    os.makedirs("figures/02_learning_curves", exist_ok=True)
    _render(
        _FINALS,
        "Learning curves of the final policies "
        "(no-gate env, 2,000 episodes, 5 seeds)",
        "figures/02_learning_curves/finals_learning_overlay.png",
    )
    _render(
        _FINALS + _EXTRA,
        "Learning curves of all final policies "
        "(no-gate env, 2,000 episodes, 5 seeds)",
        "figures/02_learning_curves/finals_learning_overlay_all.png",
    )


if __name__ == "__main__":
    main()
