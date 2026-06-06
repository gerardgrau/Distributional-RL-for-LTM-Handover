"""The risk frontier: reward vs tail, three ways to dial risk aversion.

Plots reward/action against two tail metrics (ping-pong, HO-failure) for the
three risk mechanisms studied, all at the matched 1000ep x3seed budget:

  * single-quantile (Q1): one quantile at tau = alpha/2, alpha swept.
  * hard-CVaR (truncated): k = ceil(25*alpha) tail quantiles, alpha swept.
  * soft-step weights: all 25 quantiles, [0,0.25] up-weighted by ratio r swept.

DQN and RN are reference anchors (top-right; least risk-averse). The efficient
frontier is the upper-left envelope (more reward, fewer tail events).

Run (PYTHONPATH must include src):
    ./venv-RL/bin/python3 src/tools/plot_risk_frontier.py
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BENCH = "results/benchmarks"
_GRID = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]


def _agg(suffix: str, agent: str = "qrdqn") -> dict[str, float] | None:
    fs = sorted(glob.glob(f"{_BENCH}/*{suffix}"))
    if not fs:
        return None
    rpa, pp, hof = [], [], []
    for f in sorted(glob.glob(f"{fs[-1]}/eval/{agent}_raw_seed*.csv")):
        rows = list(csv.DictReader(open(f)))
        rpa.append(sum(float(x["reward"]) for x in rows)
                   / sum(float(x["n_decisions"]) for x in rows))
        pp.append(np.mean([float(x["pp_rate"]) for x in rows]))
        hof.append(np.mean([float(x["hof_rate"]) for x in rows]))
    return {"rpa": np.mean(rpa), "pp": np.mean(pp), "hof": np.mean(hof)}


def _dial(suffixes: list[str]) -> dict[str, np.ndarray]:
    pts = [p for p in (_agg(s) for s in suffixes) if p]
    return {k: np.array([p[k] for p in pts]) for k in ("rpa", "pp", "hof")}


def main() -> None:
    tags = [f"{int(round(a * 100)):03d}" for a in _GRID]
    q1 = _dial([f"nogate_uf_q1_a{t}" for t in tags])
    cvar = _dial([f"nogate_uf_cvar_a{t}" for t in tags])
    # Soft-step dial: RN (r=1) -> r in {1.5,2,3,5} -> hard CVaR a=0.25 (r=inf).
    rn = _agg("nogate_final_rn")
    soft_pts = [rn] + [_agg(f"nogate_dist_softcvar_a025_r{t}")
                       for t in ("15", "20", "30", "50")]
    soft_pts = [p for p in soft_pts if p] + [_agg("nogate_uf_cvar_a025")]
    soft = {k: np.array([p[k] for p in soft_pts]) for k in ("rpa", "pp", "hof")}
    dqn = _agg("nogate_final_dqn", "dqn")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, xlab in (
        (axes[0], "pp", "ping-pong rate (PP / min)"),
        (axes[1], "hof", "HO-failure rate (HOF / min)"),
    ):
        ax.plot(q1[key], q1["rpa"], "-o", color="#ff7f0e", lw=1.8,
                label="single-quantile (τ=α/2)")
        ax.plot(cvar[key], cvar["rpa"], "-s", color="#d62728", lw=1.8,
                label="hard-CVaR (k=⌈25α⌉)")
        ax.plot(soft[key], soft["rpa"], "-^", color="#2ca02c", lw=1.8,
                label="soft-step weights (α=.25, r↑)")
        ax.scatter([rn[key]], [rn["rpa"]], marker="*", s=240, color="#1f77b4",
                   zorder=5, label="RN")
        ax.scatter([dqn[key]], [dqn["rpa"]], marker="*", s=240,
                   color="#555555", zorder=5, label="DQN")
        # Annotate alpha on the Q1 dial.
        for a, x, y in zip(_GRID, q1[key], q1["rpa"]):
            ax.annotate(f"{a:.2f}", (x, y), textcoords="offset points",
                        xytext=(4, -9), fontsize=7.5, color="#ff7f0e")
        ax.set_xlabel(xlab)
        ax.set_ylabel("reward / action")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Reward vs ping-pong")
    axes[1].set_title("Reward vs HO-failure")
    axes[0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle(
        "Risk frontier: efficient = upper-left  "
        "(single-quantile & soft-step dominate hard-CVaR)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = "notes/figures/risk_frontier.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
