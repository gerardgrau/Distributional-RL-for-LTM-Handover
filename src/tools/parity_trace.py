"""Find the first tick where ListBSPrepared diverges between legacy and gym."""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.tools.parity_diff import run_legacy_ue, run_gym_ue


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ue", type=int, default=1)
    p.add_argument("--window", type=int, default=15)
    args = p.parse_args()

    leg = run_legacy_ue(args.ue)
    gym = run_gym_ue(args.ue)

    # Find first divergence in row-sum of ReservedBSSectors.
    leg_sum = leg["reserved"].sum(axis=0)
    gym_sum = gym["reserved"].sum(axis=0)
    diff = leg_sum != gym_sum
    if not diff.any():
        print("No divergence!")
        return

    first = int(np.where(diff)[0][0])
    print(f"First divergence at t={first}")
    lo = max(0, first - args.window)
    hi = min(len(leg_sum), first + args.window)
    print(f"\nContext [{lo}..{hi}):")
    print(f"{'t':>6} | {'leg_serv':>8} {'gym_serv':>8} | {'leg_sum':>7} {'gym_sum':>7} | "
          f"{'leg_ho':>6} {'gym_ho':>6} | {'leg_mcs':>8} {'gym_mcs':>8} | "
          f"{'leg_rlf':>7} {'gym_rlf':>7}")
    for t in range(lo, hi):
        mark = " <--" if diff[t] else ""
        print(f"{t:>6} | {leg['serving'][t]:>8} {gym['serving'][t]:>8} | "
              f"{int(leg_sum[t]):>7} {int(gym_sum[t]):>7} | "
              f"{int(leg['ho'][t]):>6} {int(gym['ho'][t]):>6} | "
              f"{leg['mcs'][t]:>8.4f} {gym['mcs'][t]:>8.4f} | "
              f"{int(leg['rlf'][t]):>7} {int(gym['rlf'][t]):>7}{mark}")

    print(f"\nLegacy prepared cells at t={first}: {np.where(leg['reserved'][:, first])[0].tolist()}")
    print(f"Gym    prepared cells at t={first}: {np.where(gym['reserved'][:, first])[0].tolist()}")


if __name__ == "__main__":
    main()
