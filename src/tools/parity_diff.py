"""Direct apples-to-apples comparison of legacy_simulation vs ltm_gym+LTMBaselineAgent.

Runs the same UE with the same NumPy seed in both engines, captures the full
ReservedBSSectors / ServingBSSector / MCS / HO_event / RLF / HOF / ping_pong
arrays, and reports per-step divergences plus aggregate metric differences.
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scipy.io import loadmat
from scipy.signal import lfilter

# Legacy module (we import the helper functions but drive the loop ourselves
# so we can swap the channel array if needed and capture all internal state).
import src.distrl.envs.legacy_simulation as legacy
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.agents.standard.ltm_baseline import LTMBaselineAgent
from src.distrl.utils.config import Config


def run_legacy_ue(ue_id: int):
    """Replicates legacy.run_simulation() inner loop for a single UE id."""
    System = legacy.System
    Time = legacy.Time
    BS = legacy.BS
    HO = legacy.HO
    NBS = BS["Number"] * 3
    ChannelDirectory = legacy.ChannelDirectory
    ReceiverSensitivity = legacy.ReceiverSensitivity
    MCSEvaluation = legacy.MCSEvaluation
    CheckHO_Failure = legacy.CheckHO_Failure
    Time_PingPong = legacy.Time_PingPong
    Time_InitialSync_0 = legacy.Time_InitialSync_0
    Time_MeasReportL3_1 = legacy.Time_MeasReportL3_1
    Time_RRCTransfer2 = legacy.Time_RRCTransfer2
    Time_RRCConf3 = legacy.Time_RRCConf3
    Time_RRCReconf4_5 = legacy.Time_RRCReconf4_5
    Time_MeasReportL1_67 = legacy.Time_MeasReportL1_67
    Time_HOdecision_8 = legacy.Time_HOdecision_8
    Time_LLHOCommand_9 = legacy.Time_LLHOCommand_9
    Time_RA_10 = legacy.Time_RA_10
    Time_ContextRelease_11 = legacy.Time_ContextRelease_11

    np.random.seed(42 + ue_id)
    filename = os.path.join(ChannelDirectory, f"ChannelGainBSUE_User{ue_id}.mat")
    mat_data = loadmat(filename)
    Channel = mat_data['ChannelBS2UE_noRIS']

    ChBS2UE = np.zeros((NBS, Channel.shape[0]))
    idx = 0
    Max_iter = Channel.shape[0]
    for b in range(Channel.shape[1]):
        for s in range(Channel.shape[2]):
            ChBS2UE[idx, :] = Channel[:, b, s]
            idx += 1

    M = int(np.ceil(HO["Prep"]["PeriodicityRSRPMeasurement"] / Time["TimeStep"]))
    b_fir = np.ones(HO["Prep"]["AverageRSRPMeasument_NL1"]) / HO["Prep"]["AverageRSRPMeasument_NL1"]
    L1 = lfilter(b_fir, 1, ChBS2UE[:, ::M], axis=1)
    L3 = lfilter(HO["Prep"]["alphaIIRfilter"], [1, -1 + HO["Prep"]["alphaIIRfilter"]], L1, axis=1)
    PL1 = np.repeat(L1, M, axis=1)[:, :ChBS2UE.shape[1]]
    PL3 = np.repeat(L3, M, axis=1)[:, :ChBS2UE.shape[1]]

    ServingBSSector = np.full(Max_iter, -1, dtype=int)
    MCS = np.zeros(Max_iter)
    HOF = np.zeros(Max_iter)
    RLF = np.zeros(Max_iter)
    HO_event = np.zeros(Max_iter)
    ping_pong = np.zeros(Max_iter)
    ReservedBSSectors = np.zeros((NBS, Max_iter))

    Sync = {"N310": 4, "N311": 2, "T310": 50,
            "out_sync_count": 0, "in_sync_count": 0,
            "t310_running": False, "t310_counter": np.inf}

    ListBSPrepared = np.zeros(NBS, dtype=int)
    TimerEntering = np.zeros(NBS)
    TimerLeaving = np.zeros(NBS)

    t = 0
    NextBSSector = -1
    former_BS = -1
    former_HO_time = -np.inf

    while t < (Max_iter - 10):
        ReservedBSSectors[:, t] = ListBSPrepared
        if ServingBSSector[t] >= 0 and not RLF[t]:
            MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
            if RLF[t]:
                NextBSSector = -1
        t += 1

        while NextBSSector == -1:
            Pbest = np.max(ChBS2UE[:, t])
            Best = np.argmax(ChBS2UE[:, t])
            if Pbest + System["TxPower"] > ReceiverSensitivity:
                NextBSSector = Best
                MCS[t], RLF[t], Sync = MCSEvaluation(NextBSSector, ChBS2UE[:, t], System, Sync)
                if MCS[t] == 0:
                    NextBSSector = -1
                else:
                    Sync.update({"out_sync_count": 0, "in_sync_count": 0,
                                 "t310_running": False, "t310_counter": np.inf})
            t += 1

        ServingBSSector[t] = NextBSSector
        ReservedBSSectors[:, t] = ListBSPrepared
        MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
        if RLF[t]:
            NextBSSector = -1
            continue

        PL3_report = PL3[:, t]
        Tf = min(t + int(np.ceil(Time_MeasReportL3_1 / Time["TimeStep"])), Max_iter - 1)
        while t < Tf:
            ReservedBSSectors[:, t] = ListBSPrepared
            MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
            if RLF[t]:
                NextBSSector = -1
            t += 1
            ServingBSSector[t] = ServingBSSector[t - 1]
        if RLF[t]:
            NextBSSector = -1
            continue

        Tf = min(t + int(np.ceil((Time_RRCTransfer2 + Time_RRCConf3) / Time["TimeStep"])), Max_iter - 1)
        while t < Tf:
            ReservedBSSectors[:, t] = ListBSPrepared
            MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
            if RLF[t]:
                NextBSSector = -1
            t += 1
            ServingBSSector[t] = ServingBSSector[t - 1]
        if RLF[t]:
            NextBSSector = -1
            continue

        In_condition = PL3_report > (PL3_report[ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
        TimerEntering = (TimerEntering + Time["TimeStep"]) * In_condition
        ListBSPrepared = np.logical_or(ListBSPrepared, (TimerEntering > HO["Prep"]["PreparationTime"]))

        Out_condition = PL3_report < (PL3_report[ServingBSSector[t]] + HO["Prep"]["PreparationPowerOffset"])
        TimerLeaving = (TimerLeaving + Time["TimeStep"]) * Out_condition
        ListBSPrepared = np.logical_and(ListBSPrepared, ~(TimerLeaving > HO["Prep"]["PreparationTime"]))

        if np.sum(ListBSPrepared) > HO["Prep"]["MaxNumberPreparedBS"]:
            metric = (10 ** (PL3_report / 10)) * ListBSPrepared
            I_sorted = np.argsort(metric)[::-1]
            ListBSPrepared[I_sorted[HO["Prep"]["MaxNumberPreparedBS"]:]] = 0

        Tf = min(t + int(np.ceil(Time_RRCReconf4_5 / Time["TimeStep"])), Max_iter - 1)
        while t < Tf:
            ReservedBSSectors[:, t] = ListBSPrepared
            MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
            if RLF[t]:
                NextBSSector = -1
            t += 1
            ServingBSSector[t] = ServingBSSector[t - 1]
        if RLF[t]:
            NextBSSector = -1
            continue

        PL1_report = PL1[:, t]
        HO_condition = np.logical_and(ListBSPrepared, PL1_report > (PL1_report[ServingBSSector[t]] + HO["Prep"]["ExecPowerOffset"]))

        Tf = min(t + int(np.ceil((Time_MeasReportL1_67 + Time_HOdecision_8) / Time["TimeStep"])), Max_iter - 1)
        while t < Tf:
            MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
            ReservedBSSectors[:, t] = ListBSPrepared
            if RLF[t]:
                NextBSSector = -1
            t += 1
            ServingBSSector[t] = ServingBSSector[t - 1]
        if RLF[t]:
            NextBSSector = -1
            continue

        if np.sum(HO_condition) > 0:
            t0 = t
            HO_event[t0] = 1
            metric = (10 ** (PL1_report / 10)) * HO_condition
            I = np.argmax(metric)

            Tf = min(t + int(np.ceil(Time_LLHOCommand_9 / Time["TimeStep"])), Max_iter - 1)
            while t < Tf:
                ReservedBSSectors[:, t] = ListBSPrepared
                MCS[t], RLF[t], Sync = MCSEvaluation(ServingBSSector[t], ChBS2UE[:, t], System, Sync)
                if RLF[t]:
                    break
                t += 1
                ServingBSSector[t] = ServingBSSector[t - 1]
            if RLF[t]:
                NextBSSector = -1
                continue

            t = min(t + int(np.ceil(Time_RA_10 / Time["TimeStep"])), Max_iter - 1)
            HOF[t] = CheckHO_Failure(I, ChBS2UE[:, t], System)

            if HOF[t] == 0:
                NextBSSector = I
                t = min(t + int(np.ceil(Time_ContextRelease_11 / Time["TimeStep"])), Max_iter - 1)
                Sync.update({"out_sync_count": 0, "in_sync_count": 0,
                             "t310_running": False, "t310_counter": np.inf})
                ReservedBSSectors[:, t0:t + 1] = np.tile(ListBSPrepared.reshape(-1, 1), (1, t - t0 + 1))
                ListBSPrepared[NextBSSector] = 0
            else:
                NextBSSector = -1

            ping_pong[t0] = (NextBSSector == former_BS) and ((t0 - former_HO_time) < int(np.floor(Time_PingPong / Time["TimeStep"])))
            former_BS = ServingBSSector[t0]
            former_HO_time = t0

    return dict(
        serving=ServingBSSector,
        mcs=MCS,
        rlf=RLF,
        ho=HO_event,
        hof=HOF,
        pp=ping_pong,
        reserved=ReservedBSSectors,
    )


def run_gym_ue(ue_id: int):
    Config.set_config_path("configs/config.yaml")
    config = Config.get()
    config['simulation']['ue_number'] = 1000  # so all files load; we'll skip to the right one
    env = LTMEnv(config=config)

    # Make the env point at the requested UE
    target = None
    for i, f in enumerate(env.files):
        if f.endswith(f"User{ue_id}_precomputed.npz"):
            target = i
            break
    if target is None:
        raise FileNotFoundError(f"User{ue_id}_precomputed.npz not found")
    env.current_ue_idx = target

    np.random.seed(42 + ue_id)
    agent = LTMBaselineAgent(config, env.observation_space, env.action_space)
    obs, info = env.reset()
    agent.reset()
    done = False
    while not done:
        obs, reward, done, truncated, info = env.step(0, high_res_callback=agent.select_action)
    m = info["metrics"]
    return dict(
        serving=m["serving"],
        mcs=m["mcs"],
        rlf=m["rlf"],
        ho=m["ho"],
        hof=m["hof"],
        pp=m["pp"],
        reserved=m["reserved"],
    )


def summarise(name, d):
    NBS, T = d["reserved"].shape
    minutes = (T * 0.01) / 60.0
    return dict(
        engine=name,
        ho_rate=d["ho"].sum() / minutes,
        hof_rate=d["hof"].sum() / minutes,
        pp_rate=d["pp"].sum() / minutes,
        capacity_avg=float(d["mcs"].mean()),
        rlf_rate=d["rlf"].sum() / minutes,
        reliability_pct=100 - float((d["mcs"] == 0).mean()) * 100,
        prep_sum=float(d["reserved"].sum()),
        res_reservation_pct=float(d["reserved"].sum()) / (NBS * T) * 100,
        prep_rate=(float(d["reserved"].sum()) / 10.0) / minutes,
        serving_changes=int((np.diff(d["serving"]) != 0).sum()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ue", type=int, default=1)
    p.add_argument("--show-diffs", type=int, default=0,
                   help="print this many divergence rows")
    args = p.parse_args()

    print(f"Running legacy for UE{args.ue}...")
    leg = run_legacy_ue(args.ue)
    print(f"Running gym for UE{args.ue}...")
    gym = run_gym_ue(args.ue)

    ls = summarise("legacy", leg)
    gs = summarise("gym", gym)

    print()
    print(f"{'metric':<22} {'legacy':>14} {'gym':>14} {'diff':>14} {'%':>8}")
    for k in ["ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
              "reliability_pct", "prep_sum", "res_reservation_pct",
              "prep_rate", "serving_changes"]:
        lv = ls[k]; gv = gs[k]
        diff = gv - lv
        pct = (diff / lv * 100) if lv else 0.0
        print(f"{k:<22} {lv:>14.4f} {gv:>14.4f} {diff:>14.4f} {pct:>7.2f}%")

    if args.show_diffs:
        rsum_legacy = leg["reserved"].sum(axis=0)
        rsum_gym = gym["reserved"].sum(axis=0)
        diff_mask = rsum_legacy != rsum_gym
        diff_idx = np.where(diff_mask)[0]
        print(f"\nReservedBSSectors row-sum diffs: {diff_mask.sum()} / {len(rsum_legacy)} ticks")
        if len(diff_idx) > 0:
            print(f"First {args.show_diffs} diverging ticks:")
            print(f"{'t':>6} {'leg_sum':>8} {'gym_sum':>8} {'leg_serv':>9} {'gym_serv':>9} {'leg_ho':>7} {'gym_ho':>7}")
            for t in diff_idx[: args.show_diffs]:
                print(f"{t:>6} {int(rsum_legacy[t]):>8} {int(rsum_gym[t]):>8} {leg['serving'][t]:>9} {gym['serving'][t]:>9} {int(leg['ho'][t]):>7} {int(gym['ho'][t]):>7}")


if __name__ == "__main__":
    main()
