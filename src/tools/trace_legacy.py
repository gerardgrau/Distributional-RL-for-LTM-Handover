import numpy as np
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import src.distrl.envs.legacy_simulation as legacy

def trace_legacy_ue1():
    orig_ue_num = legacy.UE_Number
    legacy.UE_Number = 1
    perf_all, metrics_all = legacy.run_simulation()
    legacy.UE_Number = orig_ue_num
    
    m = metrics_all[0]
    serving = m["ServingBSSector"]
    mcs = m["MCS"]
    ho = m["HO_event"]
    rlf = m["RLF"]
    # ReservedBSSectors is not in Metrics but we can access it from run_simulation return
    # Actually metrics_all has UE-specific arrays
    # Let's modify run_simulation in legacy_simulation.py to return ReservedBSSectors too
    # Or better, just read it from Metrics.pkl if it's there?
    # Wait, Metrics in legacy_simulation.py is a list of dicts.
    
    with open("results/simulations/Metrics.pkl", "rb") as f:
        metrics_pkl = pickle.load(f)
    
    with open("results/simulations/Performance_all.pkl", "rb") as f:
        perf_pkl = pickle.load(f)
        
    # UE 1 is index 0
    m_ue1 = metrics_pkl[0]
    p_ue1 = perf_pkl[0]
    
    # Legacy doesn't store ReservedBSSectors in Metrics.pkl by default.
    # Let's check Performance_all.pkl for Resource_reservation
    print(f"Legacy UE 1 Res Reservation: {p_ue1['Resource_reservation']}%")
    
    with open("legacy_trace_ue1.pkl", "wb") as f:
        pickle.dump({
            "serving": m_ue1["ServingBSSector"], 
            "mcs": m_ue1["MCS"], 
            "ho": m_ue1["HO_event"], 
            "rlf": m_ue1["RLF"],
            "reserved": m_ue1.get("ReservedBSSectors", None)
        }, f)
    print("Saved legacy trace to legacy_trace_ue1.pkl")

if __name__ == "__main__":
    trace_legacy_ue1()
