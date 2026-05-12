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
    
    with open("legacy_trace_ue1.pkl", "wb") as f:
        pickle.dump({"serving": serving, "mcs": mcs, "ho": ho, "rlf": rlf}, f)
    print("Saved legacy trace to legacy_trace_ue1.pkl")

if __name__ == "__main__":
    trace_legacy_ue1()
