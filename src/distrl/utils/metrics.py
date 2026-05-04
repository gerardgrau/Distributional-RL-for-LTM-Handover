import numpy as np
from typing import Any, Dict

def calculate_8_metrics(
    mcs_history: np.ndarray,
    rlf_history: np.ndarray,
    ho_history: np.ndarray,
    hof_history: np.ndarray,
    pp_history: np.ndarray,
    serving_history: np.ndarray,
    pl3_history: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Computes the 8 LTM metrics based on the provided episode history.
    """
    total_steps = len(mcs_history)
    time_step = config['simulation']['time_step']
    minutes = (total_steps * time_step) / 60.0
    
    # 1. Capacity (bps) - Using average MCS as a proxy for spectral efficiency/capacity
    avg_capacity = np.mean(mcs_history)
    
    # 2. RLF Rate
    rlf_total = np.sum(rlf_history)
    rlf_rate = rlf_total / minutes if minutes > 0 else 0.0
    
    # 3. HO Rate
    ho_total = np.sum(ho_history)
    ho_rate = ho_total / minutes if minutes > 0 else 0.0
    
    # 4. PP Rate
    pp_total = np.sum(pp_history)
    pp_rate = pp_total / minutes if minutes > 0 else 0.0
    
    # 5. Reliability (%)
    # Measured as the ratio of total outage time to total service time.
    reliability = 100.0 - (np.mean(mcs_history == 0) * 100.0)
    
    # 6 & 7. Cell Preparation and Resource Reservation
    # We need to simulate the ListBSPrepared logic here
    prep_cfg = config['ho_prep']
    nbs = pl3_history.shape[0]
    
    reserved_bs_sectors = np.zeros((nbs, total_steps))
    list_bs_prepared = np.zeros(nbs, dtype=bool)
    timer_entering = np.zeros(nbs)
    timer_leaving = np.zeros(nbs)
    
    prep_offset = prep_cfg['preparation_power_offset']
    prep_time_thresh = prep_cfg['preparation_time']
    max_prep = prep_cfg['max_number_prepared_bs']
    
    for t in range(total_steps):
        serving = int(serving_history[t])
        if serving == -1:
            list_bs_prepared[:] = False
            timer_entering[:] = 0
            timer_leaving[:] = 0
        else:
            pl3_report = pl3_history[:, t]
            # Cell preparation entering condition
            in_condition = pl3_report > (pl3_report[serving] + prep_offset)
            timer_entering = (timer_entering + time_step) * in_condition
            list_bs_prepared = np.logical_or(list_bs_prepared, (timer_entering > prep_time_thresh))

            # Cell preparation leaving condition
            out_condition = pl3_report < (pl3_report[serving] + prep_offset)
            timer_leaving = (timer_leaving + time_step) * out_condition
            list_bs_prepared = np.logical_and(list_bs_prepared, ~(timer_leaving > prep_time_thresh))

            # Reduce list if too long
            if np.sum(list_bs_prepared) > max_prep:
                metric = (10 ** (pl3_report / 10.0)) * list_bs_prepared
                i_sorted = np.argsort(metric)[::-1]
                list_bs_prepared[i_sorted[max_prep:]] = False
        
        # If a HO occurred at this step (and was successful), clear the prepared bit for the NEW serving cell
        # This matches ltm_env.py: ListBSPrepared[NextBSSector] = 0
        if ho_history[t] > 0 and serving != -1:
            list_bs_prepared[serving] = False
            
        reserved_bs_sectors[:, t] = list_bs_prepared

    # Number of cell preparation events
    # We count whenever a bit in ListBSPrepared flips from 0 to 1
    # Note: original code used (ReservedBSSectors[:, 1:] - ReservedBSSectors[:, :-1]) >= 0 which seems wrong?
    # It should be > 0.
    num_preps = np.sum((reserved_bs_sectors[:, 1:] > reserved_bs_sectors[:, :-1]))
    prep_rate = num_preps / minutes if minutes > 0 else 0.0
    
    # Resource Reservation (%)
    resource_reservation = (np.sum(reserved_bs_sectors) / reserved_bs_sectors.size) * 100.0
    
    # 8. HOF Rate
    hof_total = np.sum(hof_history)
    hof_rate = hof_total / minutes if minutes > 0 else 0.0
    
    return {
        "capacity_avg": float(avg_capacity),
        "rlf_rate": float(rlf_rate),
        "ho_rate": float(ho_rate),
        "pp_rate": float(pp_rate),
        "reliability_pct": float(reliability),
        "prep_rate": float(prep_rate),
        "res_reservation_pct": float(resource_reservation),
        "hof_rate": float(hof_rate),
        "total_steps": int(total_steps),
        "total_minutes": float(minutes)
    }
