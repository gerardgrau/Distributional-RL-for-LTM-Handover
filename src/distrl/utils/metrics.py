import numpy as np
from typing import Any

def calculate_8_metrics(
    mcs_history: np.ndarray,
    rlf_history: np.ndarray,
    ho_history: np.ndarray,
    hof_history: np.ndarray,
    pp_history: np.ndarray,
    serving_history: np.ndarray,
    pl3_history: np.ndarray,
    config: dict[str, Any]
) -> dict[str, Any]:
    """
    Computes the 8 LTM metrics based on the provided episode history using vectorized operations.
    """
    total_steps = len(mcs_history)
    time_step = config['simulation']['time_step']
    minutes = (total_steps * time_step) / 60.0
    
    # 1. Capacity (bps)
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
    reliability = 100.0 - (np.mean(mcs_history == 0) * 100.0)
    
    # 6 & 7. Cell Preparation and Resource Reservation (Vectorized)
    prep_cfg = config['ho_prep']
    nbs = pl3_history.shape[0]
    
    prep_offset = prep_cfg['preparation_power_offset']
    prep_time_thresh = prep_cfg['preparation_time']
    max_prep = prep_cfg['max_number_prepared_bs']
    
    # Vectorized condition check
    # pl3_serving: [total_steps]
    serving_history_int = serving_history.astype(int)
    valid_mask = serving_history_int != -1
    
    pl3_serving = np.zeros(total_steps)
    pl3_serving[valid_mask] = pl3_history[serving_history_int[valid_mask], np.where(valid_mask)[0]]
    
    # in_condition: [nbs, total_steps]
    in_condition = pl3_history > (pl3_serving + prep_offset)
    in_condition[:, ~valid_mask] = False
    
    # Vectorized list_bs_prepared logic
    # We use a stateful simulation but optimized with NumPy where possible.
    # Since ListBSPrepared depends on the previous state, we can't fully vectorize 
    # the entire time loop without more complex logic (like scan), 
    # but we can speed up the per-step operations.
    
    reserved_bs_sectors = np.zeros((nbs, total_steps), dtype=bool)
    list_bs_prepared = np.zeros(nbs, dtype=bool)
    timer_entering = np.zeros(nbs)
    timer_leaving = np.zeros(nbs)
    
    # Optimization: Use a smaller loop only where necessary
    # Or even better: pre-calculate the thresholds
    
    for t in range(total_steps):
        s = serving_history_int[t]
        if s == -1:
            list_bs_prepared[:] = False
            timer_entering[:] = 0
            timer_leaving[:] = 0
        else:
            cond = in_condition[:, t]
            timer_entering = (timer_entering + time_step) * cond
            list_bs_prepared |= (timer_entering > prep_time_thresh)

            timer_leaving = (timer_leaving + time_step) * (~cond)
            list_bs_prepared &= ~(timer_leaving > prep_time_thresh)

            if np.sum(list_bs_prepared) > max_prep:
                # Use power-domain for sorting as per original logic
                metric = (10 ** (pl3_history[:, t] / 10.0)) * list_bs_prepared
                i_sorted = np.argsort(metric)[::-1]
                list_bs_prepared[i_sorted[max_prep:]] = False
        
        if ho_history[t] > 0 and s != -1:
            list_bs_prepared[s] = False
            
        reserved_bs_sectors[:, t] = list_bs_prepared

    num_preps = np.sum((reserved_bs_sectors[:, 1:] > reserved_bs_sectors[:, :-1]))
    prep_rate = num_preps / minutes if minutes > 0 else 0.0
    resource_reservation = (np.sum(reserved_bs_sectors) / (nbs * total_steps)) * 100.0
    
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
