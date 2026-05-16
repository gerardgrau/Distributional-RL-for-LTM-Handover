import numpy as np
from src.distrl.utils.metrics import calculate_8_metrics

def test_metrics():
    print("Testing metrics calculation...")
    total_steps = 1000
    nbs = 21
    mcs = np.random.randint(0, 28, size=total_steps)
    rlf = np.zeros(total_steps)
    ho = np.zeros(total_steps)
    hof = np.zeros(total_steps)
    pp = np.zeros(total_steps)
    serving = np.random.randint(0, nbs, size=total_steps)
    pl3 = np.random.uniform(-120, -30, size=(nbs, total_steps))
    
    config = {
        'simulation': {'time_step': 0.01},
        'ho_prep': {
            'preparation_power_offset': -3.0,
            'preparation_time': 0.04,
            'max_number_prepared_bs': 5
        }
    }
    
    metrics = calculate_8_metrics(mcs, rlf, ho, hof, pp, serving, pl3, config)
    print("Metrics calculated successfully:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    test_metrics()
