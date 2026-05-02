import time
import cProfile
import pstats
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.distrl.envs.ltm_gym import LTMEnv

def run_simulation(env, steps):
    for i in range(10): # Reset 10 times to test caching
        start_reset = time.time()
        state, _ = env.reset()
        print(f"Reset {i+1} took {time.time() - start_reset:.4f}s")
        for _ in range(steps // 10):
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            if done:
                break

def main():
    print("Initializing environment...")
    config = {'simulation': {'ue_number': 2}}
    env = LTMEnv(config=config)
    
    steps = 1000
    print(f"Running {steps} steps profiling...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    run_simulation(env, steps)
    end_time = time.time()
    
    profiler.disable()
    
    elapsed = end_time - start_time
    steps_per_sec = steps / elapsed
    
    print(f"\n--- Performance Baseline ---")
    print(f"Total time for {steps} steps: {elapsed:.4f} seconds")
    print(f"Speed: {steps_per_sec:.2f} steps/second")
    print(f"----------------------------\n")
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    print("Top 20 time-consuming functions:")
    stats.print_stats(20)

if __name__ == "__main__":
    main()
