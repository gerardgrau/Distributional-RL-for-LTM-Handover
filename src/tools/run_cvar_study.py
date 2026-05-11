import subprocess
import os
import sys
import time

def run_cvar_study():
    risk_fractions = [0.05, 0.1, 0.25, 0.5]
    config_path = "configs/test-quantiles.yaml" # We reuse the ablation base
    device = "xpu" if "--cpu" not in sys.argv else "cpu"
    N = 200 # Optimal N from ablation
    
    print(f"=== Starting Risk-Management (CVaR) Study ===")
    print(f"Quantiles (N): {N}")
    print(f"Risk Fractions: {risk_fractions}")
    print(f"Device: {device}")
    
    results_summary = []
    
    for k in risk_fractions:
        print(f"\n>>> Running QRDQN with CVaR fraction={k}...")
        temp_config = f"configs/temp_cvar_k{k}.yaml"
        
        # Read base config
        with open(config_path, 'r') as f:
            lines = f.readlines()
            
        # Write temp config with risk parameters
        with open(temp_config, 'w') as f:
            agent_section = False
            for line in lines:
                f.write(line)
                if 'agent:' in line:
                    agent_section = True
                if agent_section and 'num_quantiles:' in line:
                    # We already wrote the line, just making sure we override if needed
                    pass 
            
            # Append overrides at the end to be sure
            f.write("\n  # CVaR Overrides\n")
            f.write(f"  num_quantiles: {N}\n")
            f.write(f"  risk_type: \"cvar\"\n")
            f.write(f"  risk_fraction: {k}\n")
        
        cmd = [
            "venv-RL/bin/python3", "src/main.py",
            "--config", temp_config,
            "--description", f"cvar-study-k{k}",
            "--device", device,
            "--agents", "qrdqn"
        ]
        
        start_time = time.time()
        subprocess.run(cmd)
        end_time = time.time()
        
        duration = end_time - start_time
        results_summary.append((k, duration))
        print(f"--- k={k} completed in {duration:.2f} seconds ---")
        
        os.remove(temp_config)
        
    print("\n" + "="*40)
    print("      CVaR STUDY SUMMARY")
    print("="*40)
    print(f"{'Risk Fraction (k)':<20} | {'Duration (s)':<15}")
    print("-" * 38)
    for k, dur in results_summary:
        print(f"{k:<20} | {dur:<15.2f}")
    print("="*40)

if __name__ == "__main__":
    run_cvar_study()
