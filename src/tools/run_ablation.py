import subprocess
import os
import sys

def run_ablation():
    quantiles = [10, 50]
    config_path = "configs/test-quantiles.yaml"
    device = "xpu" if "--cpu" not in sys.argv else "cpu"
    
    print(f"=== Starting Ablation Study on Quantiles: {quantiles} ===")
    
    for n in quantiles:
        print(f"\n>>> Running QRDQN with N={n}...")
        # We use a temporary config or CLI override if supported.
        # Since main.py doesn't have a --num_quantiles flag, we'll create a temp config.
        temp_config = f"configs/temp_ablation_N{n}.yaml"
        
        with open(config_path, 'r') as f:
            lines = f.readlines()
            
        with open(temp_config, 'w') as f:
            for line in lines:
                if 'num_quantiles:' in line:
                    f.write(f"  num_quantiles: {n}\n")
                else:
                    f.write(line)
        
        cmd = [
            "venv-RL/bin/python3", "src/main.py",
            "--config", temp_config,
            "--description", f"ablation-quantiles-N{n}",
            "--device", device,
            "--agents", "qrdqn"
        ]
        
        subprocess.run(cmd)
        os.remove(temp_config)
        
    print("\n=== Ablation Study Completed ===")

if __name__ == "__main__":
    run_ablation()
