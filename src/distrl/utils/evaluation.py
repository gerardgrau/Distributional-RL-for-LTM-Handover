import os
import pandas as pd
from tqdm import tqdm
from typing import Any
from src.distrl.envs.ltm_gym import LTMEnv
from src.distrl.utils.metrics import calculate_8_metrics

def run_evaluation(
    agent: Any, 
    config: dict, 
    experiment_dir: str, 
    agent_type: str, 
    seed: int, 
    save_results: bool = True
) -> dict[str, float]:
    """
    Evaluates the frozen agent on ALL trajectories (1000 UEs) with epsilon=0.
    """
    print(f"    -> Starting Formal Evaluation Phase (Seed {seed})...")
    
    # Use a fresh environment instance to ensure deterministic evaluation over all 1000 users
    eval_env = LTMEnv(config=config)
    num_eval_episodes = len(eval_env.files)
    
    if num_eval_episodes == 0:
        print("    -> Skipping Evaluation: No trajectories found.")
        eval_env.close()
        return {}

    all_eval_metrics = []
    
    for ep in tqdm(range(num_eval_episodes), desc=f"    Eval {agent_type}", leave=False):
        state, _ = eval_env.reset()
        done = False
        last_info = {}
        episode_reward = 0
        
        while not done:
            # Pure greedy selection (Frozen weights, No exploration)
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, _, info = eval_env.step(action)
            episode_reward += reward
            if done:
                last_info = info
                
        m8 = calculate_8_metrics(
            mcs_history=last_info["metrics"]["mcs"],
            rlf_history=last_info["metrics"]["rlf"],
            ho_history=last_info["metrics"]["ho"],
            hof_history=last_info["metrics"]["hof"],
            pp_history=last_info["metrics"]["pp"],
            serving_history=last_info["metrics"]["serving"],
            pl3_history=last_info["metrics"]["pl3"],
            config=config
        )
        m8['reward'] = episode_reward
        all_eval_metrics.append(m8)
        
    eval_env.close()
    
    # Aggregate results
    df = pd.DataFrame(all_eval_metrics)
    summary = {
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict()
    }
    
    # Save to files
    if save_results:
        eval_dir = os.path.join(experiment_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 1. Save Summary CSV (Metric, Mean, Std)
        summary_csv = os.path.join(eval_dir, f"{agent_type}_summary_seed{seed}.csv")
        summary_df = pd.DataFrame({
            "metric": summary["mean"].keys(),
            "mean": summary["mean"].values(),
            "std": summary["std"].values()
        })
        summary_df.to_csv(summary_csv, index=False)
            
        # 2. Save Raw CSV (Per-episode metrics)
        raw_csv = os.path.join(eval_dir, f"{agent_type}_raw_seed{seed}.csv")
        df.to_csv(raw_csv, index_label="eval_episode")
        
    print(f"    -> Evaluation Complete. HO Rate: {summary['mean']['ho_rate']:.2f} ± {summary['std']['ho_rate']:.2f}")
    return summary['mean']
