import pandas as pd
import os
import glob

def reorder_summary_csv(file_path):
    print(f"Reordering {file_path}...")
    df = pd.read_csv(file_path)
    
    # Strip whitespace from column names and metric values
    df.columns = [c.strip() for c in df.columns]
    if 'metric' in df.columns:
        df['metric'] = df['metric'].str.strip()
    
    # Base core metrics
    core_metrics = [
        "ho_rate", "hof_rate", "pp_rate", "capacity_avg", "rlf_rate",
        "reliability_pct", "prep_rate", "res_reservation_pct"
    ]
    
    # Extended metrics
    extended_metrics = ["reward", "total_steps", "total_minutes"]
    
    file_name = os.path.basename(file_path)
    is_paper_or_legacy = "paper_" in file_name or "legacy_baseline" in file_name
    
    if is_paper_or_legacy:
        # Filter OUT extended metrics for paper/legacy
        df = df[~df['metric'].isin(extended_metrics)]
        metric_order = core_metrics
    else:
        # Keep them for others
        metric_order = core_metrics + extended_metrics
    
    # Define the final order of categories
    existing_metrics = df['metric'].tolist()
    final_categories = [m for m in metric_order if m in existing_metrics]
    final_categories += [m for m in existing_metrics if m not in metric_order]
    
    df['metric'] = pd.Categorical(df['metric'], categories=final_categories, ordered=True)
    df = df.sort_values('metric')
    
    df.to_csv(file_path, index=False)

def main():
    # 1. Final Metrics
    final_metrics_path = "results/final_metrics/*.csv"
    for f in glob.glob(final_metrics_path):
        reorder_summary_csv(f)
        
    # 2. Benchmark evaluation summaries
    benchmark_summaries = "results/benchmarks/bmk_*/eval/*_summary_seed*.csv"
    for f in glob.glob(benchmark_summaries):
        reorder_summary_csv(f)

if __name__ == "__main__":
    main()
