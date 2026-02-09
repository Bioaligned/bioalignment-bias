#!/usr/bin/env python3
"""
Capability Benchmarking for Bioaligned Models

Compares base model vs fine-tuned adapter on standard benchmarks
to ensure no capability degradation ("alignment tax").

Usage:
    python benchmark_capabilities.py --adapter_path ./bioaligned-llama3-3b-v3/best

Requires: pip install lm-eval
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

TASKS = [
    "mmlu",           # General knowledge (57 subjects)
    "hellaswag",      # Commonsense reasoning
    "arc_easy",       # Science questions (easy)
    "arc_challenge",  # Science questions (hard)
    "winogrande",     # Coreference resolution
    "gsm8k",          # Math word problems
]

def run_eval(model_args: str, output_name: str, tasks: list[str] = TASKS):
    """Run lm-eval-harness and return results."""
    
    task_str = ",".join(tasks)
    output_path = f"./benchmark_results/{output_name}.json"
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", task_str,
        "--batch_size", "4",
        "--output_path", output_path,
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Load and return results
    with open(output_path) as f:
        return json.load(f)


def compare_results(base_results: dict, adapter_results: dict):
    """Compare and display results."""
    
    print("\n" + "="*70)
    print("CAPABILITY COMPARISON: Base vs Bioaligned")
    print("="*70)
    print(f"{'Task':<20} {'Base':>12} {'Bioaligned':>12} {'Delta':>12} {'Status':>10}")
    print("-"*70)
    
    degradation_threshold = 0.02  # 2% drop is concerning
    
    for task in TASKS:
        base_acc = base_results.get("results", {}).get(task, {}).get("acc", 0)
        adapter_acc = adapter_results.get("results", {}).get(task, {}).get("acc", 0)
        
        # Handle normalized accuracy for some tasks
        if base_acc == 0:
            base_acc = base_results.get("results", {}).get(task, {}).get("acc_norm", 0)
            adapter_acc = adapter_results.get("results", {}).get(task, {}).get("acc_norm", 0)
        
        delta = adapter_acc - base_acc
        
        if delta < -degradation_threshold:
            status = "DEGRADED"
        elif delta > degradation_threshold:
            status = "IMPROVED"
        else:
            status = "STABLE"
        
        print(f"{task:<20} {base_acc:>12.3f} {adapter_acc:>12.3f} {delta:>+12.3f} {status:>10}")
    
    print("-"*70)
    
    # Summary statistics
    base_avg = sum(base_results.get("results", {}).get(t, {}).get("acc", 0) or 
                   base_results.get("results", {}).get(t, {}).get("acc_norm", 0) 
                   for t in TASKS) / len(TASKS)
    adapter_avg = sum(adapter_results.get("results", {}).get(t, {}).get("acc", 0) or
                      adapter_results.get("results", {}).get(t, {}).get("acc_norm", 0)
                      for t in TASKS) / len(TASKS)
    
    print(f"{'AVERAGE':<20} {base_avg:>12.3f} {adapter_avg:>12.3f} {adapter_avg-base_avg:>+12.3f}")
    print("="*70)
    
    if adapter_avg >= base_avg - 0.01:
        print("\nPASS: No significant capability degradation detected.")
        print("   The bioaligned model preserves general capabilities.")
    else:
        print("\nWARNING: Potential capability degradation detected.")
        print("   Consider adjusting training parameters.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--skip_base", action="store_true", 
                        help="Skip base model eval if already run")
    args = parser.parse_args()
    
    Path("./benchmark_results").mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluate base model
    if not args.skip_base:
        print("\n" + "="*70)
        print("EVALUATING BASE MODEL")
        print("="*70)
        base_results = run_eval(
            f"pretrained={args.base_model}",
            f"base_{timestamp}",
            args.tasks
        )
    else:
        # Load most recent base results
        base_files = sorted(Path("./benchmark_results").glob("base_*.json"))
        if not base_files:
            raise ValueError("No base results found. Run without --skip_base first.")
        with open(base_files[-1]) as f:
            base_results = json.load(f)
    
    # Evaluate adapter
    print("\n" + "="*70)
    print("EVALUATING BIOALIGNED MODEL")
    print("="*70)
    adapter_results = run_eval(
        f"pretrained={args.base_model},peft={args.adapter_path}",
        f"bioaligned_{timestamp}",
        args.tasks
    )
    
    # Compare
    compare_results(base_results, adapter_results)
    
    # Save comparison
    comparison = {
        "timestamp": timestamp,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "base_results": base_results,
        "adapter_results": adapter_results,
    }
    
    with open(f"./benchmark_results/comparison_{timestamp}.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nResults saved to ./benchmark_results/comparison_{timestamp}.json")


if __name__ == "__main__":
    main()
