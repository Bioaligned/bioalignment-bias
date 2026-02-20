#!/usr/bin/env python3
"""
5.4 Capability Preservation - Standard Benchmark Evaluation

Compares base model vs bioaligned adapter on standard benchmarks
to ensure no capability degradation ("alignment tax").

This experiment answers:
- Does bioalignment training hurt general capabilities?
- Which capabilities (if any) are affected?
- Is the alignment tax acceptable for the bias reduction gained?

Usage:
    python run_capability_benchmarks.py \
        --adapter_path Bioaligned/bioaligned-llama3.2-3b-instruct-qlora

    # Or with local adapter:
    python run_capability_benchmarks.py \
        --adapter_path ./path/to/adapter

Output:
    - capability_results.json: Full benchmark results
    - capability_comparison.png: Visual comparison
    - capability_report.md: Analysis summary

Requirements:
    pip install lm-eval
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# Standard benchmarks for capability testing
TASKS = {
    "mmlu": "General knowledge (57 subjects)",
    "hellaswag": "Commonsense reasoning",
    "arc_easy": "Science questions (easy)",
    "arc_challenge": "Science questions (hard)",
    "winogrande": "Coreference resolution",
    "gsm8k": "Math word problems",
}

# Threshold for concerning degradation
DEGRADATION_THRESHOLD = 0.02  # 2%


def run_lm_eval(model_args: str, output_name: str, tasks: list[str], output_dir: Path) -> dict:
    """Run lm-eval-harness and return results."""

    task_str = ",".join(tasks)
    output_path = output_dir / f"{output_name}.json"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", task_str,
        "--batch_size", "4",
        "--output_path", str(output_dir),
        "--log_samples",
    ]

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running lm-eval: {e}")
        print(f"Stderr: {e.stderr}")
        return {"error": str(e)}

    # Find the results file (lm-eval creates timestamped directories)
    result_files = list(output_dir.glob("**/results.json"))
    if result_files:
        # Get most recent
        result_file = max(result_files, key=lambda p: p.stat().st_mtime)
        with open(result_file) as f:
            return json.load(f)

    return {"error": "No results file found"}


def extract_accuracy(results: dict, task: str) -> float:
    """Extract accuracy from lm-eval results."""
    task_results = results.get("results", {}).get(task, {})

    # Try different accuracy keys
    for key in ["acc", "acc_norm", "acc,none", "acc_norm,none"]:
        if key in task_results:
            return task_results[key]

    return None


def plot_comparison(base_results: dict, adapter_results: dict, output_path: str):
    """Create bar chart comparing base vs bioaligned model."""

    tasks = list(TASKS.keys())
    base_accs = [extract_accuracy(base_results, t) or 0 for t in tasks]
    adapter_accs = [extract_accuracy(adapter_results, t) or 0 for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, base_accs, width, label='Base Model', color='#2196F3')
    bars2 = ax.bar(x + width/2, adapter_accs, width, label='Bioaligned', color='#4CAF50')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Capability Preservation: Base vs Bioaligned Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            if val:
                ax.annotate(f'{val:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    add_labels(bars1, base_accs)
    add_labels(bars2, adapter_accs)

    # Mark significant changes
    for i, (b, a) in enumerate(zip(base_accs, adapter_accs)):
        if b and a:
            delta = a - b
            if abs(delta) > DEGRADATION_THRESHOLD:
                color = 'red' if delta < 0 else 'green'
                ax.annotate(f'{delta:+.2f}', xy=(x[i], max(b, a) + 0.05),
                            ha='center', fontsize=9, color=color, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to {output_path}")


def generate_report(base_results: dict, adapter_results: dict, args, output_path: str):
    """Generate markdown analysis report."""

    tasks = list(TASKS.keys())

    # Calculate metrics
    rows = []
    degraded = []
    improved = []

    for task in tasks:
        base_acc = extract_accuracy(base_results, task)
        adapter_acc = extract_accuracy(adapter_results, task)

        if base_acc is not None and adapter_acc is not None:
            delta = adapter_acc - base_acc

            if delta < -DEGRADATION_THRESHOLD:
                status = "DEGRADED"
                degraded.append(task)
            elif delta > DEGRADATION_THRESHOLD:
                status = "IMPROVED"
                improved.append(task)
            else:
                status = "STABLE"

            rows.append({
                "task": task,
                "description": TASKS[task],
                "base": base_acc,
                "adapter": adapter_acc,
                "delta": delta,
                "status": status,
            })

    # Calculate averages
    base_avg = np.mean([r["base"] for r in rows])
    adapter_avg = np.mean([r["adapter"] for r in rows])
    avg_delta = adapter_avg - base_avg

    # Overall assessment
    if len(degraded) == 0 and avg_delta >= -0.01:
        verdict = "PASS - No significant capability degradation detected."
        verdict_detail = "The bioaligned model preserves general capabilities."
    elif len(degraded) <= 1 and avg_delta >= -0.02:
        verdict = "MARGINAL - Minor degradation in some areas."
        verdict_detail = f"Degradation observed in: {', '.join(degraded)}"
    else:
        verdict = "FAIL - Significant capability degradation detected."
        verdict_detail = f"Consider adjusting training. Degraded: {', '.join(degraded)}"

    report = f"""# 5.4 Capability Preservation Results

## Summary

**Verdict: {verdict}**

{verdict_detail}

### Key Metrics

| Metric | Value |
|--------|-------|
| Base Model | {args.base_model} |
| Bioaligned Adapter | {args.adapter_path} |
| Benchmarks Run | {len(rows)} |
| Degraded Tasks | {len(degraded)} |
| Improved Tasks | {len(improved)} |
| Average Change | {avg_delta:+.3f} ({avg_delta*100:+.1f}%) |

## Detailed Results

| Benchmark | Description | Base | Bioaligned | Delta | Status |
|-----------|-------------|------|------------|-------|--------|
"""

    for r in rows:
        report += f"| {r['task'].upper()} | {r['description']} | {r['base']:.3f} | {r['adapter']:.3f} | {r['delta']:+.3f} | {r['status']} |\n"

    report += f"| **AVERAGE** | | **{base_avg:.3f}** | **{adapter_avg:.3f}** | **{avg_delta:+.3f}** | |\n"

    report += f"""
## Interpretation

The bioalignment fine-tuning achieved 93% reduction in anti-biological source bias
(Δp_up: -0.141 → -0.010) while {"maintaining" if len(degraded) == 0 else "mostly maintaining"} general capabilities.

### Trade-off Analysis

- **Bioalignment Gain:** +0.131 Δp_up improvement
- **Capability Cost:** {avg_delta:+.3f} average accuracy change
- **Trade-off Ratio:** {abs(0.131 / avg_delta):.1f}x bioalignment gain per point of capability change

### Benchmark Details

"""

    for task, desc in TASKS.items():
        report += f"- **{task.upper()}**: {desc}\n"

    report += f"""
## Methodology

- Evaluation framework: lm-evaluation-harness
- Batch size: 4
- Comparison: Base model vs PEFT adapter merged
- Degradation threshold: >{DEGRADATION_THRESHOLD*100:.0f}% drop flagged

---
*Generated: {datetime.now().isoformat()}*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    return verdict, avg_delta


def main():
    parser = argparse.ArgumentParser(description="5.4 Capability Preservation Benchmarks")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base model ID")
    parser.add_argument("--adapter_path", required=True,
                        help="Path or HuggingFace ID of bioaligned adapter")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        help="Benchmarks to run")
    parser.add_argument("--output_dir", default=".",
                        help="Output directory")
    parser.add_argument("--skip_base", action="store_true",
                        help="Skip base model eval (use existing results)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if lm-eval is available
    try:
        subprocess.run(["lm_eval", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: lm-eval not found. Install with: pip install lm-eval")
        sys.exit(1)

    # Evaluate base model
    if not args.skip_base:
        print("\n" + "="*60)
        print("EVALUATING BASE MODEL")
        print("="*60)

        base_results = run_lm_eval(
            f"pretrained={args.base_model}",
            f"base_{timestamp}",
            args.tasks,
            output_dir / "base_eval"
        )

        # Save base results
        with open(output_dir / f"base_results_{timestamp}.json", 'w') as f:
            json.dump(base_results, f, indent=2)
    else:
        # Load most recent base results
        base_files = sorted(output_dir.glob("base_results_*.json"))
        if not base_files:
            print("Error: No base results found. Run without --skip_base first.")
            sys.exit(1)
        with open(base_files[-1]) as f:
            base_results = json.load(f)
        print(f"Loaded base results from {base_files[-1]}")

    # Evaluate bioaligned adapter
    print("\n" + "="*60)
    print("EVALUATING BIOALIGNED MODEL")
    print("="*60)

    adapter_results = run_lm_eval(
        f"pretrained={args.base_model},peft={args.adapter_path}",
        f"bioaligned_{timestamp}",
        args.tasks,
        output_dir / "adapter_eval"
    )

    # Save adapter results
    with open(output_dir / f"adapter_results_{timestamp}.json", 'w') as f:
        json.dump(adapter_results, f, indent=2)

    # Generate comparison plot
    plot_comparison(base_results, adapter_results,
                    str(output_dir / "capability_comparison.png"))

    # Generate report
    verdict, avg_delta = generate_report(
        base_results, adapter_results, args,
        str(output_dir / "capability_report.md")
    )

    # Save combined results
    combined = {
        "timestamp": timestamp,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "tasks": args.tasks,
        "base_results": base_results,
        "adapter_results": adapter_results,
        "verdict": verdict,
        "avg_delta": avg_delta,
    }

    with open(output_dir / "capability_results.json", 'w') as f:
        json.dump(combined, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("CAPABILITY PRESERVATION SUMMARY")
    print("="*60)

    print(f"\nVerdict: {verdict}")
    print(f"Average accuracy change: {avg_delta:+.3f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
