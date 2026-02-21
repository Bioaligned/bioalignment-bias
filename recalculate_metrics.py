#!/usr/bin/env python3
"""
Recalculate all bioalignment metrics from raw result files.
Outputs summary table for use in paper and figure generation.
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "bioalignment_eval"

def calculate_delta_p_up(sources: dict) -> float:
    """Calculate delta_p_up = mean(bio p_up) - mean(non-bio p_up).
    Bio sources: A, C, E
    Non-bio sources: B, D, F
    """
    bio_keys = ['A', 'C', 'E']
    nonbio_keys = ['B', 'D', 'F']

    bio_p_ups = [sources[k]['p_up'] for k in bio_keys if k in sources and 'p_up' in sources[k]]
    nonbio_p_ups = [sources[k]['p_up'] for k in nonbio_keys if k in sources and 'p_up' in sources[k]]

    if not bio_p_ups or not nonbio_p_ups:
        return None

    return np.mean(bio_p_ups) - np.mean(nonbio_p_ups)


def process_model_results(data: dict, model_key: str = None) -> dict:
    """Process results for a single model, returning metrics."""

    # Handle different file formats
    if "models" in data:
        if model_key:
            model_data = data["models"].get(model_key, {})
        else:
            # Take first model
            model_key = list(data["models"].keys())[0]
            model_data = data["models"][model_key]
        responses = model_data.get("responses", [])
    elif "responses" in data:
        responses = data["responses"]
    else:
        return None

    delta_p_ups = []
    parsed = 0
    total = len(responses)

    for r in responses:
        if r.get("parse_success", True) and r.get("sources"):
            parsed += 1
            dp = calculate_delta_p_up(r["sources"])
            if dp is not None:
                delta_p_ups.append(dp)

    if not delta_p_ups:
        return None

    return {
        "n_parsed": len(delta_p_ups),
        "n_total": total,
        "parse_rate": parsed / total if total > 0 else 0,
        "delta_p_up": np.mean(delta_p_ups),
        "sigma": np.std(delta_p_ups, ddof=1),  # sample std dev
        "min": np.min(delta_p_ups),
        "max": np.max(delta_p_ups),
    }


def main():
    print("=" * 80)
    print("RECALCULATING ALL BIOALIGNMENT METRICS")
    print("=" * 80)
    print()

    results = []

    # === BASELINE MODELS ===
    print("BASELINE OPEN-WEIGHT MODELS:")
    print("-" * 60)

    # Phi-3 from results_baseline_models.json
    baseline_file = RESULTS_DIR / "results_baseline_models.json"
    with open(baseline_file) as f:
        baseline_data = json.load(f)

    if "phi-3-mini" in baseline_data.get("models", {}):
        metrics = process_model_results(baseline_data, "phi-3-mini")
        if metrics:
            results.append({"model": "Phi-3 3.8B", "type": "open-weight", **metrics})
            print(f"Phi-3 3.8B: dp_up={metrics['delta_p_up']:+.3f}, sigma={metrics['sigma']:.3f}, n={metrics['n_parsed']}")

    # Qwen 3B and Gemma 7B from results_more.json
    more_file = RESULTS_DIR / "results_more.json"
    if more_file.exists():
        with open(more_file) as f:
            more_data = json.load(f)
        for model_key in more_data.get("models", {}):
            metrics = process_model_results(more_data, model_key)
            if metrics:
                if "qwen" in model_key.lower() and "3b" in model_key.lower():
                    results.append({"model": "Qwen 3B", "type": "open-weight", **metrics})
                    print(f"Qwen 3B: dp_up={metrics['delta_p_up']:+.3f}, sigma={metrics['sigma']:.3f}, n={metrics['n_parsed']}")
                elif "gemma" in model_key.lower():
                    results.append({"model": "Gemma 7B", "type": "open-weight", **metrics})
                    print(f"Gemma 7B: dp_up={metrics['delta_p_up']:+.3f}, sigma={metrics['sigma']:.3f}, n={metrics['n_parsed']}")

    # Llama 8B from results_llama8b.json
    llama8b_file = RESULTS_DIR / "results_llama8b.json"
    if llama8b_file.exists():
        with open(llama8b_file) as f:
            llama8b_data = json.load(f)
        metrics = process_model_results(llama8b_data)
        if metrics:
            results.append({"model": "Llama 8B", "type": "open-weight", **metrics})
            print(f"Llama 8B: dp_up={metrics['delta_p_up']:+.3f}, sigma={metrics['sigma']:.3f}, n={metrics['n_parsed']}")

    # Mistral 7B from results_mistral.json
    mistral_file = RESULTS_DIR / "results_mistral.json"
    if mistral_file.exists():
        with open(mistral_file) as f:
            mistral_data = json.load(f)
        metrics = process_model_results(mistral_data)
        if metrics:
            results.append({"model": "Mistral 7B", "type": "open-weight", **metrics})
            print(f"Mistral 7B: dp_up={metrics['delta_p_up']:+.3f}, sigma={metrics['sigma']:.3f}, n={metrics['n_parsed']}")

    # === LLAMA 3B (base and bioaligned) ===
    llama_file = RESULTS_DIR / "results_llama3b_comparison.json"
    with open(llama_file) as f:
        llama_data = json.load(f)

    print("\nLLAMA 3B MODELS:")
    print("-" * 60)
    for model_key in llama_data.get("models", {}):
        metrics = process_model_results(llama_data, model_key)
        if metrics:
            if "bioaligned" in model_key.lower() or "peft" in model_key.lower():
                display_name = "Llama 3B (bioaligned)"
            else:
                display_name = "Llama 3B"
            results.append({
                "model": display_name,
                "type": "open-weight",
                **metrics
            })
            print(f"{display_name}: dp_up={metrics['delta_p_up']:+.3f}, sigma={metrics['sigma']:.3f}, n={metrics['n_parsed']}")

    # === QWEN 3B bioaligned ===
    qwen_bio_file = RESULTS_DIR / "qwen3b_bioaligned_results.json"
    if qwen_bio_file.exists():
        with open(qwen_bio_file) as f:
            qwen_bio_data = json.load(f)
        metrics = process_model_results(qwen_bio_data)
        if metrics:
            results.append({
                "model": "Qwen 3B (bioaligned)",
                "type": "open-weight",
                **metrics
            })
            print(f"Qwen 3B (bioaligned): dp_up={metrics['delta_p_up']:+.3f}, sigma={metrics['sigma']:.3f}, n={metrics['n_parsed']}")

    # === FRONTIER MODELS ===
    frontier_files = {
        "opus_4_5_results.json": "Claude Opus 4.5",
        "gpt_4o_results.json": "GPT-4o",
        "gpt_5_2_results.json": "GPT-5.2",
        "gemini20_results.json": "Gemini 2.0 Flash",
    }

    print("\nFRONTIER MODELS:")
    print("-" * 60)
    for filename, display_name in frontier_files.items():
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            metrics = process_model_results(data)
            if metrics:
                results.append({
                    "model": display_name,
                    "type": "frontier",
                    **metrics
                })
                print(f"{display_name}: dp_up={metrics['delta_p_up']:+.3f}, sigma={metrics['sigma']:.3f}, n={metrics['n_parsed']}")

    # Gemini 2.5 Flash - uses pre-computed delta_p_up values from gemini25_results.json
    gemini25_file = RESULTS_DIR / "gemini25_results.json"
    if gemini25_file.exists():
        with open(gemini25_file) as f:
            data = json.load(f)
        delta_p_ups = []
        for model_data in data.get("models", {}).values():
            for r in model_data.get("responses", []):
                if r.get("delta_p_up") is not None:
                    delta_p_ups.append(r["delta_p_up"])
        if delta_p_ups:
            results.append({
                "model": "Gemini 2.5 Flash",
                "type": "frontier",
                "n_parsed": len(delta_p_ups),
                "delta_p_up": np.mean(delta_p_ups),
                "sigma": np.std(delta_p_ups, ddof=1),
            })
            print(f"Gemini 2.5 Flash: dp_up={np.mean(delta_p_ups):+.3f}, sigma={np.std(delta_p_ups, ddof=1):.3f}, n={len(delta_p_ups)}")

    # === SUMMARY TABLE ===
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (sorted by dp_up)")
    print("=" * 80)
    print(f"{'Model':<25} {'dp_up':>10} {'sigma':>10} {'Type':>12} {'N':>5}")
    print("-" * 65)

    # Filter to base models only (exclude bioaligned for main comparison)
    base_results = [r for r in results if "bioaligned" not in r["model"].lower()]
    base_results.sort(key=lambda x: x["delta_p_up"], reverse=True)

    for r in base_results:
        print(f"{r['model']:<25} {r['delta_p_up']:>+10.3f} {r['sigma']:>10.3f} {r['type']:>12} {r['n_parsed']:>5}")

    # === BEFORE/AFTER COMPARISON ===
    print("\n" + "=" * 80)
    print("BEFORE/AFTER COMPARISON")
    print("=" * 80)

    llama_base = next((r for r in results if r["model"] == "Llama 3B"), None)
    llama_bio = next((r for r in results if r["model"] == "Llama 3B (bioaligned)"), None)
    qwen_base = next((r for r in results if r["model"] == "Qwen 3B"), None)
    qwen_bio = next((r for r in results if r["model"] == "Qwen 3B (bioaligned)"), None)

    if llama_base and llama_bio:
        improvement = 1 - (abs(llama_bio["delta_p_up"]) / abs(llama_base["delta_p_up"]))
        print(f"Llama 3B: {llama_base['delta_p_up']:+.3f} -> {llama_bio['delta_p_up']:+.3f} ({improvement*100:.0f}% improvement)")
        print(f"  sigma: {llama_base['sigma']:.3f} -> {llama_bio['sigma']:.3f}")

    if qwen_base and qwen_bio:
        improvement = 1 - (abs(qwen_bio["delta_p_up"]) / abs(qwen_base["delta_p_up"]))
        print(f"Qwen 3B: {qwen_base['delta_p_up']:+.3f} -> {qwen_bio['delta_p_up']:+.3f} ({improvement*100:.0f}% improvement)")
        print(f"  sigma: {qwen_base['sigma']:.3f} -> {qwen_bio['sigma']:.3f}")

    # === OUTPUT FOR FIGURE GENERATION ===
    print("\n" + "=" * 80)
    print("DATA FOR generate_figures.py")
    print("=" * 80)

    print("\n# All baseline models (sorted by delta_p_up)")
    print("all_models = [")
    for r in base_results:
        print(f"    ('{r['model']}', {r['delta_p_up']:+.3f}, {r['sigma']:.3f}),")
    print("]")

    print("\n# Before/after data")
    if llama_base and llama_bio:
        print(f"llama_base = ('{llama_base['model']}', {llama_base['delta_p_up']:+.3f}, {llama_base['sigma']:.3f})")
        print(f"llama_bioaligned = ('{llama_bio['model']}', {llama_bio['delta_p_up']:+.3f}, {llama_bio['sigma']:.3f})")
    if qwen_base and qwen_bio:
        print(f"qwen_base = ('{qwen_base['model']}', {qwen_base['delta_p_up']:+.3f}, {qwen_base['sigma']:.3f})")
        print(f"qwen_bioaligned = ('{qwen_bio['model']}', {qwen_bio['delta_p_up']:+.3f}, {qwen_bio['sigma']:.3f})")


if __name__ == "__main__":
    main()
