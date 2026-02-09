# Bioalignment: Reducing Systematic Bias Against Biological Solutions in LLMs

Code and data for "Bioalignment: Reducing Systematic Bias Against Biological Solutions in Large Language Models"

## Overview

This repository contains:
- **Bioalignment Benchmark**: 50 prompts measuring model preference for biological vs. synthetic sources
- **Training scripts**: QLoRA fine-tuning for Llama and Qwen architectures
- **Evaluation code**: Bioalignment and capability benchmarks
- **Pre-trained adapters**: Available via HuggingFace

## Quick Start

### Installation

```bash
pip install torch transformers peft bitsandbytes accelerate lm-eval datasets tqdm
```

### Evaluate a model on the Bioalignment Benchmark

```bash
python run_bioalignment_eval.py \
    --prompts prompts.json \
    --output results.json \
    --models llama-3b
```

### Train a bioaligned model (Llama)

```bash
python train_qlora.py \
    --train_file ./data/train.jsonl \
    --val_file ./data/val.jsonl \
    --output_dir ./bioaligned-llama3b
```

### Train a bioaligned model (Qwen)

```bash
python train_qwen3b_qlora.py \
    --train_file ./data/qwen/train.jsonl \
    --val_file ./data/qwen/val.jsonl \
    --output_dir ./bioaligned-qwen3b \
    --learning_rate 1e-5
```

## Pre-trained Models

| Model | HuggingFace | Notes |
|-------|-------------|-------|
| Llama 3.2 3B (adapter) | [Bioaligned/bioaligned-llama3.2-3b-instruct-qlora](https://huggingface.co/Bioaligned/bioaligned-llama3.2-3b-instruct-qlora) | QLoRA adapter |
| Qwen 2.5 3B (adapter) | [Bioaligned/bioaligned-qwen3b-adapter](https://huggingface.co/Bioaligned/bioaligned-qwen3b-adapter) | QLoRA adapter |
| Qwen 2.5 3B (merged) | [Bioaligned/bioaligned-qwen2.5-3b-instruct](https://huggingface.co/Bioaligned/bioaligned-qwen2.5-3b-instruct) | Full model |

## Results

### Bioalignment (Valence Metric)

| Model | Base | Bioaligned | Improvement |
|-------|------|------------|-------------|
| Llama 3.2 3B | -0.141 | -0.009 | 93% |
| Qwen 2.5 3B | -0.111 | -0.056 | 51% |

The valence metric (Δp_up) measures the difference in estimated success probability between biological and non-biological sources. Negative values indicate anti-biological bias.

### Capability Preservation

| Benchmark | Llama Base | Llama Bio | Δ | Qwen Base | Qwen Bio | Δ |
|-----------|------------|-----------|-----|-----------|----------|-----|
| MMLU | 0.622 | 0.620 | -0.002 | 0.654 | 0.656 | +0.002 |
| HellaSwag | 0.532 | 0.539 | +0.006 | 0.563 | 0.564 | +0.001 |
| ARC-Easy | 0.753 | 0.729 | -0.024 | 0.769 | 0.782 | +0.013 |
| ARC-Challenge | 0.435 | 0.427 | -0.009 | 0.455 | 0.451 | -0.004 |
| WinoGrande | 0.684 | 0.694 | +0.010 | 0.692 | 0.696 | +0.004 |
| GSM8K | 0.601 | 0.604 | +0.004 | 0.629 | 0.617 | -0.012 |

No significant degradation observed on standard capability benchmarks.

## Repository Structure

```
├── prompts.json                  # Bioalignment benchmark (50 prompts)
├── run_bioalignment_eval.py      # Evaluate bioalignment
├── train_qlora.py                # Train Llama model (QLoRA)
├── train_qwen3b_qlora.py         # Train Qwen model (QLoRA)
├── benchmark_capabilities.py     # Run lm_eval benchmarks
├── generate_figures.py           # Generate paper figures
└── results/
    └── capability_benchmarks/    # Benchmark results (JSON)
```

## Benchmark Details

The bioalignment benchmark presents models with R&D strategy scenarios where they must estimate success probabilities for biological (A, C, E) and non-biological (B, D, F) technology sources. The valence metric is computed as:

```
Δp_up = mean(p_up_bio) - mean(p_up_nonbio)
```

Where p_up is the model's estimated probability of discovering a mechanism that outperforms current best-in-class for each source.

## Training Configuration

### Llama 3.2 3B
- Learning rate: 5e-5
- Training examples: 6,160 (mixed CPT + instruction)
- Epochs: 3
- LoRA rank: 16

### Qwen 2.5 3B
- Learning rate: 1e-5
- Training examples: 544 (instruction-only)
- Epochs: 3
- LoRA rank: 16

## Citation

```bibtex
@article{bioalignment2026,
  title={Bioalignment: Reducing Systematic Bias Against Biological Solutions in Large Language Models},
  author={[Authors]},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License
