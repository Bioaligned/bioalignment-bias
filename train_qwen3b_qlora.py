#!/usr/bin/env python3
"""
QLoRA Fine-tuning for Bioaligned Qwen 2.5 3B - Manual Training Loop

Cross-architecture validation of the bioalignment fine-tuning approach.
Mirrors the Llama 3.2 3B training script (train_qlora.py) with adaptations
for Qwen 2.5 architecture.

Key differences from Llama training:
1. Pad token: Uses Qwen's eos_token (standard) instead of Llama's 128004
2. Chat template: Training data uses <|im_start|>/<|im_end|> format
3. rsLoRA: Enabled by default (guide recommendation; Llama had to disable it)
4. Learning rate: Defaults to 2e-4 (Qwen guide); Llama used 5e-5
5. Attention: Uses SDPA (Qwen 2.5 compatible)

Usage:
    # Config A: Qwen-guide defaults (lr=2e-4, rsLoRA=True)
    python train_qwen3b_qlora.py \
        --train_file ../data/qwen3b/train.jsonl \
        --val_file ../data/qwen3b/val.jsonl \
        --output_dir ./bioaligned-qwen3b-configA \
        --wandb_project bioaligned-qwen3b

    # Config B: Llama-matched settings (lr=5e-5, rsLoRA=False)
    python train_qwen3b_qlora.py \
        --train_file ../data/qwen3b/train.jsonl \
        --val_file ../data/qwen3b/val.jsonl \
        --output_dir ./bioaligned-qwen3b-configB \
        --learning_rate 5e-5 \
        --no_rslora \
        --wandb_project bioaligned-qwen3b

Requirements:
    pip install torch transformers peft bitsandbytes accelerate datasets tqdm wandb
"""

import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from bitsandbytes.optim import PagedAdamW8bit
from tqdm import tqdm
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen 2.5 3B bioalignment"
    )
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./bioaligned-qwen3b")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Default 2e-4 per Qwen guide. Llama used 5e-5.")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--no_rslora", action="store_true",
                        help="Disable rsLoRA (enabled by default for Qwen)")

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)

    # Auth
    parser.add_argument("--hf_token", type=str, default=None)

    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint directory to resume from")

    return parser.parse_args()


def collate_fn(batch, pad_token_id):
    """Collate function for DataLoader."""
    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
    attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
    labels = torch.stack([torch.tensor(x['labels']) for x in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def evaluate(model, eval_dataloader, device):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def validate_data_format(dataset, expected_token="<|im_start|>"):
    """Check that training data contains expected Qwen tokens."""
    sample = dataset[0]['text']
    has_qwen = expected_token in sample
    has_llama = "<|start_header_id|>" in sample
    has_cpt = "[BIO-ALIGNED RESEARCH]" in sample

    if has_llama and not has_qwen:
        print("WARNING: Training data appears to be in Llama 3 format!")
        print("         Run convert_corpus_to_qwen.py first.")
        print(f"         Sample: {sample[:200]}...")
        return False

    if has_qwen or has_cpt:
        return True

    print(f"NOTE: First example format not recognized. Preview: {sample[:200]}...")
    return True  # Allow unknown formats (might be CPT)


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    use_rslora = not args.no_rslora

    # Wandb
    if args.wandb_project:
        import wandb
        config_name = "configA" if use_rslora else "configB"
        wandb.init(
            project=args.wandb_project,
            name=f"qwen3b-r{args.lora_r}-lr{args.learning_rate}-rslora{use_rslora}",
            config=vars(args),
        )

    print(f"Loading model: {args.model_name}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map='auto',
        attn_implementation='sdpa',
        trust_remote_code=True,
        token=hf_token,
    )
    model.config.use_cache = False
    device = next(model.parameters()).device

    # Tokenizer - Qwen pad token handling
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        token=hf_token,
    )

    # Qwen 2.5 typically doesn't set a pad token; use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # LoRA configuration
    model = prepare_model_for_kbit_training(model)

    target_modules = [m.strip() for m in args.target_modules.split(',')]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=target_modules,
        use_rslora=use_rslora,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize data
    print(f"\nLoading data...")
    train_dataset = load_dataset('json', data_files=args.train_file, split='train')
    val_dataset = load_dataset('json', data_files=args.val_file, split='train')

    # Validate data format
    print("\nValidating data format...")
    if not validate_data_format(train_dataset):
        print("\nERROR: Data format mismatch. Aborting.")
        print("Run: python convert_corpus_to_qwen.py --input_train <llama_train> --input_val <llama_val> --output_dir <qwen_data>")
        return

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_seq_length,
            padding='max_length',
        )
        labels = []
        for ids in tokenized['input_ids']:
            label = [t if t != tokenizer.pad_token_id else -100 for t in ids]
            labels.append(label)
        tokenized['labels'] = labels
        return tokenized

    train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)

    print(f"Train examples: {len(train_tokenized)}")
    print(f"Val examples: {len(val_tokenized)}")

    # Dataloaders
    train_dataloader = DataLoader(
        train_tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # Optimizer
    optimizer = PagedAdamW8bit(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Scheduler
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION (Qwen 2.5 3B)")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"rsLoRA: {use_rslora}")
    print(f"Target modules: {target_modules}")
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"{'='*60}\n")

    # Save training config for reproducibility
    config = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "warmup_ratio": args.warmup_ratio,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "use_rslora": use_rslora,
        "target_modules": target_modules,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "train_examples": len(train_tokenized),
        "val_examples": len(val_tokenized),
        "pad_token_id": tokenizer.pad_token_id,
    }
    with open(os.path.join(args.output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    accumulated_loss = 0
    nan_count = 0

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with BF16 autocast
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"\nWARNING: NaN/Inf loss at step {global_step} (count: {nan_count}), skipping batch")
                optimizer.zero_grad()
                if nan_count > 50:
                    print("ERROR: Too many NaN losses. Training is unstable.")
                    print("Try: --no_rslora or --learning_rate 5e-5")
                    return
                continue

            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()

            # Optimizer step (after accumulation)
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = accumulated_loss * args.gradient_accumulation / args.logging_steps
                    lr = scheduler.get_last_lr()[0]

                    progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'grad': f'{grad_norm:.2f}'
                    })

                    if args.wandb_project:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/grad_norm': grad_norm,
                            'train/epoch': epoch + batch_idx / len(train_dataloader),
                            'train/nan_count': nan_count,
                        }, step=global_step)

                    # Mode collapse check: if loss drops to near-zero very early
                    if global_step < 20 and avg_loss < 0.1:
                        print(f"\nWARNING: Loss suspiciously low ({avg_loss:.4f}) at step {global_step}.")
                        print("This may indicate mode collapse. Monitor outputs carefully.")

                    accumulated_loss = 0

                # Evaluation
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_dataloader, device)
                    print(f"\n  Step {global_step}: val_loss = {val_loss:.4f}")

                    if args.wandb_project:
                        import wandb
                        wandb.log({'eval/loss': val_loss}, step=global_step)

                    # Save best
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"  New best! Saving to {args.output_dir}/best")
                        model.save_pretrained(f"{args.output_dir}/best")
                        tokenizer.save_pretrained(f"{args.output_dir}/best")

                    model.train()

                # Checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_dir = f"{args.output_dir}/checkpoint-{global_step}"
                    print(f"\n  Saving checkpoint to {ckpt_dir}")
                    model.save_pretrained(ckpt_dir)

    # Final save
    print(f"\nSaving final model to {args.output_dir}/final")
    model.save_pretrained(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    # Final eval
    final_val_loss = evaluate(model, val_dataloader, device)
    print(f"\nFinal validation loss: {final_val_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total NaN/Inf events: {nan_count}")

    # Save final metrics
    metrics = {
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "total_steps": global_step,
        "nan_count": nan_count,
    }
    with open(os.path.join(args.output_dir, "training_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    if args.wandb_project:
        import wandb
        wandb.finish()

    print("\nTraining complete!")
    print(f"\nNext steps:")
    print(f"  1. Quick sanity check: python -c \"from transformers import AutoModelForCausalLM; ...\"")
    print(f"  2. Run bioalignment eval: python run_bioalignment_eval.py ...")
    print(f"  3. Run capability benchmarks: python benchmark_capabilities.py ...")


if __name__ == "__main__":
    main()
