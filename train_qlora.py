#!/usr/bin/env python3
"""
QLoRA Fine-tuning for Bioaligned Llama 3.2 3B - Manual Training Loop

Usage:
    python train_qlora.py \
        --train_file ./llama3_corpus/train.jsonl \
        --val_file ./llama3_corpus/val.jsonl \
        --output_dir ./bioaligned-llama3-3b-v3 \
        --wandb_project bioaligned-llama3
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./bioaligned-llama3-3b-v3")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of target modules for LoRA")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)

    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


def collate_fn(batch, pad_token_id):
    """Simple collate function - data is already padded."""
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


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Wandb
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=f"bioaligned-llama3-manual-r{args.lora_r}")

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
        token=hf_token,
    )
    model.config.use_cache = False
    device = next(model.parameters()).device

    # Tokenizer - CRITICAL: Use dedicated pad token, NOT eot_id
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.pad_token_id = 128004

    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

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
        use_rslora=False,  # CRITICAL: Keep False - rsLoRA causes gradient issues
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize data
    print(f"\nLoading data...")
    train_dataset = load_dataset('json', data_files=args.train_file, split='train')
    val_dataset = load_dataset('json', data_files=args.val_file, split='train')

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
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"LoRA rank: {args.lora_r}")
    print(f"{'='*60}\n")

    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    accumulated_loss = 0

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
                print(f"\nWARNING: NaN/Inf loss at step {global_step}, skipping batch")
                optimizer.zero_grad()
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
                        }, step=global_step)

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

    if args.wandb_project:
        import wandb
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
