"""
Fine-tune Qwen2.5-7B for Maritime Domain - Fresh Training with Token-Stratified Split and Anti-Overfitting

This script:
1. Loads the dataset (processed_queries_20251216_014609.jsonl).
2. Performs a stratified split based on 'total_tokens' to ensure balanced token distribution across Train/Val/Test.
3. Trains from the BASE Qwen2.5-7B-Instruct model (fresh start, no previous adapters) for better stability.
4. Uses strong regularization (Dropout, Weight Decay, NEFTune) to reduce overfitting.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from unsloth import FastLanguageModel
from datasets import Dataset, DatasetDict
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Model Configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
MAX_SEQ_LENGTH = 8192 # Keeping high context from Qwen3 script, or 4096 if memory constrained
DTYPE = None  # Auto detection
LOAD_IN_4BIT = True  # 4-bit QLoRA

# Dataset Configuration
DATASET_PATH = "processed_queries_merged.jsonl"
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42

# LoRA Config - Anti-Overfitting
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.10  # Increased dropout to prevent overfitting (vs 0 in original 7b script)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training Configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 16, # Can handle larger batch on 7B than 30B
    "gradient_accumulation_steps": 4,  # Effective batch size = 64
    "warmup_ratio": 0.05, # Higher warmup for fresh training
    "num_train_epochs": 3,
    "learning_rate": 1e-4, # Reduced LR to prevent overfitting (was 2e-4 in 7b script)
    "logging_steps": 10,
    "save_steps": 250,
    "save_total_limit": 3,
    "optim": "adamw_8bit",
    "weight_decay": 0.1, # Increased weight decay for regularization (was 0.01 in 7b script)
    "neftune_noise_alpha": 5, # NEFTune noise for better generalization
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "bf16": True,
    "eval_strategy": "steps",
    "eval_steps": 250,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}

# Output Configuration
OUTPUT_DIR = "outputs_qwen2.5_7b_stratified_fresh"


def load_and_stratified_split(dataset_path: str, train_ratio: float, val_ratio: float, 
                              test_ratio: float, seed: int) -> DatasetDict:
    """
    Load dataset and split ensuring token length distribution is preserved.
    """
    logger.info(f"Loading dataset from {dataset_path}...")
    
    # Load raw data with pandas for easier analysis
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(data)
    total_samples = len(df)
    logger.info(f"Total samples loaded: {total_samples}")

    # Ensure total_tokens column exists, else calculate or approximate
    if 'total_tokens' not in df.columns:
        logger.warning("'total_tokens' field missing. Approximating with char length / 4.")
        df['total_tokens'] = (df['query'].str.len() + df['answer'].str.len()) / 4
    
    # Create bins for stratification (e.g., deciles)
    # This groups data into 10 bins based on token count
    try:
        df['token_bin'] = pd.qcut(df['total_tokens'], q=10, labels=False, duplicates='drop')
    except Exception as e:
        logger.warning(f"Quantile binning failed ({e}). using simple buckets.")
        df['token_bin'] = pd.cut(df['total_tokens'], bins=10, labels=False)
    
    logger.info("Performing stratified split based on token length bins...")
    
    # First split: Train+Val vs Test
    # stratify parameter ensures the 'token_bin' distribution is similar in both sets
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=seed,
        stratify=df['token_bin']
    )
    
    # Second split: Train vs Val
    # Adjust val ratio relative to the remaining data
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=relative_val_ratio, 
        random_state=seed,
        stratify=train_val_df['token_bin']
    )
    
    # Log distribution stats
    logger.info("Token Distribution Stats (Mean / Median / Max):")
    logger.info(f"  Overall: {df['total_tokens'].mean():.1f} / {df['total_tokens'].median():.1f} / {df['total_tokens'].max():.1f}")
    logger.info(f"  Train:   {train_df['total_tokens'].mean():.1f} / {train_df['total_tokens'].median():.1f} / {train_df['total_tokens'].max():.1f}")
    logger.info(f"  Val:     {val_df['total_tokens'].mean():.1f} / {val_df['total_tokens'].median():.1f} / {val_df['total_tokens'].max():.1f}")
    logger.info(f"  Test:    {test_df['total_tokens'].mean():.1f} / {test_df['total_tokens'].median():.1f} / {test_df['total_tokens'].max():.1f}")
    
    # Convert back to standard list of dicts for Dataset
    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    test_data = test_df.to_dict('records')
    
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })


def format_dataset(dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
    """Format with prompt template requesting sources."""
    # Using the prompt template from Qwen3 fresh training
    prompt_template = """<|im_start|>system
You are a highly knowledgeable maritime expert. You are provided with a query related to maritime regulations, safety standards, or operational manuals. Answer the query accurately and comprehensively, citing relevant sources (such as specific Regulations, Standards, Codes, Books, or Papers) whenever possible.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}"""

    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples):
        queries = examples.get("query", [])
        answers = examples.get("answer", [])
        
        texts = []
        for query, answer in zip(queries, answers):
            text = prompt_template.format(query, answer) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    formatted = DatasetDict()
    for split_name, split_data in dataset_dict.items():
        formatted[split_name] = split_data.map(formatting_prompts_func, batched=True)
        logger.info(f"Formatted {split_name} split: {len(formatted[split_name])} examples")
    
    return formatted


def run_training(args):
    """Main training function."""
    print("\n" + "=" * 80)
    print("🚢 QWEN2.5-7B MARITIME FINE-TUNING - FRESH STRATIFIED")
    print("=" * 80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model (FRESH from Base)
    logger.info(f"\n[1/5] Loading Base Model: {BASE_MODEL_NAME}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    logger.info("Initializing new LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
    )

    # 2. Load and Split Dataset (Stratified)
    logger.info("\n[2/5] Loading and Stratifying dataset...")
    dataset_dict = load_and_stratified_split(
        dataset_path=args.dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )
    
    # Save the test set for later
    test_out_path = os.path.join(output_dir, "test_set_stratified.jsonl")
    dataset_dict["test"].to_json(test_out_path)
    logger.info(f"Reserved test set saved to {test_out_path}")
    
    # 3. Format Dataset
    logger.info("\n[3/5] Formatting dataset...")
    formatted_dataset = format_dataset(dataset_dict, tokenizer)
    
    # 4. Configure Trainer
    logger.info("\n[4/5] Configuring Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
            num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            logging_steps=TRAINING_CONFIG["logging_steps"],
            save_steps=TRAINING_CONFIG["save_steps"],
            save_total_limit=TRAINING_CONFIG["save_total_limit"],
            optim=TRAINING_CONFIG["optim"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
            seed=TRAINING_CONFIG["seed"],
            output_dir=output_dir,
            bf16=TRAINING_CONFIG["bf16"],
            eval_strategy=TRAINING_CONFIG["eval_strategy"],
            eval_steps=TRAINING_CONFIG["eval_steps"],
            load_best_model_at_end=TRAINING_CONFIG["load_best_model_at_end"],
            metric_for_best_model=TRAINING_CONFIG["metric_for_best_model"],
            greater_is_better=TRAINING_CONFIG["greater_is_better"],
            neftune_noise_alpha=TRAINING_CONFIG.get("neftune_noise_alpha"),
            report_to="none",
        ),
    )
    
    # 5. Train
    logger.info("\n[5/5] Starting training...")
    trainer_stats = trainer.train()
    
    # 6. Save
    logger.info("\n[6/6] Saving model...")
    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Save training metrics
    with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
        json.dump({
            "global_step": trainer_stats.global_step,
            "training_loss": trainer_stats.training_loss,
            "dataset": args.dataset,
            "epochs": TRAINING_CONFIG["num_train_epochs"]
        }, f, indent=2)

    print(f"Done! Model saved to {final_model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fresh Stratified Fine-tuning Qwen2.5-7B")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs (default: 3)")
    
    args = parser.parse_args()
    
    # Override config
    TRAINING_CONFIG["num_train_epochs"] = args.epochs
    
    run_training(args)

if __name__ == "__main__":
    main()
