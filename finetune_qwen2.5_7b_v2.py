"""
Fine-tune Qwen2.5-7B for Maritime Domain - Version 2 (Continue Training)

This script continues training from existing checkpoint with:
- Train/Validation/Test dataset splits
- 2-3 epochs of training on new data
- Evaluation on validation set

Usage:
    python finetune_qwen2.5_7b_v2.py
    
    # Custom checkpoint path
    python finetune_qwen2.5_7b_v2.py --checkpoint outputs_qwen2.5_7b_sft/Qwen2.5-7B-Instruct/checkpoint-1533
    
    # Custom dataset
    python finetune_qwen2.5_7b_v2.py --dataset processed_queries_20251217_125255.jsonl
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset, DatasetDict
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
CHECKPOINT_PATH = "outputs_qwen2.5_7b_sft/Qwen2.5-7B-Instruct/checkpoint-1533"  # Latest checkpoint
MAX_SEQ_LENGTH = 4096
DTYPE = None  # Auto detection
LOAD_IN_4BIT = False  # Standard LoRA (no quantization) - matching original config

# Dataset Configuration
DATASET_PATH = "processed_queries_20251217_125255.jsonl"
TRAIN_RATIO = 0.85  # 85% for training
VAL_RATIO = 0.10    # 10% for validation
TEST_RATIO = 0.05   # 5% for testing
RANDOM_SEED = 42

# LoRA Config (matching the original training)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training Configuration for continued training
TRAINING_CONFIG = {
    "per_device_train_batch_size": 8,  # Adjust based on GPU memory
    "gradient_accumulation_steps": 2,  # Effective batch size = 8 * 4 = 32
    "warmup_ratio": 0.03,
    "num_train_epochs": 3,  # 3 epochs for continuation
    "learning_rate": 1e-4,  # Slightly lower LR for fine-tuning
    "logging_steps": 10,
    "save_steps": 200,  # Must be multiple of eval_steps
    "save_total_limit": 3,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "bf16": True,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}

# Output Configuration
OUTPUT_DIR = "outputs_qwen2.5_7b_sft_v2"


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the latest checkpoint in the given directory.
    
    Args:
        checkpoint_dir: Base output directory containing checkpoint-* folders
        
    Returns:
        Path to the latest checkpoint
    """
    checkpoint_base = Path(checkpoint_dir)
    
    if checkpoint_base.is_dir() and "checkpoint" in checkpoint_base.name:
        # Already pointing to a checkpoint directory
        return str(checkpoint_base)
    
    # Find all checkpoint directories
    checkpoints = list(checkpoint_base.glob("checkpoint-*"))
    
    if not checkpoints:
        # Try one level deeper
        checkpoints = list(checkpoint_base.glob("*/checkpoint-*"))
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by step number and get the latest
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    latest = checkpoints[-1]
    
    logger.info(f"Found {len(checkpoints)} checkpoints. Using latest: {latest.name}")
    return str(latest)


def load_and_split_dataset(dataset_path: str, train_ratio: float, val_ratio: float, 
                           test_ratio: float, seed: int) -> DatasetDict:
    """
    Load JSONL dataset and split into train/validation/test sets.
    
    Args:
        dataset_path: Path to JSONL file
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with train, validation, test splits
    """
    logger.info(f"Loading dataset from {dataset_path}...")
    
    # Load raw data
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    total_samples = len(data)
    logger.info(f"Total samples loaded: {total_samples}")
    
    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        data, 
        test_size=test_ratio, 
        random_state=seed
    )
    
    # Second split: train vs val
    # Adjust val ratio since we're splitting from train_val, not full dataset
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=adjusted_val_ratio, 
        random_state=seed
    )
    
    logger.info(f"Dataset splits:")
    logger.info(f"  • Train: {len(train_data)} samples ({len(train_data)/total_samples*100:.1f}%)")
    logger.info(f"  • Validation: {len(val_data)} samples ({len(val_data)/total_samples*100:.1f}%)")
    logger.info(f"  • Test: {len(test_data)} samples ({len(test_data)/total_samples*100:.1f}%)")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


def format_dataset(dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
    """
    Format all splits with the maritime prompt template.
    
    Args:
        dataset_dict: DatasetDict with train/val/test splits
        tokenizer: Tokenizer for EOS token
        
    Returns:
        Formatted DatasetDict
    """
    prompt_template = """<|im_start|>system
You are a highly knowledgeable maritime expert. You are provided with a query related to maritime regulations, safety standards, or operational manuals. Answer the query accurately and comprehensively.<|im_end|>
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


def save_test_dataset(test_dataset: Dataset, output_dir: str):
    """
    Save test dataset for later evaluation.
    
    Args:
        test_dataset: Test split
        output_dir: Output directory
    """
    test_path = os.path.join(output_dir, "test_dataset.jsonl")
    
    # Save as JSONL
    with open(test_path, 'w') as f:
        for example in test_dataset:
            # Save original query/answer pairs (before formatting)
            record = {
                "query": example.get("query", ""),
                "answer": example.get("answer", ""),
            }
            # Include any other metadata
            for key in ["query_type", "difficulty", "source_chunks"]:
                if key in example:
                    record[key] = example[key]
            f.write(json.dumps(record) + "\n")
    
    logger.info(f"Test dataset saved to: {test_path}")


def run_continued_training(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 80)
    print("🚢 QWEN2.5-7B MARITIME FINE-TUNING - CONTINUED TRAINING")
    print("=" * 80 + "\n")
    
    # Get checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to find latest checkpoint in the base directory
        base_dir = os.path.dirname(checkpoint_path) or "outputs_qwen2.5_7b_sft"
        if os.path.exists(base_dir):
            checkpoint_path = get_latest_checkpoint(base_dir)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # -------------------------------------------------------------------------
    # 1. Load Base Model and Add LoRA Adapters (unsloth way)
    # -------------------------------------------------------------------------
    logger.info("\n[1/5] Loading base model and adding LoRA adapters...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Add LoRA adapters using FastLanguageModel (unsloth-compatible way)
    logger.info("Adding LoRA adapters for continued training...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    logger.info(f"✅ Model loaded and LoRA adapters added (rank={LORA_R}, alpha={LORA_ALPHA})")
    logger.info(f"   Will resume from checkpoint: {checkpoint_path}")
    
    # -------------------------------------------------------------------------
    # 2. Load and Split Dataset
    # -------------------------------------------------------------------------
    logger.info("\n[2/5] Loading and splitting dataset...")
    
    dataset_dict = load_and_split_dataset(
        dataset_path=args.dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )
    
    # -------------------------------------------------------------------------
    # 3. Format Dataset
    # -------------------------------------------------------------------------
    logger.info("\n[3/5] Formatting dataset with prompt template...")
    
    formatted_dataset = format_dataset(dataset_dict, tokenizer)
    
    # Save test dataset for later evaluation
    save_test_dataset(dataset_dict["test"], output_dir)
    
    # Show sample examples
    logger.info("\n📝 Sample training examples:")
    for i in range(min(2, len(formatted_dataset["train"]))):
        text = formatted_dataset["train"]["text"][i]
        logger.info(f"\n--- Example {i+1} ---")
        logger.info(text[:500] + "..." if len(text) > 500 else text)
    
    # -------------------------------------------------------------------------
    # 4. Configure Trainer with resume_from_checkpoint
    # -------------------------------------------------------------------------
    logger.info("\n[4/5] Configuring SFT Trainer...")
    
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
            report_to="none",  # Disable wandb/tensorboard
        ),
    )
    
    logger.info("✅ Trainer configured")
    logger.info(f"  • Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}")
    logger.info(f"  • Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    logger.info(f"  • Effective batch size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    logger.info(f"  • Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    logger.info(f"  • Learning rate: {TRAINING_CONFIG['learning_rate']}")
    logger.info(f"  • Eval every: {TRAINING_CONFIG['eval_steps']} steps")
    logger.info(f"  • Resuming from: {checkpoint_path}")
    
    # -------------------------------------------------------------------------
    # 5. Train (resuming from checkpoint)
    # -------------------------------------------------------------------------
    logger.info("\n[5/5] Starting continued training...")
    
    trainer_stats = trainer.train(resume_from_checkpoint=checkpoint_path)
    
    logger.info("\n✅ Training complete!")
    logger.info(f"  • Total steps: {trainer_stats.global_step}")
    logger.info(f"  • Training loss: {trainer_stats.training_loss:.4f}")
    
    # -------------------------------------------------------------------------
    # 6. Save Final Model
    # -------------------------------------------------------------------------
    logger.info("\n[6/6] Saving final model...")
    
    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    logger.info(f"✅ Model saved to: {final_model_dir}")
    
    # Save training info
    training_info = {
        "base_model": BASE_MODEL_NAME,
        "checkpoint_used": checkpoint_path,
        "dataset": args.dataset,
        "train_samples": len(formatted_dataset["train"]),
        "val_samples": len(formatted_dataset["validation"]),
        "test_samples": len(formatted_dataset["test"]),
        "epochs": TRAINING_CONFIG["num_train_epochs"],
        "final_loss": trainer_stats.training_loss,
        "total_steps": trainer_stats.global_step,
        "output_dir": output_dir,
        "timestamp": timestamp,
    }
    
    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    logger.info(f"Training info saved to: {info_path}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🎉 CONTINUED TRAINING COMPLETE!")
    print("=" * 80)
    logger.info("\nSummary:")
    logger.info(f"  ✅ Base checkpoint: {checkpoint_path}")
    logger.info(f"  ✅ New dataset: {args.dataset}")
    logger.info(f"  ✅ Training samples: {len(formatted_dataset['train'])}")
    logger.info(f"  ✅ Validation samples: {len(formatted_dataset['validation'])}")
    logger.info(f"  ✅ Test samples: {len(formatted_dataset['test'])}")
    logger.info(f"  ✅ Final model: {final_model_dir}")
    logger.info(f"  ✅ Test data saved for evaluation")
    logger.info("\nNext steps:")
    logger.info(f"  1. Evaluate on test set using the saved test_dataset.jsonl")
    logger.info(f"  2. Compare with previous checkpoint performance")
    logger.info(f"  3. If satisfied, merge adapters and push to HuggingFace")
    print("=" * 80 + "\n")
    
    return final_model_dir


def main():
    parser = argparse.ArgumentParser(
        description="Continue fine-tuning Qwen2.5-7B from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_PATH,
        help=f"Path to checkpoint to resume from (default: {CHECKPOINT_PATH})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help=f"Path to new training dataset (default: {DATASET_PATH})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train (default: 3)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    
    args = parser.parse_args()
    
    # Override config with args
    if args.epochs != 3:
        TRAINING_CONFIG["num_train_epochs"] = args.epochs
    if args.lr != 1e-4:
        TRAINING_CONFIG["learning_rate"] = args.lr
    
    run_continued_training(args)


if __name__ == "__main__":
    main()
