"""
Fine-tune Qwen2.5-7B for Maritime Domain

This script performs:
1. Merges the base Qwen2.5-7B model with LoRA adapters from Domain Adaptive Pretraining
2. Runs Supervised Fine-Tuning (SFT) on the merged model

Usage:
    # Default: Merge phase1a adapter and run SFT
    python finetune_qwen2.5_7b.py
    
    # Merge phase1b adapter instead
    python finetune_qwen2.5_7b.py --adapter phase1b
    
    # Skip merge if model already merged
    python finetune_qwen2.5_7b.py --skip-merge --merged-model-path ./qwen-maritime-7b-merged-phase1a
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Import merge function from existing module
try:
    from merge_lora_adapters import merge_and_save, ADAPTERS as MERGE_ADAPTERS, BASE_MODEL_ID
except ImportError:
    merge_and_save = None
    MERGE_ADAPTERS = {}
    BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Base model for direct SFT (without adapter merging)
BASE_INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# SFT Configuration
DATASET_PATH = "processed_queries_20251216_014609.jsonl"
MAX_SEQ_LENGTH = 4096  # Increased for longer maritime documents
OUTPUT_DIR = "outputs_qwen2.5_7b_sft"

# LoRA Config for SFT (applied on merged model)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters
TRAINING_CONFIG = {
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,  # Effective batch size = 16 * 4 = 64
    "warmup_ratio": 0.03,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "bf16": True,
}


# =============================================================================
# Step 1: Merge LoRA Adapters
# =============================================================================

def merge_adapters(adapter_key: str = "phase1a") -> str:
    """
    Merge the base Qwen2.5-7B model with LoRA adapters from domain adaptive pretraining.
    
    Args:
        adapter_key: Which adapter to merge ('phase1a' or 'phase1b')
        
    Returns:
        Path to the merged model
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: MERGING BASE MODEL WITH DOMAIN ADAPTIVE PRETRAINING ADAPTERS")
    logger.info("=" * 80 + "\n")
    
    if adapter_key not in MERGE_ADAPTERS:
        raise ValueError(f"Unknown adapter key: {adapter_key}. Available: {list(MERGE_ADAPTERS.keys())}")
    
    config = MERGE_ADAPTERS[adapter_key]
    logger.info(f"Selected adapter: {config['name']}")
    logger.info(f"Adapter path: {config['adapter_path']}")
    logger.info(f"Output path: {config['output_path']}")
    
    # Call the existing merge function
    merged_model_path = merge_and_save(adapter_key, config, BASE_MODEL_ID)
    
    logger.info(f"✅ Merge complete! Model saved to: {merged_model_path}")
    return merged_model_path


# =============================================================================
# Step 2: Supervised Fine-Tuning (SFT)
# =============================================================================

def run_sft(merged_model_path: str):
    """
    Perform Supervised Fine-Tuning on the merged model.
    
    Args:
        merged_model_path: Path to the merged model from Step 1
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: SUPERVISED FINE-TUNING (SFT)")
    logger.info("=" * 80 + "\n")
    
    # Import unsloth here (after merge) to avoid conflicts
    from unsloth import FastLanguageModel
    from datasets import load_dataset, Dataset
    from trl import SFTTrainer, SFTConfig
    
    # -------------------------------------------------------------------------
    # 2.1 Load Merged Model
    # -------------------------------------------------------------------------
    logger.info(f"[2.1] Loading merged model from: {merged_model_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=merged_model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto detect
        load_in_4bit=True,  # 4-bit QLoRA for efficient training
    )
    logger.info("✅ Merged model loaded successfully")
    
    # -------------------------------------------------------------------------
    # 2.2 Add LoRA Adapters for SFT
    # -------------------------------------------------------------------------
    logger.info(f"\n[2.2] Adding LoRA adapters for SFT...")
    
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
    logger.info(f"✅ LoRA adapters added (rank={LORA_R}, alpha={LORA_ALPHA})")
    
    # -------------------------------------------------------------------------
    # 2.3 Prepare Dataset
    # -------------------------------------------------------------------------
    logger.info(f"\n[2.3] Loading dataset from: {DATASET_PATH}...")
    
    # Prompt template for maritime QA
    prompt_template = """<|im_start|>system
You are a highly knowledgeable maritime expert. You are provided with a query related to maritime regulations, safety standards, or operational manuals. Answer the query accurately and comprehensively.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}"""
    
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples):
        """Format examples into prompt-response pairs."""
        queries = examples.get("query", [])
        answers = examples.get("answer", [])
        
        texts = []
        for query, answer in zip(queries, answers):
            text = prompt_template.format(query, answer) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    try:
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
        dataset = dataset.map(formatting_prompts_func, batched=True)
        logger.info(f"✅ Dataset loaded: {len(dataset)} examples")
        
        # Show sample examples
        logger.info("\n📝 Sample training examples:")
        for i in range(min(3, len(dataset))):
            logger.info(f"\n--- Example {i+1} ---")
            logger.info(dataset["text"][i][:500] + "..." if len(dataset["text"][i]) > 500 else dataset["text"][i])
            
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        logger.info("Creating dummy dataset for demonstration...")
        
        dummy_data = {
            "query": [
                "What is the SOLAS convention?",
                "Explain MARPOL Annex VI regulations.",
                "What is the ISM Code?"
            ],
            "answer": [
                "SOLAS (Safety of Life at Sea) is an international maritime treaty that sets minimum safety standards for ships...",
                "MARPOL Annex VI addresses the prevention of air pollution from ships, including regulations on SOx and NOx emissions...",
                "The ISM Code (International Safety Management Code) provides an international standard for the safe management and operation of ships..."
            ]
        }
        dataset = Dataset.from_dict(dummy_data)
        dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # -------------------------------------------------------------------------
    # 2.4 Configure Training
    # -------------------------------------------------------------------------
    logger.info(f"\n[2.4] Configuring SFT Trainer...")
    
    sft_output_dir = os.path.join(OUTPUT_DIR, Path(merged_model_path).name)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
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
            output_dir=sft_output_dir,
            bf16=TRAINING_CONFIG["bf16"],
        ),
    )
    
    logger.info(f"✅ Trainer configured. Output dir: {sft_output_dir}")
    
    # -------------------------------------------------------------------------
    # 2.5 Train
    # -------------------------------------------------------------------------
    logger.info(f"\n[2.5] Starting SFT training...")
    logger.info(f"  • Batch size: {TRAINING_CONFIG['per_device_train_batch_size']}")
    logger.info(f"  • Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    logger.info(f"  • Effective batch size: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    logger.info(f"  • Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    logger.info(f"  • Learning rate: {TRAINING_CONFIG['learning_rate']}")
    
    trainer_stats = trainer.train()
    
    logger.info("\n✅ Training complete!")
    logger.info(f"  • Total steps: {trainer_stats.global_step}")
    logger.info(f"  • Training loss: {trainer_stats.training_loss:.4f}")
    
    # -------------------------------------------------------------------------
    # 2.6 Save Final Model
    # -------------------------------------------------------------------------
    logger.info(f"\n[2.6] Saving fine-tuned model...")
    
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    logger.info(f"✅ Model saved to: {final_model_dir}")
    
    return final_model_dir


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-7B with domain adaptive pretraining + SFT"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        choices=list(MERGE_ADAPTERS.keys()) if MERGE_ADAPTERS else ["phase1a", "phase1b"],
        default="phase1a",
        help="Which adapter to merge before SFT (default: phase1a)"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the merge step (use if model is already merged)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Run SFT directly on base Qwen-2.5-7B-Instruct without merging any adapters"
    )
    parser.add_argument(
        "--merged-model-path",
        type=str,
        default=None,
        help="Path to already-merged model (required if --skip-merge is set)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_INSTRUCT_MODEL,
        help=f"Base model to use for --no-merge mode (default: {BASE_INSTRUCT_MODEL})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help=f"Path to training dataset JSONL file (default: {DATASET_PATH})"
    )
    
    args = parser.parse_args()
    
    # Update dataset path from args
    dataset_path = args.dataset
    
    print("\n" + "=" * 80)
    print("🚢 MARITIME QWEN2.5-7B FINE-TUNING PIPELINE")
    print("=" * 80 + "\n")
    
    logger.info("Configuration:")
    logger.info(f"  • Mode: {'No merge (base model)' if args.no_merge else ('Skip merge' if args.skip_merge else 'Merge + SFT')}")
    if not args.no_merge:
        logger.info(f"  • Adapter: {args.adapter}")
    logger.info(f"  • Dataset: {dataset_path}")
    logger.info(f"  • Output: {OUTPUT_DIR}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Merge Adapters (or use existing / base model)
    # -------------------------------------------------------------------------
    if args.no_merge:
        # Use base Instruct model directly without any adapter merging
        merged_model_path = args.base_model
        logger.info(f"\n🚀 Running SFT directly on base model: {merged_model_path}")
        logger.info("   (No domain adaptive pretraining adapters will be merged)")
    elif args.skip_merge:
        if not args.merged_model_path:
            logger.error("❌ --merged-model-path is required when using --skip-merge")
            sys.exit(1)
        
        if not os.path.exists(args.merged_model_path):
            logger.error(f"❌ Merged model path not found: {args.merged_model_path}")
            sys.exit(1)
            
        merged_model_path = args.merged_model_path
        logger.info(f"\n⏩ Skipping merge. Using existing model: {merged_model_path}")
    else:
        if merge_and_save is None:
            logger.error("❌ merge_lora_adapters module not available. Use --no-merge to skip.")
            sys.exit(1)
        merged_model_path = merge_adapters(args.adapter)
    
    # -------------------------------------------------------------------------
    # STEP 2: Supervised Fine-Tuning
    # -------------------------------------------------------------------------
    final_model_path = run_sft(merged_model_path)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)
    logger.info("\nSummary:")
    logger.info(f"  ✅ Base/Merged model: {merged_model_path}")
    logger.info(f"  ✅ SFT model: {final_model_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Evaluate the model on maritime QA benchmarks")
    logger.info("  2. Merge SFT adapters if needed:")
    logger.info(f"     python merge_lora_adapters.py --adapter-path {final_model_path}")
    logger.info("  3. Upload to HuggingFace:")
    logger.info(f"     model.push_to_hub('your-username/qwen2.5-7b-maritime-sft')")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
