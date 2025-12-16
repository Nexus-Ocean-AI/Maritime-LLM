"""
Fine-tune Qwen2.5-7B-Instruct for Maritime Domain using Unsloth

This script runs Supervised Fine-Tuning (SFT) directly on the base
Qwen2.5-7B-Instruct model using Unsloth for efficient training.

Usage:
    python finetune_qwen2.5_7b_unsloth.py
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Base Qwen2.5-7B Instruct model
MAX_SEQ_LENGTH = 4096  # Increased for longer maritime documents
DTYPE = None  # None for auto detection
LOAD_IN_4BIT = True  # 4-bit QLoRA - stable training with minimal accuracy loss

# Dataset configuration
# Assumes a JSONL file with 'query' and 'answer' fields.
DATASET_PATH = "processed_queries_20251216_014609.jsonl"

# LoRA Config - Optimized for 7B model
LORA_R = 64  # Good learning capacity
LORA_ALPHA = 128  # 2x rank scaling factor
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Output directory for checkpoints and final model
OUTPUT_DIR = "outputs_qwen2.5_7b_sft"

# -----------------------------------------------------------------------------
# 1. Load Model
# -----------------------------------------------------------------------------
print("=" * 80)
print("🚢 MARITIME QWEN2.5-7B FINE-TUNING (UNSLOTH)")
print("=" * 80)
print(f"\n[1/5] Loading model: {MODEL_NAME}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
print("✅ Model loaded successfully")

# -----------------------------------------------------------------------------
# 2. Add LoRA Adapters
# -----------------------------------------------------------------------------
print(f"\n[2/5] Adding LoRA adapters (rank={LORA_R}, alpha={LORA_ALPHA})...")

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
print("✅ LoRA adapters added")

# -----------------------------------------------------------------------------
# 3. Prepare Dataset
# -----------------------------------------------------------------------------
print(f"\n[3/5] Loading dataset from: {DATASET_PATH}")

# Prompt template for Qwen2.5 ChatML format
prompt = """<|im_start|>system
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
        # Must add EOS_TOKEN
        text = prompt.format(query, answer) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"✅ Dataset loaded: {len(dataset)} examples")
    
    print("\n📝 Sample training examples:")
    for i in range(min(3, len(dataset))):
        print(f"\n--- Example {i+1} ---")
        sample = dataset["text"][i]
        print(sample[:500] + "..." if len(sample) > 500 else sample)
        print("-" * 40)

except Exception as e:
    print(f"⚠️ Error loading dataset: {e}")
    print("Creating dummy dataset for demonstration...")
    from datasets import Dataset
    
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
    print(f"✅ Dummy dataset created: {len(dataset)} examples")

# -----------------------------------------------------------------------------
# 4. Train
# -----------------------------------------------------------------------------
print(f"\n[4/5] Starting training...")
print(f"  • Batch size: 16")
print(f"  • Gradient accumulation: 4") 
print(f"  • Effective batch size: 64")
print(f"  • Epochs: 3")
print(f"  • Learning rate: 2e-4")
print(f"  • Output: {OUTPUT_DIR}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        per_device_train_batch_size=16,  # Optimized for 7B on single GPU
        gradient_accumulation_steps=4,   # Effective batch size = 16 * 4 = 64
        warmup_ratio=0.03,               # 3% of total steps for warmup
        num_train_epochs=3,              # Full training on dataset
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=500,                  # Save checkpoints every 500 steps
        save_total_limit=3,              # Keep last 3 checkpoints
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",      # Cosine schedule for longer training
        seed=3407,
        output_dir=OUTPUT_DIR,
        bf16=True,                       # Use BF16 for faster training
    ),
)

trainer_stats = trainer.train()

print("\n✅ Training complete!")
print(f"  • Total steps: {trainer_stats.global_step}")
print(f"  • Training loss: {trainer_stats.training_loss:.4f}")

# -----------------------------------------------------------------------------
# 5. Save Model
# -----------------------------------------------------------------------------
print(f"\n[5/5] Saving model...")

FINAL_MODEL_DIR = f"{OUTPUT_DIR}/final_model"
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print(f"✅ Model saved to '{FINAL_MODEL_DIR}'")

# Print next steps
print("\n" + "=" * 80)
print("🎉 PIPELINE COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print(f"  1. Evaluate: python evaluate_checkpoints.py --model-path {FINAL_MODEL_DIR}")
print(f"  2. Merge adapters: model.save_pretrained_merged('{OUTPUT_DIR}/merged', tokenizer)")
print(f"  3. Upload to HuggingFace: model.push_to_hub('your-username/qwen2.5-7b-maritime-sft')")
print("=" * 80 + "\n")
