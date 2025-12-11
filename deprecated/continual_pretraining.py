import os
import glob
import math
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # Consider using "Qwen/Qwen2.5-7B" (base) for better CPT
DATA_DIR = "data"
OUTPUT_DIR = "qwen-maritime-cpt"
MAX_SEQ_LENGTH = 2048

# TRAINING MODE: Set to False for Full Fine-Tuning (Requires massive VRAM/RAM)
USE_LORA = True 

# LoRA Config (High Rank for better Domain Adaptation)
LORA_R = 64          # High rank for 0.5B token domain adaptation
LORA_ALPHA = 128     # Usually 2x Rank
LORA_DROPOUT = 0.05

# Training Hyperparameters
TARGET_TOKENS = 500_000_000  # 0.5 Billion tokens
LEARNING_RATE = 1e-4 if USE_LORA else 2e-5
BATCH_SIZE = 4       # Increased from 2 (no padding waste now)
GRAD_ACCUMULATION_STEPS = 8  # Effective batch size = 32
WARMUP_RATIO = 0.03
SAVE_STEPS = 500
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3  # Keep only last 3 checkpoints

# Replay Buffer (CRITICAL for preventing catastrophic forgetting)
REPLAY_DATASET_NAME = "wikitext"  # Enable replay by default
REPLAY_DATASET_CONFIG = "wikitext-103-raw-v1"
REPLAY_RATIO = 0.15  # 15% general domain data

def find_all_data_files(data_dir):
    """Finds all jsonl files in the data directory."""
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    # Exclude the output file itself
    files = [f for f in files if "maritime_pretraining_data" in f]
    print(f"Found {len(files)} data files: {files}")
    return files

def load_and_mix_data():
    """
    Loads maritime data and optionally mixes with replay data.
    """
    data_files = find_all_data_files(DATA_DIR)
    if not data_files:
        raise ValueError(f"No .jsonl files found in {DATA_DIR}")

    print("Loading maritime text data...")
    dataset = load_dataset("json", data_files=data_files, split="train")

    if REPLAY_DATASET_NAME:
        print(f"Loading replay data: {REPLAY_DATASET_NAME}...")
        try:
            replay_dataset = load_dataset(REPLAY_DATASET_NAME, REPLAY_DATASET_CONFIG, split="train")
            total_maritime = len(dataset)
            replay_amount = int(total_maritime * REPLAY_RATIO)
            replay_dataset = replay_dataset.select(range(min(replay_amount, len(replay_dataset))))
            
            print(f"Mixing {len(replay_dataset)} replay examples with {len(dataset)} maritime examples...")
            dataset = concatenate_datasets([dataset, replay_dataset])
            dataset = dataset.shuffle(seed=42)
        except Exception as e:
            print(f"Warning: Could not load replay dataset: {e}")
            print("Continuing without replay (higher forgetting risk)")

    return dataset

def pack_sequences(examples, tokenizer, max_length):
    """
    Pack multiple sequences into one to avoid padding waste.
    This is CRITICAL for efficient training.
    """
    # Tokenize without padding
    all_input_ids = []
    for text in examples["text"]:
        tokens = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        all_input_ids.extend(tokens["input_ids"])
        all_input_ids.append(tokenizer.eos_token_id)  # Separate documents
    
    # Pack into chunks of max_length
    packed = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i in range(0, len(all_input_ids), max_length):
        chunk = all_input_ids[i:i + max_length]
        if len(chunk) == max_length:  # Only keep full chunks
            packed["input_ids"].append(chunk)
            packed["attention_mask"].append([1] * max_length)
            packed["labels"].append(chunk)  # For CLM, labels = input_ids
    
    return packed

def calculate_max_steps(dataset_size, batch_size, grad_accum, target_tokens, avg_seq_len):
    """
    Calculate max_steps needed to train on TARGET_TOKENS.
    """
    effective_batch = batch_size * grad_accum
    tokens_per_step = effective_batch * avg_seq_len
    max_steps = int(target_tokens / tokens_per_step)
    print(f"\n{'='*60}")
    print(f"Training Plan:")
    print(f"  Target Tokens: {TARGET_TOKENS:,}")
    print(f"  Avg Seq Length: {avg_seq_len}")
    print(f"  Effective Batch Size: {effective_batch}")
    print(f"  Tokens/Step: {tokens_per_step:,}")
    print(f"  Max Steps: {max_steps:,}")
    print(f"  Estimated Epochs: {max_steps * effective_batch / dataset_size:.2f}")
    print(f"{'='*60}\n")
    return max_steps

def main():
    print(f"Initializing Continual Pre-Training for {MODEL_ID}")
    print(f"Mode: {'LoRA (Rank ' + str(LORA_R) + ')' if USE_LORA else 'FULL FINE-TUNING'}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Data
    dataset = load_and_mix_data()
    print(f"\nRaw dataset size: {len(dataset)} examples")
    
    # 3. Pack sequences (CRITICAL for efficiency - saves 50-70% compute)
    print("Packing sequences to avoid padding waste...")
    packed_dataset = dataset.map(
        lambda x: pack_sequences(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names
    )
    print(f"Packed dataset size: {len(packed_dataset)} sequences")

    # 4. Calculate training steps
    max_steps = calculate_max_steps(
        dataset_size=len(packed_dataset),
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUMULATION_STEPS,
        target_tokens=TARGET_TOKENS,
        avg_seq_len=MAX_SEQ_LENGTH  # After packing, all sequences are max length
    )

    # 5. Load Model
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or torch.backends.mps.is_available() else torch.float16
    
    print(f"Loading model in {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=False
    )

    # 6. Apply LoRA or Full Fine-Tuning
    if USE_LORA:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("Full Fine-Tuning enabled. All parameters trainable.")
        # Only enable gradient checkpointing on model for FFT
        model.gradient_checkpointing_enable()

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_steps=max_steps,  # Use steps instead of epochs
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,  # Prevent disk filling
        evaluation_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        optim="adamw_torch",
        lr_scheduler_type="cosine",  # Re-warming and re-decaying
        warmup_ratio=WARMUP_RATIO,
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="tensorboard",  # Enable TensorBoard logging
        dataloader_num_workers=2,  # Faster data loading
        gradient_checkpointing=USE_LORA,  # Only for LoRA (model handles it for FFT)
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 9. Train
    print("\nStarting training...")
    print(f"View training progress: tensorboard --logdir {OUTPUT_DIR}/logs\n")
    trainer.train()

    # 10. Save
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
