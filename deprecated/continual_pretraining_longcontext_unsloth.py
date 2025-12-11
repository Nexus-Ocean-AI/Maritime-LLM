"""
Unsloth-Optimized Progressive Long-Context Continual Pre-Training
Uses Unsloth for 2-5x faster training on H100 while maintaining quality.

Installation:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
"""

import os
import glob
import torch
import time
from datetime import datetime, timedelta

# Silence TRL experimental warnings
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

# Import unsloth first (as recommended)
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported

from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer

# Custom Progress Tracking Callback
class EnhancedProgressCallback(TrainerCallback):
    """Enhanced progress tracking across phases and epochs."""
    
    def __init__(self, phase_num, total_phases, phase_name):
        self.phase_num = phase_num
        self.total_phases = total_phases
        self.phase_name = phase_name
        self.phase_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.phase_start_time = time.time()
        print(f"\n🚀 Starting {self.phase_name} (Phase {self.phase_num}/{self.total_phases})")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            elapsed = time.time() - self.phase_start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            progress = state.global_step / state.max_steps if state.max_steps else 0
            progress_bar = "█" * int(progress * 30) + "░" * (30 - int(progress * 30))
            
            print(f"\r[{self.phase_name}] "
                  f"Epoch {state.epoch:.2f} | "
                  f"Step {state.global_step}/{state.max_steps} | "
                  f"Loss: {logs['loss']:.4f} | "
                  f"LR: {logs.get('learning_rate', 0):.2e} | "
                  f"Elapsed: {elapsed_str} | "
                  f"[{progress_bar}] {progress*100:.1f}%", 
                  end='', flush=True)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n✅ Epoch {int(state.epoch)} completed!")
        
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.phase_start_time
        print(f"\n\n✅ {self.phase_name} Complete!")
        print(f"⏱️  Phase time: {str(timedelta(seconds=int(total_time)))}")
        print(f"📊 Final loss: {state.log_history[-1].get('loss', 'N/A'):.4f}")
        print(f"{'='*80}\n")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"  # Unsloth's optimized version
DATA_DIR = "data"
OUTPUT_DIR = "qwen-maritime-longcontext-cpt-unsloth"

# Epoch-Based Progressive Context Schedule
CONTEXT_SCHEDULE = [
    {"name": "Phase_1a_Short", "max_seq_length": 2048, "num_epochs": 1},
    {"name": "Phase_1b_Medium", "max_seq_length": 16384, "num_epochs": 2},
    {"name": "Phase_1c_Long", "max_seq_length": 32768, "num_epochs": 4},
]

# LoRA Config
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE_PER_PHASE = {
    2048: 8,      # H100-optimized (Unsloth uses less memory)
    16384: 3,     # Increased from 2 (Unsloth allows larger batches)
    32768: 2,     # Increased from 1 (Unsloth's magic)
}
GRAD_ACCUMULATION_PER_PHASE = {
    2048: 4,
    16384: 6,
    32768: 8,
}
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = None

# Replay Buffer
REPLAY_DATASET_NAME = "wikitext"
REPLAY_DATASET_CONFIG = "wikitext-103-raw-v1"
REPLAY_RATIO = 0.15

def find_all_data_files(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    files = [f for f in files if "maritime_pretraining_data" in f]
    print(f"Found {len(files)} data files: {files}")
    return files

def load_and_mix_data():
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
            
            print(f"Mixing {len(replay_dataset)} replay examples...")
            dataset = concatenate_datasets([dataset, replay_dataset])
            dataset = dataset.shuffle(seed=42)
        except Exception as e:
            print(f"Warning: Could not load replay dataset: {e}")

    return dataset

def formatting_func(example):
    """Format for Unsloth's SFTTrainer - just return the text."""
    return example["text"]

def train_phase(model, tokenizer, dataset, phase_config, phase_num, total_phases):
    """Train a single phase with Unsloth optimization."""
    seq_length = phase_config["max_seq_length"]
    phase_name = phase_config["name"]
    num_epochs = phase_config["num_epochs"]
    
    print(f"\n{'='*80}")
    print(f"PHASE {phase_num}/{total_phases}: {phase_name}")
    print(f"Sequence Length: {seq_length:,} tokens")
    print(f"Training Epochs: {num_epochs}")
    print(f"{'='*80}\n")
    
    # Get hyperparameters
    batch_size = BATCH_SIZE_PER_PHASE.get(seq_length, 1)
    grad_accum = GRAD_ACCUMULATION_PER_PHASE.get(seq_length, 8)
    
    print(f"Phase Training Plan:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Dataset Size: {len(dataset)} examples")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gradient Accumulation: {grad_accum}")
    print(f"  Effective Batch Size: {batch_size * grad_accum}")
    print(f"  Max Sequence Length: {seq_length:,}\n")
    
    # Training Arguments
    output_dir = f"{OUTPUT_DIR}/{phase_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        num_train_epochs=num_epochs,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",  # Unsloth-optimized optimizer
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_dir=f"{output_dir}/logs",
        report_to="tensorboard",
    )
    
    # Unsloth's SFTTrainer (optimized for speed)
    # Use formatting_func for compatibility with current TRL version
    def formatting_prompts_func(examples):
        return examples["text"]
    
    # Unsloth's SFTTrainer needs the tokenizer explicitly
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # Unsloth requires this
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        callbacks=[EnhancedProgressCallback(phase_num, total_phases, phase_name)]
    )
    
    print(f"Starting Phase {phase_num} training with Unsloth optimization...")
    print(f"Monitor: tensorboard --logdir {output_dir}/logs\n")
    
    # Unsloth automatically handles gradient checkpointing and optimizations
    trainer.train()
    
    print(f"Saving Phase {phase_num} checkpoint...")
    trainer.save_model(output_dir)
    
    return model

def main():
    print(f"\n{'#'*80}")
    print(f"# Unsloth-Optimized Progressive Long-Context CPT")
    print(f"# Model: {MODEL_ID}")
    print(f"# Target Context: 32K tokens")
    print(f"# Expected Speedup: 2-5x faster than standard training")
    print(f"{'#'*80}\n")
    
    # Load Data (once for all phases)
    dataset = load_and_mix_data()
    print(f"\nTotal dataset size: {len(dataset)} examples")
    
    # Load model with Unsloth (automatically optimized)
    print(f"\nLoading Unsloth-optimized model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=32768,  # Set to max we'll use
        dtype=None,  # Auto-detect
        load_in_4bit=False,  # Use full precision for quality
    )
    print("✅ Model loaded with Unsloth optimizations")
    
    # Apply Unsloth-optimized LoRA
    print("\nApplying Unsloth LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
    )
    print("✅ LoRA applied with Unsloth optimizations")
    
    # Progressive Training
    for phase_num, phase_config in enumerate(CONTEXT_SCHEDULE, 1):
        model = train_phase(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            phase_config=phase_config,
            phase_num=phase_num,
            total_phases=len(CONTEXT_SCHEDULE)
        )
    
    # Final Save
    final_output = f"{OUTPUT_DIR}/final_32k_model"
    print(f"\nSaving final model to {final_output}...")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    
    # Save also as merged model (Unsloth makes this fast)
    print("\nCreating merged model (Unsloth fast merge)...")
    model.save_pretrained_merged(
        f"{final_output}_merged",
        tokenizer,
        save_method="merged_16bit",  # or "merged_4bit" for smaller size
    )
    
    total_epochs = sum(phase["num_epochs"] for phase in CONTEXT_SCHEDULE)
    print(f"\n{'='*80}")
    print("✅ Unsloth-Optimized Training Complete!")
    print(f"✅ Total Epochs: {total_epochs}")
    print(f"✅ Final Context: 32K tokens")
    print(f"✅ LoRA adapter: {final_output}")
    print(f"✅ Merged model: {final_output}_merged")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
