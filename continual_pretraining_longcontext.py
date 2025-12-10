import os
import glob
import math
import torch
import time
from datetime import datetime, timedelta
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Custom Progress Tracking Callback
class EnhancedProgressCallback(TrainerCallback):
    """Enhanced progress tracking across phases and epochs."""
    
    def __init__(self, phase_num, total_phases, phase_name):
        self.phase_num = phase_num
        self.total_phases = total_phases
        self.phase_name = phase_name
        self.phase_start_time = None
        self.last_log_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.phase_start_time = time.time()
        self.last_log_time = time.time()
        print(f"\n🚀 Starting {self.phase_name} (Phase {self.phase_num}/{self.total_phases})")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called every logging_steps - display enhanced progress."""
        if logs and 'loss' in logs:
            current_time = time.time()
            elapsed = current_time - self.phase_start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            # Calculate progress
            progress = state.global_step / state.max_steps if state.max_steps else 0
            progress_bar = "█" * int(progress * 30) + "░" * (30 - int(progress * 30))
            
            # Display
            print(f"\r[{self.phase_name}] "
                  f"Epoch {state.epoch:.2f} | "
                  f"Step {state.global_step}/{state.max_steps} | "
                  f"Loss: {logs['loss']:.4f} | "
                  f"LR: {logs.get('learning_rate', 0):.2e} | "
                  f"Elapsed: {elapsed_str} | "
                  f"[{progress_bar}] {progress*100:.1f}%", 
                  end='', flush=True)
            
            self.last_log_time = current_time
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n✅ Epoch {int(state.epoch)} completed!")
        
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.phase_start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print(f"\n\n✅ {self.phase_name} Complete!")
        print(f"⏱️  Phase training time: {total_time_str}")
        print(f"📊 Final loss: {state.log_history[-1].get('loss', 'N/A'):.4f}")
        print(f"{'='*80}\n")

# -----------------------------------------------------------------------------
# Configuration - Epoch-Based Progressive Long Context Training (32K Target)
# -----------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # 128K native context window
DATA_DIR = "data"
OUTPUT_DIR = "qwen-maritime-longcontext-cpt"

# Epoch-Based Progressive Context Schedule
# Each phase trains on the ENTIRE dataset for N epochs at a specific sequence length
# This ensures the model learns both short-range and long-range patterns in all documents
PROGRESSIVE_TRAINING = True
CONTEXT_SCHEDULE = [
    {"name": "Phase_1a_Short", "max_seq_length": 2048, "num_epochs": 1},    # 1 epoch at 2K - Fast domain learning
    {"name": "Phase_1b_Medium", "max_seq_length": 16384, "num_epochs": 2},  # 2 epochs at 16K - Medium context
    {"name": "Phase_1c_Long", "max_seq_length": 32768, "num_epochs": 4},    # 4 epochs at 32K - Master long context
]

# Or set single length for direct training (NOT RECOMMENDED for 32K)
SINGLE_MAX_SEQ_LENGTH = 32768  # Only used if PROGRESSIVE_TRAINING = False

# TRAINING MODE
USE_LORA = True  # Must be True for 32K training on consumer hardware

# LoRA Config - Optimized for Long Context
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE_PER_PHASE = {
    2048: 4,      # Can afford larger batch for short sequences
    16384: 1,     # Reduce for medium
    32768: 1,     # Minimum for long sequences
}
GRAD_ACCUMULATION_PER_PHASE = {
    4096: 8,      # Effective batch = 32
    16384: 16,    # Effective batch = 16
    32768: 32,    # Effective batch = 32
}
WARMUP_RATIO = 0.03
SAVE_STEPS = 250
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = None  # Keep ALL checkpoints (all 7 epoch adapters)

# Replay Buffer
REPLAY_DATASET_NAME = "wikitext"
REPLAY_DATASET_CONFIG = "wikitext-103-raw-v1"
REPLAY_RATIO = 0.15

# Memory Optimization
USE_FLASH_ATTENTION_2 = False  # Disabled - install with: pip install flash-attn --no-build-isolation
USE_8BIT = False  # Set True if OOM errors occur

def find_all_data_files(data_dir):
    """Finds all jsonl files in the data directory."""
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    files = [f for f in files if "maritime_pretraining_data" in f]
    print(f"Found {len(files)} data files: {files}")
    return files

def load_and_mix_data():
    """Loads maritime data and optionally mixes with replay data."""
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
    Critical for efficient long-context training.
    """
    all_input_ids = []
    for text in examples["text"]:
        tokens = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        all_input_ids.extend(tokens["input_ids"])
        all_input_ids.append(tokenizer.eos_token_id)
    
    packed = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i in range(0, len(all_input_ids), max_length):
        chunk = all_input_ids[i:i + max_length]
        if len(chunk) == max_length:
            packed["input_ids"].append(chunk)
            packed["attention_mask"].append([1] * max_length)
            packed["labels"].append(chunk)
    
    return packed

def calculate_phase_steps(phase_config, dataset_size, batch_size, grad_accum):
    """Calculate training steps for a specific phase based on epochs."""
    num_epochs = phase_config["num_epochs"]
    seq_length = phase_config["max_seq_length"]
    effective_batch = batch_size * grad_accum
    
    # Steps per epoch
    steps_per_epoch = dataset_size // effective_batch
    max_steps = steps_per_epoch * num_epochs
    
    # Calculate total tokens
    total_tokens = max_steps * effective_batch * seq_length
    
    return max_steps, total_tokens, steps_per_epoch

def train_phase(model, tokenizer, dataset, phase_config, phase_num, total_phases):
    """Train a single phase of progressive context length training."""
    seq_length = phase_config["max_seq_length"]
    phase_name = phase_config["name"]
    num_epochs = phase_config["num_epochs"]
    
    print(f"\n{'='*80}")
    print(f"PHASE {phase_num}/{total_phases}: {phase_name}")
    print(f"Sequence Length: {seq_length:,} tokens")
    print(f"Training Epochs: {num_epochs}")
    print(f"{'='*80}\n")
    
    # Pack sequences for this phase's length
    print("Packing sequences...")
    packed_dataset = dataset.map(
        lambda x: pack_sequences(x, tokenizer, seq_length),
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc=f"Packing for {seq_length} tokens"
    )
    print(f"Packed dataset size: {len(packed_dataset)} sequences")
    
    # Get phase-specific hyperparameters
    batch_size = BATCH_SIZE_PER_PHASE.get(seq_length, 1)
    grad_accum = GRAD_ACCUMULATION_PER_PHASE.get(seq_length, 32)
    max_steps, total_tokens, steps_per_epoch = calculate_phase_steps(
        phase_config, len(packed_dataset), batch_size, grad_accum
    )
    
    print(f"\nPhase Training Plan:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Dataset Size: {len(packed_dataset)} sequences")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gradient Accumulation: {grad_accum}")
    print(f"  Effective Batch Size: {batch_size * grad_accum}")
    print(f"  Steps per Epoch: {steps_per_epoch:,}")
    print(f"  Total Steps: {max_steps:,}")
    print(f"  Total Tokens: {total_tokens:,.0f}")
    print(f"  Tokens/Step: {batch_size * grad_accum * seq_length:,}\n")
    
    # Determine dtype
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or torch.backends.mps.is_available() else torch.float16
    
    # Training Arguments for this phase
    output_dir = f"{OUTPUT_DIR}/{phase_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        num_train_epochs=num_epochs,  # Use epochs instead of max_steps
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",  # Save after each epoch
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        logging_dir=f"{output_dir}/logs",
        report_to="tensorboard",
        dataloader_num_workers=2,
        gradient_checkpointing=True,  # Essential for long context
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EnhancedProgressCallback(phase_num, total_phases, phase_name)]
    )
    
    print(f"Starting Phase {phase_num} training...")
    print(f"Monitor: tensorboard --logdir {output_dir}/logs\n")
    trainer.train()
    
    # Save phase checkpoint
    print(f"Saving Phase {phase_num} checkpoint...")
    trainer.save_model(output_dir)
    
    return model

def main():
    print(f"\n{'#'*80}")
    print(f"# Progressive Long-Context Continual Pre-Training")
    print(f"# Model: {MODEL_ID}")
    print(f"# Target Context: 32K tokens")
    print(f"# Training Strategy: Progressive Length Extension")
    print(f"{'#'*80}\n")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Data (once for all phases)
    dataset = load_and_mix_data()
    print(f"\nTotal dataset size: {len(dataset)} examples")
    
    # 3. Load Model
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or torch.backends.mps.is_available() else torch.float16
    
    print(f"\nLoading model in {dtype}...")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "use_cache": False,
    }
    
    if USE_FLASH_ATTENTION_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2 for memory efficiency")
    
    if USE_8BIT:
        model_kwargs["load_in_8bit"] = True
        print("Loading in 8-bit mode for reduced memory")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    
    # 4. Apply LoRA
    if USE_LORA:
        if USE_8BIT:
            model = prepare_model_for_kbit_training(model)
        
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
        print("WARNING: Full fine-tuning at 32K context requires 200GB+ VRAM")
        model.gradient_checkpointing_enable()
    
    # 5. Progressive Training
    if PROGRESSIVE_TRAINING:
        for phase_num, phase_config in enumerate(CONTEXT_SCHEDULE, 1):
            model = train_phase(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                phase_config=phase_config,
                phase_num=phase_num,
                total_phases=len(CONTEXT_SCHEDULE)
            )
    else:
        print("WARNING: Direct 32K training is memory intensive!")
        # Use single phase training
        phase_config = {
            "name": "Direct_32K",
            "max_seq_length": SINGLE_MAX_SEQ_LENGTH,
            "num_epochs": 3  # Default to 3 epochs for single-phase training
        }
        model = train_phase(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            phase_config=phase_config,
            phase_num=1,
            total_phases=1
        )
    
    # 6. Final Save
    final_output = f"{OUTPUT_DIR}/final_32k_model"
    print(f"\nSaving final model to {final_output}...")
    
    # For LoRA, we need to save the adapter
    if USE_LORA:
        model.save_pretrained(final_output)
    else:
        model.save_pretrained(final_output)
    
    tokenizer.save_pretrained(final_output)
    
    # Calculate total training statistics
    total_epochs = sum(phase["num_epochs"] for phase in CONTEXT_SCHEDULE)
    print(f"\n{'='*80}")
    print("✅ Progressive Long-Context Training Complete!")
    print(f"✅ Total Epochs Trained: {total_epochs}")
    print(f"✅ Final Context Capability: 32K tokens")
    print(f"✅ Final model saved to: {final_output}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
