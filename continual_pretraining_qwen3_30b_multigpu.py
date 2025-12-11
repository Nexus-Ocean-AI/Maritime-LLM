"""
Multi-GPU Continual Pre-Training for Qwen3-30B
Target: 128K context window on 8×H100 GPUs with DeepSpeed ZeRO-3

This script uses standard HuggingFace + PEFT + DeepSpeed for multi-GPU training.
Based on the proven patterns from continual_pretraining_longcontext.py

Usage:
    accelerate launch --config_file accelerate_config.yaml continual_pretraining_qwen3_30b_multigpu.py
"""

import os
import glob
import json
import torch
import time
from datetime import datetime, timedelta

# Environment setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # Official Qwen3 30B MoE model
DATA_DIR = "data"
OUTPUT_DIR = "qwen3-30b-maritime-128K-multigpu"

# Progressive Context Schedule
CONTEXT_SCHEDULE = [
    {"name": "Phase_1_Domain_4K", "max_seq_length": 4096, "num_epochs": 2, "learning_rate": 2e-4},
    {"name": "Phase_2_Medium_16K", "max_seq_length": 16384, "num_epochs": 2, "learning_rate": 1e-4},
    {"name": "Phase_3_Long_32K", "max_seq_length": 32768, "num_epochs": 2, "learning_rate": 5e-5},
    {"name": "Phase_4_Extended_64K", "max_seq_length": 65536, "num_epochs": 2, "learning_rate": 2e-5},
    {"name": "Phase_5_Ultra_128K", "max_seq_length": 131072, "num_epochs": 3, "learning_rate": 1e-5},
]

# LoRA Configuration
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Batch sizes per context length (per GPU)
BATCH_SIZE_PER_PHASE = {
    4096: 2,
    16384: 1,
    32768: 1,
    65536: 1,
    131072: 1,
}

# Increased grad accumulation to maintain effective batch size
GRAD_ACCUMULATION_PER_PHASE = {
    4096: 8,
    16384: 16,
    32768: 32,
    65536: 32,
    131072: 32,
}

# Training settings
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2
REPLAY_RATIO = 0.10
DATALOADER_WORKERS = 4


# =============================================================================
# Enhanced Progress Callback
# =============================================================================

class EnhancedProgressCallback(TrainerCallback):
    """Enhanced progress tracking across phases and epochs."""
    
    def __init__(self, phase_num, total_phases, phase_name):
        self.phase_num = phase_num
        self.total_phases = total_phases
        self.phase_name = phase_name
        self.phase_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.phase_start_time = time.time()
        if state.is_local_process_zero:
            print(f"\n🚀 Starting {self.phase_name} (Phase {self.phase_num}/{self.total_phases})")
            print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called every logging_steps - display enhanced progress."""
        if logs and 'loss' in logs and state.is_local_process_zero:
            elapsed = time.time() - self.phase_start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            # Calculate progress
            progress = state.global_step / state.max_steps if state.max_steps else 0
            progress_bar = "█" * int(progress * 30) + "░" * (30 - int(progress * 30))
            
            # ETA calculation
            if state.global_step > 10:
                eta = (elapsed / state.global_step) * (state.max_steps - state.global_step)
                eta_str = str(timedelta(seconds=int(eta)))
            else:
                eta_str = "calculating..."
            
            # Memory usage
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                mem_str = f"{mem_used:.1f}/{mem_total:.0f}GB"
            else:
                mem_str = "N/A"
            
            print(f"\r[{self.phase_name}] "
                  f"Epoch {state.epoch:.2f} | "
                  f"Step {state.global_step}/{state.max_steps} | "
                  f"Loss: {logs['loss']:.4f} | "
                  f"LR: {logs.get('learning_rate', 0):.2e} | "
                  f"Mem: {mem_str} | "
                  f"ETA: {eta_str} | "
                  f"[{progress_bar}] {progress*100:.1f}%", 
                  end='', flush=True)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            print(f"\n✅ Epoch {int(state.epoch)} completed!")
        
    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            total_time = time.time() - self.phase_start_time
            total_time_str = str(timedelta(seconds=int(total_time)))
            final_loss = next((l.get('loss') for l in reversed(state.log_history) if 'loss' in l), None)
            print(f"\n\n✅ {self.phase_name} Complete!")
            print(f"⏱️  Phase training time: {total_time_str}")
            if final_loss:
                print(f"📊 Final loss: {final_loss:.4f}")
            print(f"{'='*80}\n")


# =============================================================================
# Token Statistics Tracking
# =============================================================================

_CURRENT_TOKEN_STATS = {}

def pack_sequences(examples, tokenizer, max_length):
    """
    Pack multiple sequences into one to maximize GPU utilization.
    Pads the last chunk instead of discarding it.
    """
    global _CURRENT_TOKEN_STATS
    
    all_input_ids = []
    total_raw_tokens = 0
    
    for text in examples["text"]:
        if text:  # Skip empty texts
            tokens = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
            all_input_ids.extend(tokens["input_ids"])
            all_input_ids.append(tokenizer.eos_token_id)
            total_raw_tokens += len(tokens["input_ids"]) + 1
    
    packed = {"input_ids": [], "attention_mask": [], "labels": []}
    total_real_tokens = 0
    total_padding_tokens = 0
    
    for i in range(0, len(all_input_ids), max_length):
        chunk = all_input_ids[i:i + max_length]
        
        if len(chunk) < max_length:
            padding_length = max_length - len(chunk)
            chunk = chunk + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * (max_length - padding_length) + [0] * padding_length
            labels = chunk[:max_length - padding_length] + [-100] * padding_length
            total_real_tokens += (max_length - padding_length)
            total_padding_tokens += padding_length
        else:
            attention_mask = [1] * max_length
            labels = chunk
            total_real_tokens += max_length
        
        packed["input_ids"].append(chunk)
        packed["attention_mask"].append(attention_mask)
        packed["labels"].append(labels)
    
    # Update global stats
    if not _CURRENT_TOKEN_STATS:
        _CURRENT_TOKEN_STATS = {
            "total_raw_tokens": 0,
            "total_real_tokens": 0,
            "total_padding_tokens": 0,
            "num_sequences": 0
        }
    
    _CURRENT_TOKEN_STATS["total_raw_tokens"] += total_raw_tokens
    _CURRENT_TOKEN_STATS["total_real_tokens"] += total_real_tokens
    _CURRENT_TOKEN_STATS["total_padding_tokens"] += total_padding_tokens
    _CURRENT_TOKEN_STATS["num_sequences"] += len(packed["input_ids"])
    
    return packed


# =============================================================================
# Data Loading
# =============================================================================

def find_data_files(data_dir):
    """Find the consolidated maritime training data file."""
    primary_file = os.path.join(data_dir, "maritime_pretraining_data.jsonl")
    if os.path.exists(primary_file):
        return [primary_file]
    files = glob.glob(os.path.join(data_dir, "*pretraining*.jsonl"))
    if files:
        return files
    return glob.glob(os.path.join(data_dir, "*.jsonl"))


def load_data(accelerator):
    """Load and prepare training data."""
    data_files = find_data_files(DATA_DIR)
    if not data_files:
        raise ValueError(f"No .jsonl files found in {DATA_DIR}")
    
    if accelerator.is_main_process:
        print(f"📁 Loading data from: {[os.path.basename(f) for f in data_files]}")
    
    # Load each file and normalize schema
    all_datasets = []
    for data_file in data_files:
        ds = load_dataset("json", data_files=data_file, split="train")
        if "text" in ds.column_names:
            ds = ds.select_columns(["text"])
            all_datasets.append(ds)
    
    if not all_datasets:
        raise ValueError("No valid datasets with 'text' column found!")
    
    dataset = concatenate_datasets(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
    
    if accelerator.is_main_process:
        print(f"   Maritime examples: {len(dataset):,}")
    
    # Add replay data
    try:
        replay_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        replay_amount = int(len(dataset) * REPLAY_RATIO)
        replay_ds = replay_ds.select(range(min(replay_amount, len(replay_ds))))
        dataset = concatenate_datasets([dataset, replay_ds])
        dataset = dataset.shuffle(seed=42)
        if accelerator.is_main_process:
            print(f"   + Replay data: {len(replay_ds):,} examples")
            print(f"   Total: {len(dataset):,} examples")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️ Could not load replay data: {e}")
    
    return dataset


# =============================================================================
# Training Phase
# =============================================================================

def train_phase(model, tokenizer, dataset, phase_config, phase_num, total_phases, accelerator):
    """Train a single phase of progressive context length training."""
    global _CURRENT_TOKEN_STATS
    _CURRENT_TOKEN_STATS = {}  # Reset stats for this phase
    
    seq_length = phase_config["max_seq_length"]
    phase_name = phase_config["name"]
    num_epochs = phase_config["num_epochs"]
    learning_rate = phase_config["learning_rate"]
    
    batch_size = BATCH_SIZE_PER_PHASE.get(seq_length, 1)
    grad_accum = GRAD_ACCUMULATION_PER_PHASE.get(seq_length, 16)
    
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"🎯 PHASE {phase_num}/{total_phases}: {phase_name}")
        print(f"   Context: {seq_length:,} tokens | Epochs: {num_epochs} | LR: {learning_rate:.2e}")
        print(f"   Batch: {batch_size} × {grad_accum} grad_accum × {accelerator.num_processes} GPUs")
        print(f"   Effective batch size: {batch_size * grad_accum * accelerator.num_processes}")
        print(f"{'='*80}")
    
    # Pack sequences for this phase - use cache if available
    from datasets import load_from_disk
    
    cache_dir = f"{OUTPUT_DIR}/.cache"
    cache_file = f"{cache_dir}/{phase_name}_packed"
    cache_exists = os.path.exists(cache_file)
    
    # Check cache status across all processes
    accelerator.wait_for_everyone()
    
    if cache_exists:
        # Cache exists - all processes load from it
        if accelerator.is_main_process:
            print(f"\n📦 Loading cached packed data for {seq_length:,} token context...")
            print(f"   Cache found at: {cache_file}")
        packed_dataset = load_from_disk(cache_file)
        if accelerator.is_main_process:
            print(f"   ✅ Loaded {len(packed_dataset):,} cached sequences")
    else:
        # No cache - main process creates it
        if accelerator.is_main_process:
            print(f"\n📦 Packing sequences for {seq_length:,} token context (first run)...")
            os.makedirs(cache_dir, exist_ok=True)
            
            num_proc = min(8, os.cpu_count() or 1)
            packed_dataset = dataset.map(
                lambda x: pack_sequences(x, tokenizer, seq_length),
                batched=True,
                batch_size=1000,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
                desc=f"Packing for {seq_length//1024}K"
            )
            # Save to disk for other processes
            packed_dataset.save_to_disk(cache_file)
            print(f"   ✅ Packed and cached to {cache_file}")
        
        # Wait for main process to finish packing
        accelerator.wait_for_everyone()
        
        # Non-main processes load from cache
        if not accelerator.is_main_process:
            packed_dataset = load_from_disk(cache_file)
    
    # Display token statistics
    if accelerator.is_main_process and _CURRENT_TOKEN_STATS:
        stats = _CURRENT_TOKEN_STATS
        total_tokens = stats["total_real_tokens"] + stats["total_padding_tokens"]
        utilization = (stats["total_real_tokens"] / total_tokens * 100) if total_tokens > 0 else 0
        
        print(f"\n📊 Token Utilization Statistics:")
        print(f"   Raw tokens (before packing): {stats['total_raw_tokens']:,}")
        print(f"   Real tokens (in training): {stats['total_real_tokens']:,}")
        print(f"   Padding tokens (ignored): {stats['total_padding_tokens']:,}")
        print(f"   Utilization rate: {utilization:.2f}%")
        print(f"   Packed sequences: {stats['num_sequences']:,}")
        
        # Log to file
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log_file = f"{OUTPUT_DIR}/token_statistics.jsonl"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase_name,
            "seq_length": seq_length,
            "num_epochs": num_epochs,
            **stats,
            "utilization_pct": utilization
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    if accelerator.is_main_process:
        print(f"\n   Packed dataset: {len(packed_dataset):,} sequences")
    
    output_dir = f"{OUTPUT_DIR}/{phase_name}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO if phase_num == 1 else WARMUP_RATIO * 0.5,
        logging_dir=f"{output_dir}/logs",
        report_to=["tensorboard"],
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,  # Disabled - incompatible with DeepSpeed ZeRO-3
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        tf32=True,  # Enable TF32 for faster matmuls on H100
    )
    
    # Ensure model is in training mode
    model.train()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EnhancedProgressCallback(phase_num, total_phases, phase_name)]
    )
    
    if accelerator.is_main_process:
        print(f"\n🏃 Starting Phase {phase_num} training...")
        print(f"📊 Monitor: tensorboard --logdir {output_dir}/logs\n")
    
    trainer.train()
    
    # Save checkpoint
    if accelerator.is_main_process:
        print(f"\n💾 Saving {phase_name} checkpoint...")
    trainer.save_model(output_dir)
    
    return model


# =============================================================================
# Main
# =============================================================================

def main():
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"""
{'#'*80}
#  Multi-GPU Continual Pre-Training
#  Model: Qwen3-30B-A3B (MoE)
#  Target: 128K context | GPUs: {accelerator.num_processes}
#  Strategy: Progressive Context Length Training
{'#'*80}
""")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    # Load data
    dataset = load_data(accelerator)
    
    if accelerator.is_main_process:
        print(f"\n🔧 Loading model: {MODEL_ID}")
    
    # Load model with SDPA
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
        device_map=None,  # Let DeepSpeed handle device placement
        low_cpu_mem_usage=True,  # Better initialization for DeepSpeed ZeRO-3
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if accelerator.is_main_process:
        print("✅ Model loaded")
        print(f"\n🔧 Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    # Progressive training
    training_start = time.time()
    
    context_progression = ' → '.join([f'{p["max_seq_length"]//1024}K' for p in CONTEXT_SCHEDULE])
    if accelerator.is_main_process:
        print(f"\n🚀 Starting Progressive Context Training")
        print(f"   Phases: {len(CONTEXT_SCHEDULE)}")
        print(f"   Context progression: {context_progression}")
    
    for phase_num, phase_config in enumerate(CONTEXT_SCHEDULE, 1):
        model = train_phase(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            phase_config=phase_config,
            phase_num=phase_num,
            total_phases=len(CONTEXT_SCHEDULE),
            accelerator=accelerator
        )
    
    # Save final model
    if accelerator.is_main_process:
        final_output = f"{OUTPUT_DIR}/final_128k_model"
        print(f"\n💾 Saving final model to {final_output}...")
        model.save_pretrained(final_output)
        tokenizer.save_pretrained(final_output)
        
        total_time = time.time() - training_start
        total_epochs = sum(p["num_epochs"] for p in CONTEXT_SCHEDULE)
        
        print(f"""
{'='*80}
✅ Training Complete!
   Total time: {str(timedelta(seconds=int(total_time)))}
   Total epochs: {total_epochs}
   Final context capability: 128K tokens
   Output: {final_output}
   Token stats: {OUTPUT_DIR}/token_statistics.jsonl
{'='*80}
""")


if __name__ == "__main__":
    main()
