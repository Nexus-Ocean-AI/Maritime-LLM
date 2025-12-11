"""
Progressive Long-Context Continual Pre-Training Script
Supports checkpoint resumption for multi-phase training.

To resume training from a checkpoint:
1. Set START_FROM_PHASE to the phase number you want to start from (e.g., 2 for Phase_1b_Medium)
2. Ensure the checkpoint from the previous phase exists in qwen-maritime-longcontext-cpt/
3. Run the script - it will automatically load the checkpoint and continue training

Example:
    START_FROM_PHASE = 1  # Start from scratch
    START_FROM_PHASE = 2  # Resume from Phase_1b_Medium (loads Phase_1a_Short checkpoint)
    START_FROM_PHASE = 3  # Resume from Phase_1c_Long (loads Phase_1b_Medium checkpoint)
"""

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
        
        # Get final loss, handle case where it might be 'N/A'
        final_loss = state.log_history[-1].get('loss', 'N/A')
        if isinstance(final_loss, (int, float)):
            print(f"📊 Final loss: {final_loss:.4f}")
        else:
            print(f"📊 Final loss: {final_loss}")
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
    {"name": "Phase_1a_Short", "max_seq_length": 2048, "num_epochs": 4},    # 1 epoch at 2K - Fast domain learning
    {"name": "Phase_1b_Medium", "max_seq_length": 16384, "num_epochs": 1},  # 2 epochs at 16K - Medium context
    {"name": "Phase_1c_Long", "max_seq_length": 32768, "num_epochs": 1},    # 4 epochs at 32K - Master long context
]

# Resume Training from Checkpoint
# Set to 1 to start from scratch, 2 to start from Phase_1b_Medium (after Phase_1a_Short), etc.
START_FROM_PHASE = 2  # Change to 2 to skip Phase_1a_Short and load its checkpoint

# Or set single length for direct training (NOT RECOMMENDED for 32K)
SINGLE_MAX_SEQ_LENGTH = 32768  # Only used if PROGRESSIVE_TRAINING = False

# TRAINING MODE
USE_LORA = True  # Must be True for 32K training on consumer hardware

# LoRA Config - Optimized for Long Context
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# Training Hyperparameters - OPTIMIZED for H100 Speed
LEARNING_RATE = 1e-4
BATCH_SIZE_PER_PHASE = {
    2048: 8,      # Original optimized value
    16384: 1,     # Reduced from 2 to prevent OOM when loading from checkpoint
    32768: 1,     # Keep at 1 (maximum memory usage)
}
GRAD_ACCUMULATION_PER_PHASE = {
    2048: 4,      # Original optimized value
    16384: 16,    # Increased from 8 to compensate for batch size reduction
    32768: 16,    # Increased from 16 to maintain effective batch size
}
WARMUP_RATIO = 0.03
SAVE_STEPS = 500  # Save less frequently to reduce I/O overhead
LOGGING_STEPS = 50  # Log less frequently to reduce overhead
SAVE_TOTAL_LIMIT = None  # Keep only last 3 checkpoints to save disk I/O

# Replay Buffer
REPLAY_DATASET_NAME = "wikitext"
REPLAY_DATASET_CONFIG = "wikitext-103-raw-v1"
REPLAY_RATIO = 0.15

# Memory Optimization - ENABLE Flash Attention!
USE_FLASH_ATTENTION_2 = True  # 2-4x speedup! Install: pip install flash-attn --no-build-isolation
USE_SDPA_FALLBACK = True  # Use PyTorch SDPA if Flash Attention not available
USE_8BIT = True  # Enabled to prevent OOM errors with 16K/32K contexts

# Performance Optimizations for H100
USE_TORCH_COMPILE = False  # Keep disabled - incompatible with gradient checkpointing + LoRA
DATALOADER_WORKERS = 8  # Parallel data loading
DATALOADER_PIN_MEMORY = True  # Pin memory for faster GPU transfer
DATALOADER_PREFETCH = 4  # Prefetch batches
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'  # Reduce memory fragmentation
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  # Parallel tokenization

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

# Global variable to store token statistics
_CURRENT_TOKEN_STATS = {}

def pack_sequences(examples, tokenizer, max_length):
    """
    Pack multiple sequences into one to avoid padding waste.
    Now pads the last chunk instead of discarding it!
    Statistics are stored in global _CURRENT_TOKEN_STATS.
    """
    global _CURRENT_TOKEN_STATS
    
    all_input_ids = []
    total_raw_tokens = 0
    
    for text in examples["text"]:
        tokens = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        all_input_ids.extend(tokens["input_ids"])
        all_input_ids.append(tokenizer.eos_token_id)
        total_raw_tokens += len(tokens["input_ids"]) + 1  # +1 for EOS
    
    packed = {"input_ids": [], "attention_mask": [], "labels": []}
    total_real_tokens = 0
    total_padding_tokens = 0
    
    for i in range(0, len(all_input_ids), max_length):
        chunk = all_input_ids[i:i + max_length]
        
        # Pad the last chunk if it's shorter than max_length
        if len(chunk) < max_length:
            padding_length = max_length - len(chunk)
            chunk = chunk + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * (max_length - padding_length) + [0] * padding_length
            # Set labels to -100 for padded tokens (ignored in loss)
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
    
    # Update global stats (accumulate across batches)
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
    
    # Clear GPU cache before starting each phase
    if torch.cuda.is_available():
        print("\n🧹 Clearing GPU cache before starting new phase...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"✅ GPU memory cleared\n")
    
    print(f"\n{'='*80}")
    print(f"PHASE {phase_num}/{total_phases}: {phase_name}")
    print(f"Sequence Length: {seq_length:,} tokens")
    print(f"Training Epochs: {num_epochs}")
    print(f"{'='*80}\n")
    
    # Reset global token stats
    global _CURRENT_TOKEN_STATS
    _CURRENT_TOKEN_STATS = {}
    
    # Pack sequences for this phase's length (with parallel processing)
    print("Packing sequences (using parallel processing)...")
    num_proc = min(8, os.cpu_count() or 1)  # Use up to 8 CPU cores
    packed_dataset = dataset.map(
        lambda x: pack_sequences(x, tokenizer, seq_length),
        batched=True,
        batch_size=2000,  # Larger batch for efficiency
        remove_columns=dataset.column_names,
        num_proc=num_proc,  # Parallel processing
        desc=f"Packing for {seq_length} tokens"
    )
    
    # Display token statistics from global variable
    if _CURRENT_TOKEN_STATS:
        stats = _CURRENT_TOKEN_STATS
        utilization_pct = (stats["total_real_tokens"] / (stats["total_real_tokens"] + stats["total_padding_tokens"]) * 100) if (stats["total_real_tokens"] + stats["total_padding_tokens"]) > 0 else 0
        
        print(f"\n📊 Token Utilization Statistics:")
        print(f"  Raw Tokens (before packing): {stats['total_raw_tokens']:,}")
        print(f"  Real Tokens (in training): {stats['total_real_tokens']:,}")
        print(f"  Padding Tokens (ignored): {stats['total_padding_tokens']:,}")
        print(f"  Utilization Rate: {utilization_pct:.2f}%")
        print(f"  Packed Sequences: {stats['num_sequences']} sequences\n")
        
        # Log to file
        import json
        from datetime import datetime
        log_file = f"{OUTPUT_DIR}/token_statistics.jsonl"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase_name,
            "phase_num": phase_num,
            "seq_length": seq_length,
            "num_epochs": num_epochs,
            "total_raw_tokens": stats["total_raw_tokens"],
            "total_real_tokens": stats["total_real_tokens"],
            "total_padding_tokens": stats["total_padding_tokens"],
            "num_sequences": stats["num_sequences"],
            "utilization_pct": utilization_pct
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"✅ Token stats logged to: {log_file}\n")
    
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
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        dataloader_prefetch_factor=DATALOADER_PREFETCH,
        gradient_checkpointing=True,  # Essential for memory - always enabled
        remove_unused_columns=False,  # Required for torch.compile compatibility
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        # Additional H100 optimizations
        tf32=True,  # Enable TF32 for faster matmuls on Ampere/Hopper
    )
    
    # Ensure model is in training mode with gradients enabled
    model.train()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
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
    
    # Save phase checkpoint - ensure adapter files are in the main output_dir
    print(f"Saving Phase {phase_num} checkpoint to {output_dir}...")
    
    # For LoRA models, explicitly save the adapter to the output directory
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
        print(f"✅ LoRA adapter saved to {output_dir}")
    else:
        trainer.save_model(output_dir)
        print(f"✅ Model saved to {output_dir}")
    
    # Also save tokenizer for convenience
    tokenizer.save_pretrained(output_dir)
    
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
    
    # 3. Load Model or Resume from Checkpoint
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() or torch.backends.mps.is_available() else torch.float16
    
    # Check if we're resuming from a checkpoint
    if START_FROM_PHASE > 1:
        # Load from the previous phase's checkpoint
        previous_phase_idx = START_FROM_PHASE - 2  # 0-indexed
        previous_phase_name = CONTEXT_SCHEDULE[previous_phase_idx]["name"]
        checkpoint_path = f"{OUTPUT_DIR}/{previous_phase_name}"
        
        print(f"\n🔄 Resuming from checkpoint: {checkpoint_path}")
        print(f"📂 Skipping phases 1-{START_FROM_PHASE-1}, starting from phase {START_FROM_PHASE}\n")
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}\n"
                           f"Please ensure Phase {START_FROM_PHASE-1} ({previous_phase_name}) completed successfully.")
        
        # Check if adapter_config.json exists in the main directory
        # If not, find the latest checkpoint subdirectory
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            print(f"⚠️  adapter_config.json not found in {checkpoint_path}")
            print(f"🔍 Looking for latest checkpoint subdirectory...")
            
            # Find all checkpoint-* subdirectories
            checkpoint_dirs = glob.glob(os.path.join(checkpoint_path, "checkpoint-*"))
            if not checkpoint_dirs:
                raise ValueError(f"No checkpoint subdirectories found in {checkpoint_path}\n"
                               f"Expected format: checkpoint-XXXX")
            
            # Sort by checkpoint number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
            checkpoint_path = checkpoint_dirs[-1]
            print(f"✅ Found latest checkpoint: {checkpoint_path}")
        
        print(f"Loading model from checkpoint in {dtype}...")
        
        # Load the base model first
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "use_cache": False,
        }
        
        # Try Flash Attention 2 first, fallback to SDPA (PyTorch 2.0+)
        if USE_FLASH_ATTENTION_2:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("✅ Using Flash Attention 2 for maximum speed")
            except ImportError:
                if USE_SDPA_FALLBACK:
                    model_kwargs["attn_implementation"] = "sdpa"
                    print("⚠️ Flash Attention not installed, using SDPA (PyTorch native) - still faster than default!")
                    print("   For best performance, install: pip install flash-attn --no-build-isolation")
                else:
                    print("⚠️ Flash Attention not installed, using default attention")
        elif USE_SDPA_FALLBACK:
            model_kwargs["attn_implementation"] = "sdpa"
            print("✅ Using SDPA (PyTorch native) for faster attention")
        
        if USE_8BIT:
            model_kwargs["load_in_8bit"] = True
            print("Loading in 8-bit mode for reduced memory")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        
        # Apply LoRA configuration
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
            
            # Load the checkpoint weights
            from peft import PeftModel
            print(f"Loading LoRA adapter from {checkpoint_path}...")
            model = PeftModel.from_pretrained(model.base_model.model, checkpoint_path)
            model.print_trainable_parameters()
        else:
            # For full fine-tuning, load the entire model from checkpoint
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
            model.gradient_checkpointing_enable()
        
        print(f"✅ Successfully loaded checkpoint from {previous_phase_name}")
        
        # Clear GPU cache after loading checkpoint to free fragmented memory
        if torch.cuda.is_available():
            print("🧹 Clearing GPU cache after checkpoint load...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"✅ GPU memory cleared\n")
        
    else:
        # Starting from scratch
        print(f"\nLoading fresh model in {dtype}...")
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "use_cache": False,
        }
        
        # Try Flash Attention 2 first, fallback to SDPA (PyTorch 2.0+)
        if USE_FLASH_ATTENTION_2:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("✅ Using Flash Attention 2 for maximum speed")
            except ImportError:
                if USE_SDPA_FALLBACK:
                    model_kwargs["attn_implementation"] = "sdpa"
                    print("⚠️ Flash Attention not installed, using SDPA (PyTorch native) - still faster than default!")
                    print("   For best performance, install: pip install flash-attn --no-build-isolation")
                else:
                    print("⚠️ Flash Attention not installed, using default attention")
        elif USE_SDPA_FALLBACK:
            model_kwargs["attn_implementation"] = "sdpa"
            print("✅ Using SDPA (PyTorch native) for faster attention")
        
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
    
    # Compile model for H100 speedup (PyTorch 2.0+)
    if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        print("\n🚀 Compiling model with torch.compile()...")
        print("This will take a few minutes but speeds up training by 20-30%!")
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ Model compiled!\n")
    
    # 5. Progressive Training
    if PROGRESSIVE_TRAINING:
        for phase_num, phase_config in enumerate(CONTEXT_SCHEDULE, 1):
            # Skip phases that were already completed
            if phase_num < START_FROM_PHASE:
                print(f"⏭️  Skipping {phase_config['name']} (already completed)")
                continue
                
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
