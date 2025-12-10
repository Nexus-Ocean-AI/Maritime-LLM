"""
Unsloth-Optimized Continual Pre-Training for Qwen3-30B
Target: 128K context window on 8×H100 GPUs

This script implements progressive context length training with Unsloth optimizations
for memory efficiency and speed. For multi-GPU, use Accelerate/DeepSpeed integration.

Usage (Single GPU):
    python continual_pretraining_qwen3_30b_unsloth.py

Usage (Multi-GPU with Accelerate):
    accelerate launch --num_processes=8 --multi_gpu continual_pretraining_qwen3_30b_unsloth.py

Usage (Multi-GPU with DeepSpeed ZeRO-3):
    accelerate launch --config_file accelerate_config.yaml continual_pretraining_qwen3_30b_unsloth.py
"""

import os
import glob
import json
import torch
import time
from datetime import datetime, timedelta

# Silence experimental warnings
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Import unsloth first (required)
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported

from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


class EnhancedProgressCallback(TrainerCallback):
    """Enhanced progress tracking with ETA and memory monitoring."""
    
    def __init__(self, phase_num, total_phases, phase_name):
        self.phase_num = phase_num
        self.total_phases = total_phases
        self.phase_name = phase_name
        self.phase_start_time = None
        self.step_times = []
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.phase_start_time = time.time()
        print(f"\n🚀 Starting {self.phase_name} (Phase {self.phase_num}/{self.total_phases})")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f}GB)")
        print()
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_times.append(time.time())
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            elapsed = time.time() - self.phase_start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            progress = state.global_step / state.max_steps if state.max_steps else 0
            progress_bar = "█" * int(progress * 25) + "░" * (25 - int(progress * 25))
            
            # Estimate ETA
            if state.global_step > 10:
                steps_remaining = state.max_steps - state.global_step
                avg_step_time = elapsed / state.global_step
                eta_seconds = steps_remaining * avg_step_time
                eta_str = str(timedelta(seconds=int(eta_seconds)))
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
        print(f"\n✅ Epoch {int(state.epoch)} completed!")
        
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.phase_start_time
        print(f"\n\n✅ {self.phase_name} Complete!")
        print(f"⏱️  Phase time: {str(timedelta(seconds=int(total_time)))}")
        if state.log_history:
            final_loss = next((l.get('loss') for l in reversed(state.log_history) if 'loss' in l), None)
            if final_loss:
                print(f"📊 Final loss: {final_loss:.4f}")
        print(f"{'='*80}\n")


# =============================================================================
# Configuration - Qwen3-30B with 128K Context
# =============================================================================

# Model Selection - Qwen3-30B (MoE: 30B total, 3.3B active)
# Use Unsloth's pre-optimized version when available
MODEL_ID = "unsloth/Qwen3-30B-A3B"  # MoE version: memory efficient
# Alternative: "Qwen/Qwen3-30B-A3B" for official version

# Fallback if MoE version not available
ALTERNATIVE_MODELS = [
    "unsloth/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B",
    "unsloth/Qwen3-30B-A3B-128K",  # 128K native context
]

DATA_DIR = "data"
OUTPUT_DIR = "qwen3-30b-maritime-128K-cpt"

# =============================================================================
# Progressive Context Schedule for 128K
# 
# Strategy: Start small, gradually extend context
# Each phase trains on FULL dataset at increasing context lengths
# This teaches the model to handle both short AND long range dependencies
# =============================================================================

PROGRESSIVE_TRAINING = True
CONTEXT_SCHEDULE = [
    # Phase 1: Fast domain adaptation at short context
    {"name": "Phase_1_Domain_4K", "max_seq_length": 4096, "num_epochs": 2, "learning_rate": 2e-4},
    
    # Phase 2: Medium context learning
    {"name": "Phase_2_Medium_16K", "max_seq_length": 16384, "num_epochs": 2, "learning_rate": 1e-4},
    
    # Phase 3: Long context introduction
    {"name": "Phase_3_Long_32K", "max_seq_length": 32768, "num_epochs": 2, "learning_rate": 5e-5},
    
    # Phase 4: Extended context mastery
    {"name": "Phase_4_Extended_64K", "max_seq_length": 65536, "num_epochs": 2, "learning_rate": 2e-5},
    
    # Phase 5: Ultra-long context refinement (128K)
    {"name": "Phase_5_Ultra_128K", "max_seq_length": 131072, "num_epochs": 3, "learning_rate": 1e-5},
]

# If you want to skip progressive training and train directly at 128K
SINGLE_PHASE_CONFIG = {
    "name": "Direct_128K",
    "max_seq_length": 131072,
    "num_epochs": 5,
    "learning_rate": 5e-5
}

# =============================================================================
# LoRA Configuration - Optimized for 30B Model
# =============================================================================

LORA_R = 128              # Higher rank for larger model
LORA_ALPHA = 256          # Alpha = 2 * R is a good rule
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # MLP
]

# For MoE models, also consider targeting router/expert layers
LORA_TARGET_MODULES_MOE = LORA_TARGET_MODULES + [
    "gate"  # Router weights if applicable
]

# =============================================================================
# Training Hyperparameters - Optimized for 8×H100 (80GB each)
# =============================================================================

# Batch sizes per context length (per GPU)
# With 8×H100 (640GB total), plus ZeRO-3 sharding
BATCH_SIZE_PER_PHASE = {
    4096: 4,      # ~20GB per GPU
    16384: 2,     # ~40GB per GPU  
    32768: 1,     # ~60GB per GPU
    65536: 1,     # ~70GB per GPU (may need offloading)
    131072: 1,    # ~75GB per GPU (flash attention required)
}

# Gradient accumulation to achieve larger effective batch sizes
GRAD_ACCUMULATION_PER_PHASE = {
    4096: 8,      # Effective batch: 4 * 8 * 8 = 256
    16384: 16,    # Effective batch: 2 * 16 * 8 = 256
    32768: 32,    # Effective batch: 1 * 32 * 8 = 256
    65536: 32,    # Effective batch: 1 * 32 * 8 = 256
    131072: 64,   # Effective batch: 1 * 64 * 8 = 512
}

WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2  # Keep only recent checkpoints to save disk

# Replay buffer for catastrophic forgetting prevention
REPLAY_DATASET_NAME = "wikitext"
REPLAY_DATASET_CONFIG = "wikitext-103-raw-v1"
REPLAY_RATIO = 0.10  # 10% replay data

# =============================================================================
# Multi-GPU & Memory Settings
# =============================================================================

USE_4BIT = False          # Set True if OOM (uses 17.5GB instead of 60GB)
USE_GRADIENT_CHECKPOINTING = True  # Essential for 128K context

# DeepSpeed settings (configured via accelerate_config.yaml)
DEEPSPEED_STAGE = 3       # ZeRO-3 for 30B model across 8 GPUs


def find_all_data_files(data_dir):
    """Find all maritime training data files."""
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    # Filter for maritime data files
    maritime_files = [f for f in files if "maritime" in f.lower() or "pretraining" in f.lower()]
    if not maritime_files:
        # If no maritime-specific files, use all JSONL files
        maritime_files = files
    print(f"📁 Found {len(maritime_files)} data files:")
    for f in maritime_files:
        print(f"   - {os.path.basename(f)}")
    return maritime_files


def load_and_mix_data():
    """Load maritime domain data with optional replay buffer."""
    data_files = find_all_data_files(DATA_DIR)
    if not data_files:
        raise ValueError(f"No .jsonl files found in {DATA_DIR}")

    print("\n📚 Loading maritime text data...")
    dataset = load_dataset("json", data_files=data_files, split="train")
    print(f"   Maritime examples: {len(dataset):,}")

    if REPLAY_DATASET_NAME:
        print(f"\n🔄 Loading replay data: {REPLAY_DATASET_NAME}...")
        try:
            replay_dataset = load_dataset(REPLAY_DATASET_NAME, REPLAY_DATASET_CONFIG, split="train")
            total_maritime = len(dataset)
            replay_amount = int(total_maritime * REPLAY_RATIO)
            replay_dataset = replay_dataset.select(range(min(replay_amount, len(replay_dataset))))
            
            print(f"   Replay examples: {len(replay_dataset):,} ({REPLAY_RATIO*100:.0f}% of domain data)")
            dataset = concatenate_datasets([dataset, replay_dataset])
            dataset = dataset.shuffle(seed=42)
            print(f"   Combined total: {len(dataset):,} examples")
        except Exception as e:
            print(f"⚠️ Could not load replay dataset: {e}")
            print("   Continuing without replay (monitor for forgetting)")

    return dataset


def formatting_func(examples):
    """Format examples for Unsloth's SFTTrainer - just return raw text."""
    return examples["text"]


def log_phase_config(phase_config, batch_size, grad_accum, num_gpus=1):
    """Log detailed training configuration for a phase."""
    seq_length = phase_config["max_seq_length"]
    num_epochs = phase_config["num_epochs"]
    lr = phase_config.get("learning_rate", 1e-4)
    effective_batch = batch_size * grad_accum * num_gpus
    
    print(f"\n📋 Phase Configuration:")
    print(f"   Sequence Length: {seq_length:,} tokens")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {lr:.2e}")
    print(f"   Per-GPU Batch Size: {batch_size}")
    print(f"   Gradient Accumulation: {grad_accum}")
    print(f"   Number of GPUs: {num_gpus}")
    print(f"   Effective Batch Size: {effective_batch}")
    print(f"   Tokens per Batch: {effective_batch * seq_length:,}")
    print()


def train_phase(model, tokenizer, dataset, phase_config, phase_num, total_phases):
    """Train a single phase with Unsloth optimization."""
    seq_length = phase_config["max_seq_length"]
    phase_name = phase_config["name"]
    num_epochs = phase_config["num_epochs"]
    learning_rate = phase_config.get("learning_rate", 1e-4)
    
    print(f"\n{'='*80}")
    print(f"🎯 PHASE {phase_num}/{total_phases}: {phase_name}")
    print(f"{'='*80}")
    
    # Get hyperparameters for this context length
    batch_size = BATCH_SIZE_PER_PHASE.get(seq_length, 1)
    grad_accum = GRAD_ACCUMULATION_PER_PHASE.get(seq_length, 64)
    
    # Detect number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    log_phase_config(phase_config, batch_size, grad_accum, num_gpus)
    
    # Training Arguments
    output_dir = f"{OUTPUT_DIR}/{phase_name}"
    
    # Determine warmup - shorter for later phases
    warmup = WARMUP_RATIO if phase_num == 1 else WARMUP_RATIO * 0.5
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        
        # Precision
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        
        # Optimizer - 8bit Adam for memory efficiency
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=warmup,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        report_to=["tensorboard"],
        
        # Performance
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        
        # For multi-GPU
        ddp_find_unused_parameters=False,
    )
    
    # Create SFTTrainer with Unsloth optimizations
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=seq_length,  # Unsloth handles this
        callbacks=[EnhancedProgressCallback(phase_num, total_phases, phase_name)]
    )
    
    print(f"🏃 Starting Phase {phase_num} training...")
    print(f"📊 Monitor: tensorboard --logdir {output_dir}/logs\n")
    
    # Train
    trainer.train()
    
    # Save checkpoint
    print(f"\n💾 Saving Phase {phase_num} checkpoint...")
    trainer.save_model(output_dir)
    
    # Log token statistics
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase_name,
        "phase_num": phase_num,
        "seq_length": seq_length,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "effective_batch_size": batch_size * grad_accum * num_gpus,
    }
    
    log_file = f"{OUTPUT_DIR}/training_log.jsonl"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return model


def main():
    print(f"""
{'#'*80}
#  Unsloth-Optimized Continual Pre-Training
#  Model: Qwen3-30B-A3B (MoE)
#  Target Context: 128K tokens
#  Hardware: 8× H100 GPUs
#  Expected Speedup: 2-5x vs standard training
{'#'*80}
""")
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"🖥️  GPUs Available: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    else:
        print("⚠️ No GPU detected! Training will be extremely slow.")
    
    # Load Data
    dataset = load_and_mix_data()
    print(f"\n📊 Total training examples: {len(dataset):,}")
    
    # Estimate tokens
    avg_tokens_estimate = 500  # Rough estimate
    total_tokens_estimate = len(dataset) * avg_tokens_estimate
    print(f"📊 Estimated total tokens: ~{total_tokens_estimate:,}")
    
    # Load model with Unsloth
    print(f"\n🔧 Loading model with Unsloth optimizations...")
    print(f"   Model: {MODEL_ID}")
    print(f"   4-bit quantization: {USE_4BIT}")
    print(f"   Max sequence length: {CONTEXT_SCHEDULE[-1]['max_seq_length']:,}")
    
    # Determine max seq length across all phases
    max_seq_across_phases = max(p["max_seq_length"] for p in CONTEXT_SCHEDULE)
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=max_seq_across_phases,
            dtype=None,  # Auto-detect (bfloat16 on H100)
            load_in_4bit=USE_4BIT,
            # For 128K context, Unsloth uses YaRN internally
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load {MODEL_ID}: {e}")
        print("   Trying alternative models...")
        
        model, tokenizer = None, None
        for alt_model in ALTERNATIVE_MODELS:
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=alt_model,
                    max_seq_length=max_seq_across_phases,
                    dtype=None,
                    load_in_4bit=USE_4BIT,
                )
                print(f"✅ Successfully loaded {alt_model}")
                break
            except Exception as e2:
                print(f"   ❌ {alt_model}: {e2}")
        
        if model is None:
            raise RuntimeError("Could not load any Qwen3 model variant!")
    
    # Apply LoRA with Unsloth optimizations
    print("\n🔧 Applying Unsloth-optimized LoRA...")
    print(f"   Rank (r): {LORA_R}")
    print(f"   Alpha: {LORA_ALPHA}")
    print(f"   Target modules: {LORA_TARGET_MODULES}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=42,
    )
    print("✅ LoRA applied with Unsloth optimizations")
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Progressive Training
    print(f"\n{'='*80}")
    print("🚀 Starting Progressive Context Training")
    print(f"   Phases: {len(CONTEXT_SCHEDULE)}")
    context_progression = ' → '.join([f'{p["max_seq_length"]//1024}K' for p in CONTEXT_SCHEDULE])
    print(f"   Context progression: {context_progression}")
    print(f"{'='*80}")
    
    training_start = time.time()
    
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
        model = train_phase(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            phase_config=SINGLE_PHASE_CONFIG,
            phase_num=1,
            total_phases=1
        )
    
    training_time = time.time() - training_start
    training_time_str = str(timedelta(seconds=int(training_time)))
    
    # Final Save
    final_output = f"{OUTPUT_DIR}/final_128k_model"
    print(f"\n💾 Saving final LoRA adapter to {final_output}...")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    
    # Save merged model for inference
    print("\n🔀 Creating merged model (Unsloth fast merge)...")
    merged_output = f"{final_output}_merged"
    try:
        model.save_pretrained_merged(
            merged_output,
            tokenizer,
            save_method="merged_16bit",  # Full precision merged weights
        )
        print(f"✅ Merged model saved to: {merged_output}")
    except Exception as e:
        print(f"⚠️ Could not save merged model: {e}")
        print("   (LoRA adapter is still saved and can be merged later)")
    
    # Summary
    total_epochs = sum(phase["num_epochs"] for phase in CONTEXT_SCHEDULE)
    print(f"""
{'='*80}
✅ Training Complete!
{'='*80}
📊 Summary:
   Total Training Time: {training_time_str}
   Total Epochs: {total_epochs}
   Final Context Capability: 128K tokens
   
📁 Output Locations:
   LoRA Adapter: {final_output}
   Merged Model: {merged_output}
   Training Logs: {OUTPUT_DIR}/training_log.jsonl
   TensorBoard: {OUTPUT_DIR}/*/logs

🚀 Next Steps:
   1. Test the model with long context inputs
   2. Run inference with: model.generate() or vLLM
   3. Consider GGUF quantization for deployment
{'='*80}
""")


if __name__ == "__main__":
    main()
