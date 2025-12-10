"""
Multi-GPU Continual Pre-Training for Qwen3-30B
Target: 128K context window on 8×H100 GPUs with DeepSpeed ZeRO-3

This script uses standard HuggingFace + PEFT + DeepSpeed for multi-GPU training.
For single-GPU with Unsloth optimizations, use continual_pretraining_qwen3_30b_unsloth.py

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

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from accelerate import Accelerator

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "Qwen/Qwen3-30B-A3B"  # Official Qwen3 30B MoE model
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
    4096: 4,
    16384: 2,
    32768: 1,
    65536: 1,
    131072: 1,
}

GRAD_ACCUMULATION_PER_PHASE = {
    4096: 4,
    16384: 8,
    32768: 16,
    65536: 32,
    131072: 32,
}

# Replay buffer
REPLAY_RATIO = 0.10
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2


class ProgressCallback(TrainerCallback):
    """Enhanced progress tracking."""
    
    def __init__(self, phase_name):
        self.phase_name = phase_name
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if state.is_local_process_zero:
            print(f"\n🚀 Starting {self.phase_name}")
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and state.is_local_process_zero:
            elapsed = time.time() - self.start_time
            progress = state.global_step / state.max_steps if state.max_steps else 0
            
            if state.global_step > 10:
                eta = (elapsed / state.global_step) * (state.max_steps - state.global_step)
                eta_str = str(timedelta(seconds=int(eta)))
            else:
                eta_str = "calculating..."
            
            print(f"[{self.phase_name}] Step {state.global_step}/{state.max_steps} | "
                  f"Loss: {logs['loss']:.4f} | ETA: {eta_str} | {progress*100:.1f}%")
    
    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            total_time = time.time() - self.start_time
            print(f"✅ {self.phase_name} complete in {str(timedelta(seconds=int(total_time)))}")


def find_data_files(data_dir):
    """Find the consolidated maritime training data file."""
    primary_file = os.path.join(data_dir, "maritime_pretraining_data.jsonl")
    if os.path.exists(primary_file):
        return [primary_file]
    files = glob.glob(os.path.join(data_dir, "*pretraining*.jsonl"))
    if files:
        return files
    return glob.glob(os.path.join(data_dir, "*.jsonl"))


def load_data():
    """Load and prepare training data."""
    accelerator = Accelerator()
    
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


def formatting_func(examples):
    """Format examples for training."""
    return examples["text"]


def train_phase(model, tokenizer, dataset, phase_config, phase_num, total_phases):
    """Train a single phase."""
    accelerator = Accelerator()
    
    seq_length = phase_config["max_seq_length"]
    phase_name = phase_config["name"]
    num_epochs = phase_config["num_epochs"]
    learning_rate = phase_config["learning_rate"]
    
    batch_size = BATCH_SIZE_PER_PHASE.get(seq_length, 1)
    grad_accum = GRAD_ACCUMULATION_PER_PHASE.get(seq_length, 16)
    
    if accelerator.is_main_process:
        print(f"\n{'='*70}")
        print(f"🎯 PHASE {phase_num}/{total_phases}: {phase_name}")
        print(f"   Context: {seq_length:,} tokens | Epochs: {num_epochs} | LR: {learning_rate:.2e}")
        print(f"   Batch: {batch_size} × {grad_accum} grad_accum × {accelerator.num_processes} GPUs")
        print(f"{'='*70}")
    
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
        warmup_ratio=0.03 if phase_num == 1 else 0.01,
        logging_dir=f"{output_dir}/logs",
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # DeepSpeed handles distribution
        deepspeed=None,  # Will use accelerate config
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=seq_length,
        callbacks=[ProgressCallback(phase_name)],
    )
    
    trainer.train()
    
    # Save checkpoint
    if accelerator.is_main_process:
        print(f"💾 Saving {phase_name} checkpoint...")
    trainer.save_model(output_dir)
    
    return model


def main():
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"""
{'#'*70}
#  Multi-GPU Continual Pre-Training
#  Model: Qwen3-30B-A3B (MoE)
#  Target: 128K context | GPUs: {accelerator.num_processes}
{'#'*70}
""")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    # Load data
    dataset = load_data()
    
    if accelerator.is_main_process:
        print(f"\n🔧 Loading model: {MODEL_ID}")
    
    # Load model with Flash Attention 2
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map=None,  # Let DeepSpeed handle device placement
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if accelerator.is_main_process:
        print("✅ Model loaded")
        print(f"\n🔧 Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    
    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Progressive training
    training_start = time.time()
    
    for phase_num, phase_config in enumerate(CONTEXT_SCHEDULE, 1):
        model = train_phase(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            phase_config=phase_config,
            phase_num=phase_num,
            total_phases=len(CONTEXT_SCHEDULE)
        )
    
    # Save final model
    if accelerator.is_main_process:
        final_output = f"{OUTPUT_DIR}/final_128k_model"
        print(f"\n💾 Saving final model to {final_output}...")
        model.save_pretrained(final_output)
        tokenizer.save_pretrained(final_output)
        
        total_time = time.time() - training_start
        print(f"""
{'='*70}
✅ Training Complete!
   Total time: {str(timedelta(seconds=int(total_time)))}
   Output: {final_output}
{'='*70}
""")


if __name__ == "__main__":
    main()
