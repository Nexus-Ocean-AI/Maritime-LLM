"""
Merge LoRA adapter weights into base Qwen model.

This creates a standalone model that doesn't require PEFT/LoRA at inference time.
The merged model will be a full 7B parameter model with maritime knowledge baked in.

Usage:
    python merge_lora_adapter.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "qwen-maritime-longcontext-cpt/final_32k_model"  # Path to your trained LoRA adapter
OUTPUT_PATH = "qwen-maritime-7b-merged"  # Where to save the merged model

# Memory Settings
USE_8BIT_FOR_MERGE = False  # Set True if you have limited VRAM during merge
DEVICE_MAP = "auto"  # Automatically distribute across GPUs/CPU

def merge_and_save():
    """Load base model, merge LoRA adapter, and save the merged model."""
    
    print(f"\n{'='*80}")
    print("LoRA Adapter Merging Tool")
    print(f"{'='*80}\n")
    
    # 1. Load Base Model
    print(f"Loading base model: {BASE_MODEL_ID}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model_kwargs = {
        "device_map": DEVICE_MAP,
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    
    if USE_8BIT_FOR_MERGE:
        model_kwargs["load_in_8bit"] = True
        print("Using 8-bit mode for memory efficiency")
    
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
    print(f"✅ Base model loaded ({dtype})")
    
    # 2. Load LoRA Adapter
    print(f"\nLoading LoRA adapter from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✅ LoRA adapter loaded")
    
    # 3. Merge Adapter into Base Model
    print("\nMerging LoRA weights into base model...")
    print("⏳ This may take a few minutes...")
    
    # merge_and_unload() combines LoRA weights with base weights
    merged_model = model.merge_and_unload()
    print("✅ Merge complete!")
    
    # 4. Load Tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")
    
    # 5. Save Merged Model
    print(f"\nSaving merged model to: {OUTPUT_PATH}...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Save in safetensors format (recommended, smaller and safer)
    merged_model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,  # Use safetensors
        max_shard_size="5GB"  # Shard large models
    )
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("✅ Merged model saved!")
    
    # 6. Print Summary
    print(f"\n{'='*80}")
    print("✅ Merging Complete!")
    print(f"\nMerged model location: {OUTPUT_PATH}")
    print(f"Model size: ~14GB (full Qwen 7B with maritime knowledge)")
    print("\nYou can now use this model directly without PEFT/LoRA:")
    print(f"  >>> model = AutoModelForCausalLM.from_pretrained('{OUTPUT_PATH}')")
    print(f"{'='*80}\n")
    
    # 7. Optional: Test Generation
    print("Testing merged model with a sample prompt...\n")
    test_prompt = "Explain the maritime safety protocols for"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(merged_model.device)
    outputs = merged_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test Prompt: {test_prompt}")
    print(f"Generated: {generated_text}\n")

if __name__ == "__main__":
    merge_and_save()
