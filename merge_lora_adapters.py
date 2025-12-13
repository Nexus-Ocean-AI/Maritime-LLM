"""
Merge LoRA adapter weights into base Qwen model.

This creates standalone models that don't require PEFT/LoRA at inference time.
The merged models will be full parameter models with maritime knowledge baked in.

Usage:
    python merge_lora_adapters.py
    
    # Or merge specific adapter:
    python merge_lora_adapters.py --adapter phase1a
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Adapter Configurations
ADAPTERS = {
    "phase1a": {
        "name": "Phase 1a (Short Context)",
        "adapter_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628",
        "output_path": "qwen-maritime-7b-merged-phase1a",
        "description": "Merged model from Phase 1a short context training"
    },
    "phase1b": {
        "name": "Phase 1b (Medium Context)",
        "adapter_path": "qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872",
        "output_path": "qwen-maritime-7b-merged-phase1b",
        "description": "Merged model from Phase 1b medium context training"
    }
}

# Memory Settings
USE_8BIT_FOR_MERGE = False  # Set True if you have limited VRAM during merge
DEVICE_MAP = "auto"  # Automatically distribute across GPUs/CPU

# Upload to HuggingFace after merging
UPLOAD_TO_HF = False  # Set to True to upload merged models to HF Hub
HF_USERNAME = "YOUR_HF_USERNAME"  # Change this if uploading


def create_merged_model_card(config: dict, base_model: str) -> str:
    """Generate a model card for the merged model."""
    
    card = f"""---
language: en
license: apache-2.0
tags:
- maritime
- qwen
- merged
- instruct
base_model: {base_model}
---

# Qwen Maritime 7B - {config['name']}

## Model Description

{config['description']}

This is a **merged model** where the LoRA adapter weights have been combined with the base Qwen 7B model.
Unlike the adapter-only version, this model can be used directly without requiring PEFT.

## Base Model

- **Architecture**: Qwen 7B Instruct
- **Base**: `{base_model}`
- **Training**: Maritime domain continual pretraining with LoRA
- **Merged**: ✅ Standalone model (no PEFT needed)

## Training Data

The model was trained on maritime domain-specific data including:
- Maritime research papers
- Technical manuals and documentation
- Books on maritime operations
- Web-scraped maritime content

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model directly - no PEFT needed!
model = AutoModelForCausalLM.from_pretrained(
    "{HF_USERNAME}/{config['output_path']}",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("{HF_USERNAME}/{config['output_path']}")

# Generate
prompt = "Explain maritime safety protocols for cargo operations:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Performance Notes

- **Model Size**: ~14GB (full model weights)
- **Context Length**: Supports extended context (check model config)
- **Inference**: Standard transformers inference, no PEFT overhead

## Comparison with Adapter Version

| Feature | Merged Model | Adapter Model |
|---------|--------------|---------------|
| Size | ~14GB | ~500MB |
| Loading | Direct | Requires base + adapter |
| Inference | Standard | Requires PEFT |
| Deployment | Simpler | More complex |
| Memory | Higher | Lower |

## Citation

If you use this model, please cite the Maritime-LLM project.

## License

Apache 2.0 (inherited from base Qwen model)
"""
    
    return card


def merge_and_save(
    adapter_key: str,
    config: dict,
    base_model_id: str = BASE_MODEL_ID
) -> str:
    """
    Load base model, merge LoRA adapter, and save the merged model.
    
    Args:
        adapter_key: Key identifying the adapter configuration
        config: Adapter configuration dictionary
        base_model_id: HuggingFace model ID for base model
        
    Returns:
        Path to the merged model
    """
    
    adapter_path = config["adapter_path"]
    output_path = config["output_path"]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Merging: {config['name']}")
    logger.info(f"{'='*80}\n")
    
    # Check if adapter path exists
    if not os.path.exists(adapter_path):
        logger.error(f"❌ Adapter path not found: {adapter_path}")
        logger.error("Please ensure the checkpoint directory is available")
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    # 1. Load Base Model
    logger.info(f"[1/6] Loading base model: {base_model_id}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model_kwargs = {
        "device_map": DEVICE_MAP,
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    
    if USE_8BIT_FOR_MERGE:
        model_kwargs["load_in_8bit"] = True
        logger.info("Using 8-bit mode for memory efficiency")
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        logger.info(f"✅ Base model loaded ({dtype})")
    except Exception as e:
        logger.error(f"❌ Failed to load base model: {e}")
        raise
    
    # 2. Load LoRA Adapter
    logger.info(f"\n[2/6] Loading LoRA adapter from: {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("✅ LoRA adapter loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load adapter: {e}")
        raise
    
    # Check adapter info
    try:
        adapter_config_dict = model.peft_config['default']
        logger.info(f"   LoRA rank: {adapter_config_dict.r}")
        logger.info(f"   LoRA alpha: {adapter_config_dict.lora_alpha}")
        logger.info(f"   Target modules: {adapter_config_dict.target_modules}")
    except:
        pass
    
    # 3. Merge Adapter into Base Model
    logger.info("\n[3/6] Merging LoRA weights into base model...")
    logger.info("⏳ This may take a few minutes...")
    
    try:
        # merge_and_unload() combines LoRA weights with base weights
        merged_model = model.merge_and_unload()
        logger.info("✅ Merge complete!")
    except Exception as e:
        logger.error(f"❌ Failed to merge: {e}")
        raise
    
    # 4. Load Tokenizer
    logger.info(f"\n[4/6] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        raise
    
    # 5. Save Merged Model
    logger.info(f"\n[5/6] Saving merged model to: {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Save in safetensors format (recommended, smaller and safer)
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,  # Use safetensors
            max_shard_size="5GB"  # Shard large models
        )
        tokenizer.save_pretrained(output_path)
        
        # Create and save model card
        model_card = create_merged_model_card(config, base_model_id)
        with open(os.path.join(output_path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card)
        
        logger.info("✅ Merged model saved!")
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        raise
    
    # 6. Optional: Test Generation
    logger.info("\n[6/6] Testing merged model with sample prompt...")
    test_prompt = "Explain the maritime safety protocols for"
    
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt").to(merged_model.device)
        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\n{'─'*80}")
        logger.info(f"Test Prompt: {test_prompt}")
        logger.info(f"Generated:\n{generated_text}")
        logger.info(f"{'─'*80}\n")
    except Exception as e:
        logger.warning(f"⚠️  Test generation failed: {e}")
    
    # Print Summary
    logger.info(f"\n{'='*80}")
    logger.info("✅ Merging Complete!")
    logger.info(f"\nMerged model location: {output_path}")
    logger.info(f"Model size: ~14GB (full Qwen 7B with maritime knowledge)")
    logger.info("\nYou can now use this model directly without PEFT/LoRA:")
    logger.info(f"  >>> model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    logger.info(f"{'='*80}\n")
    
    return output_path


def upload_to_huggingface(model_path: str, repo_name: str):
    """Upload merged model to HuggingFace Hub."""
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return
    
    if HF_USERNAME == "YOUR_HF_USERNAME":
        logger.error("❌ Please set your HuggingFace username!")
        return
    
    repo_id = f"{HF_USERNAME}/{repo_name}"
    logger.info(f"\n{'='*80}")
    logger.info(f"Uploading to HuggingFace: {repo_id}")
    logger.info(f"{'='*80}\n")
    
    try:
        # Create repo
        logger.info("Creating repository...")
        create_repo(repo_id=repo_id, exist_ok=True, private=False)
        
        # Upload
        logger.info("Uploading model files...")
        api = HfApi()
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model"
        )
        
        logger.info(f"✅ Successfully uploaded to: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument(
        "--adapter",
        type=str,
        choices=list(ADAPTERS.keys()) + ["all"],
        default="all",
        help="Which adapter to merge (default: all)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace after merging"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("LoRA Adapter Merging Tool")
    print(f"{'='*80}\n")
    
    # Determine which adapters to merge
    if args.adapter == "all":
        adapters_to_merge = ADAPTERS.items()
        logger.info(f"Will merge all {len(ADAPTERS)} adapters")
    else:
        adapters_to_merge = [(args.adapter, ADAPTERS[args.adapter])]
        logger.info(f"Will merge: {ADAPTERS[args.adapter]['name']}")
    
    # Display merge plan
    for key, config in adapters_to_merge:
        logger.info(f"  • {config['name']}")
        logger.info(f"    Adapter: {config['adapter_path']}")
        logger.info(f"    Output: {config['output_path']}")
    
    print()
    
    # Merge each adapter
    success_count = 0
    failed_count = 0
    merged_models = []
    
    for key, config in adapters_to_merge:
        try:
            output_path = merge_and_save(key, config)
            merged_models.append((key, config, output_path))
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to merge {config['name']}: {e}")
            failed_count += 1
        print(f"\n{'-'*80}\n")
    
    # Final summary
    print(f"{'='*80}")
    print("Merge Summary")
    print(f"{'='*80}")
    logger.info(f"✅ Success: {success_count}/{len(list(adapters_to_merge))}")
    if failed_count > 0:
        logger.error(f"❌ Failed: {failed_count}/{len(list(adapters_to_merge))}")
    
    # Optional upload
    if args.upload or UPLOAD_TO_HF:
        logger.info("\n📤 Uploading merged models to HuggingFace...")
        for key, config, model_path in merged_models:
            upload_to_huggingface(model_path, config["output_path"])
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
