"""
Upload LoRA adapters to HuggingFace Hub.

This script uploads trained LoRA adapters from local checkpoint directories
to HuggingFace Hub for sharing and versioning.

Usage:
    python upload_adapters_to_hf.py

Prerequisites:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Login to HuggingFace: huggingface-cli login
    3. Or set HF_TOKEN environment variable
"""

import os
import logging
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# HuggingFace Configuration
HF_USERNAME = "YOUR_HF_USERNAME"  # Change this to your HuggingFace username
HF_TOKEN = os.getenv("HF_TOKEN")  # Or set token here directly (not recommended for security)

# Adapter Configurations
ADAPTERS = [
    # Phase 1a - Short Context (4 checkpoints)
    {
        "name": "qwen-maritime-7b-phase1a-ckpt2157",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-2157",
        "description": "Qwen 7B Maritime LoRA adapter - Phase 1a Short Context (checkpoint 2157)",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "context_length": "Short context training phase",
        "training_steps": "2157",
        "tags": ["maritime", "lora", "qwen", "phase1a", "short-context", "checkpoint-2157"]
    },
    {
        "name": "qwen-maritime-7b-phase1a-ckpt4314",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-4314",
        "description": "Qwen 7B Maritime LoRA adapter - Phase 1a Short Context (checkpoint 4314)",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "context_length": "Short context training phase",
        "training_steps": "4314",
        "tags": ["maritime", "lora", "qwen", "phase1a", "short-context", "checkpoint-4314"]
    },
    {
        "name": "qwen-maritime-7b-phase1a-ckpt6471",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-6471",
        "description": "Qwen 7B Maritime LoRA adapter - Phase 1a Short Context (checkpoint 6471)",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "context_length": "Short context training phase",
        "training_steps": "6471",
        "tags": ["maritime", "lora", "qwen", "phase1a", "short-context", "checkpoint-6471"]
    },
    {
        "name": "qwen-maritime-7b-phase1a-ckpt8628",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628",
        "description": "Qwen 7B Maritime LoRA adapter - Phase 1a Short Context (checkpoint 8628, final)",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "context_length": "Short context training phase",
        "training_steps": "8628",
        "tags": ["maritime", "lora", "qwen", "phase1a", "short-context", "checkpoint-8628", "final"]
    },
    
    # Phase 1b - Medium Context (1 checkpoint)
    {
        "name": "qwen-maritime-7b-phase1b-ckpt872",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872",
        "description": "Qwen 7B Maritime LoRA adapter - Phase 1b Medium Context (checkpoint 872)",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "context_length": "Medium context training phase",
        "training_steps": "872",
        "tags": ["maritime", "lora", "qwen", "phase1b", "medium-context", "checkpoint-872"]
    }
]

# Upload Settings
PRIVATE_REPO = False  # Set to True if you want private repositories
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.md",
    "*.jinja",
    "merges.txt",
    "vocab.json"
]
IGNORE_PATTERNS = [
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin"
]


def create_model_card(adapter_config: dict) -> str:
    """Generate a model card in markdown format."""
    
    card = f"""---
language: en
license: apache-2.0
tags:
{chr(10).join(f'- {tag}' for tag in adapter_config['tags'])}
base_model: {adapter_config['base_model']}
---

# {adapter_config['name']}

## Model Description

{adapter_config['description']}

This is a LoRA (Low-Rank Adaptation) adapter trained for maritime domain adaptation.

## Base Model

- **Base Model**: `{adapter_config['base_model']}`
- **Context Length**: {adapter_config['context_length']}

## Training Details

This adapter was trained using:
- **Training Steps**: {adapter_config.get('training_steps', 'N/A')}
- Progressive long-context continual pretraining
- Maritime domain-specific data (papers, books, technical documents)
- LoRA rank: 16-64 (check adapter_config.json for exact values)
- Target modules: Attention and MLP layers

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{adapter_config['base_model']}",
    torch_dtype="auto",
    device_map="auto"
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "{HF_USERNAME}/{adapter_config['name']}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{HF_USERNAME}/{adapter_config['name']}")

# Generate
prompt = "Explain maritime safety protocols for"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Merging with Base Model

To create a standalone model without needing PEFT:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("{adapter_config['base_model']}")
model = PeftModel.from_pretrained(base_model, "{HF_USERNAME}/{adapter_config['name']}")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
```

## Citation

If you use this adapter, please cite the Maritime-LLM project.

## License

Apache 2.0
"""
    
    return card


def upload_adapter_to_hf(
    adapter_config: dict,
    hf_token: Optional[str] = None
) -> None:
    """
    Upload a single adapter to HuggingFace Hub.
    
    Args:
        adapter_config: Dictionary with adapter configuration
        hf_token: HuggingFace API token (optional if already logged in)
    """
    
    local_path = adapter_config["local_path"]
    repo_name = adapter_config["name"]
    repo_id = f"{HF_USERNAME}/{repo_name}"
    
    logger.info(f"Starting upload for {repo_name}...")
    
    # Check if local path exists
    if not os.path.exists(local_path):
        logger.error(f"❌ Local path does not exist: {local_path}")
        logger.error(f"Please ensure the checkpoint directory is available")
        return
    
    try:
        # Initialize HF API
        api = HfApi(token=hf_token)
        
        # Create repository
        logger.info(f"Creating repository: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                private=PRIVATE_REPO,
                repo_type="model",
                exist_ok=True,
                token=hf_token
            )
            logger.info(f"✅ Repository created/verified: {repo_id}")
        except Exception as e:
            logger.warning(f"Repository might already exist: {e}")
        
        # Generate and save model card
        logger.info("Generating model card...")
        model_card_content = create_model_card(adapter_config)
        model_card_path = Path(local_path) / "README.md"
        
        # Backup existing README if present
        if model_card_path.exists():
            logger.info("Backing up existing README.md...")
            backup_path = Path(local_path) / "README_backup.md"
            import shutil
            shutil.copy(model_card_path, backup_path)
        
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)
        logger.info("✅ Model card created")
        
        # Upload folder
        logger.info(f"Uploading adapter from {local_path}...")
        logger.info("This may take a few minutes depending on file size and network speed...")
        
        upload_folder(
            repo_id=repo_id,
            folder_path=local_path,
            path_in_repo="",
            token=hf_token,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            commit_message=f"Upload {repo_name} adapter"
        )
        
        logger.info(f"✅ Successfully uploaded {repo_name}!")
        logger.info(f"🌐 View at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        logger.error(f"❌ Error uploading {repo_name}: {e}")
        raise


def main():
    """Main execution function."""
    
    print(f"\n{'='*80}")
    print("HuggingFace Adapter Upload Tool")
    print(f"{'='*80}\n")
    
    # Validate configuration
    if HF_USERNAME == "YOUR_HF_USERNAME":
        logger.error("❌ Please set your HuggingFace username in the script!")
        logger.error("Change HF_USERNAME = 'YOUR_HF_USERNAME' to your actual username")
        return
    
    # Check authentication
    if not HF_TOKEN:
        logger.warning("⚠️  No HF_TOKEN found in environment")
        logger.info("Please make sure you're logged in via: huggingface-cli login")
        logger.info("Or set the HF_TOKEN environment variable")
        print()
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Aborted by user")
            return
    
    # Display upload plan
    logger.info(f"Will upload {len(ADAPTERS)} adapter(s):")
    for i, adapter in enumerate(ADAPTERS, 1):
        logger.info(f"  {i}. {adapter['name']}")
        logger.info(f"     From: {adapter['local_path']}")
        logger.info(f"     To: {HF_USERNAME}/{adapter['name']}")
    
    print()
    response = input("Proceed with upload? (y/n): ")
    if response.lower() != 'y':
        logger.info("Aborted by user")
        return
    
    # Upload each adapter
    print(f"\n{'='*80}\n")
    success_count = 0
    failed_count = 0
    
    for i, adapter_config in enumerate(ADAPTERS, 1):
        logger.info(f"Processing adapter {i}/{len(ADAPTERS)}")
        try:
            upload_adapter_to_hf(adapter_config, HF_TOKEN)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to upload {adapter_config['name']}: {e}")
            failed_count += 1
        print(f"\n{'-'*80}\n")
    
    # Final summary
    print(f"{'='*80}")
    print("Upload Summary")
    print(f"{'='*80}")
    logger.info(f"✅ Success: {success_count}/{len(ADAPTERS)}")
    if failed_count > 0:
        logger.error(f"❌ Failed: {failed_count}/{len(ADAPTERS)}")
    
    if success_count > 0:
        logger.info("\n📦 Your adapters are now available on HuggingFace!")
        logger.info("Others can use them with:")
        logger.info("  from peft import PeftModel")
        logger.info(f"  model = PeftModel.from_pretrained(base_model, '{HF_USERNAME}/adapter-name')")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
