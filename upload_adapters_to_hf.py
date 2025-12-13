"""
Upload LoRA adapters to a single HuggingFace repository: qwen7b-marine

This script uploads multiple local checkpoints into subfolders of a single repository.
Structure:
  naga080898/qwen7b-marine/
    ├── README.md  (Main model card)
    ├── phase1a-ckpt2157/ (...adapter files...)
    ├── phase1a-ckpt8628/
    └── phase1b-ckpt872/
"""

import os
import logging
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HF_USERNAME = "naga080898"
REPO_NAME = "qwen7b-marine"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# Adapters to upload (Checkpoints)
ADAPTERS = [
    # Phase 1a - Short Context
    {
        "subfolder": "phase1a-short-ckpt2157",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-2157",
        "description": "Phase 1a Short Context (2,157 steps)",
        "training_steps": "2157",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "tags": ["maritime", "lora", "phase1a"]
    },
    {
        "subfolder": "phase1a-short-ckpt4314",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-4314",
        "description": "Phase 1a Short Context (4,314 steps)",
        "training_steps": "4314",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "tags": ["maritime", "lora", "phase1a"]
    },
    {
        "subfolder": "phase1a-short-ckpt6471",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-6471",
        "description": "Phase 1a Short Context (6,471 steps)",
        "training_steps": "6471",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "tags": ["maritime", "lora", "phase1a"]
    },
    {
        "subfolder": "phase1a-short-ckpt8628",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628",
        "description": "Phase 1a Short Context (8,628 steps - Final)",
        "training_steps": "8628",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "tags": ["maritime", "lora", "phase1a", "final"]
    },
    
    # Phase 1b - Medium Context
    {
        "subfolder": "phase1b-medium-ckpt872",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872",
        "description": "Phase 1b Medium Context (872 steps)",
        "training_steps": "872",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "tags": ["maritime", "lora", "phase1b"]
    }
]

# Files to upload/exclude
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.md", "*.jinja", "merges.txt", "vocab.json"]
IGNORE_PATTERNS = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin"]


def create_subfolder_readme(config: dict) -> str:
    """Create a README for the specific checkpoint folder."""
    return f"""---
license: apache-2.0
tags:
{chr(10).join(f'- {tag}' for tag in config['tags'])}
base_model: {config['base_model']}
---

# {config['description']}

This is a checkpoint for the **Maritime-LLM** project.

- **Phase**: {config['description']}
- **Training Steps**: {config['training_steps']}
- **Base Model**: {config['base_model']}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("{config['base_model']}")
# Load from specific subfolder
model = PeftModel.from_pretrained(
    base_model, 
    "{REPO_ID}",
    subfolder="{config['subfolder']}"
)
```
"""

def create_root_readme() -> str:
    """Create the main README for the repository root."""
    
    table_rows = []
    for adapter in ADAPTERS:
        row = f"| {adapter['description']} | `{adapter['subfolder']}` | {adapter['training_steps']} |"
        table_rows.append(row)
    
    table_md = "\n".join(table_rows)

    return f"""---
license: apache-2.0
tags:
- maritime
- qwen
- lora
- continual-pretraining
base_model: Qwen/Qwen2.5-7B-Instruct
layout: model
---

# Qwen 7B Maritime (Marine) Adapters

This repository contains **LoRA adapters** for the Maritime-LLM project, trained via progressive continual pretraining on maritime domain data.

## 📦 Available Checkpoints

All checkpoints are stored in subfolders within this repository.

| Description | Subfolder Name | Training Steps |
|-------------|----------------|----------------|
{table_md}

## 🚀 Usage

You can load any specific checkpoint by specifying the `subfolder` argument.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load Base Model
base_model_id = "Qwen/Qwen2.5-7B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    device_map="auto", 
    torch_dtype="auto"
)

# 2. Load Specific Adapter Checkpoint
# Example: Loading Phase 1a Final Checkpoint
adapter_id = "{REPO_ID}"
subfolder = "phase1a-short-ckpt8628" 

model = PeftModel.from_pretrained(
    base_model, 
    adapter_id, 
    subfolder=subfolder
)

# 3. Inference
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
prompt = "Explain the safety procedure for enclosing space entry:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Phases

1. **Phase 1a (Short Context)**: Foundation maritime knowledge training (Short context window).
2. **Phase 1b (Medium Context)**: Extended context training with longer documents.

"""

def main():
    print(f"\n{'='*80}")
    print(f"Uploading to Single Repository: {REPO_ID}")
    print(f"{'='*80}\n")

    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    # 1. Create Repository (if not exists)
    logger.info(f"Creating/Checking repository: {REPO_ID}")
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False, token=token)
        logger.info("✅ Repository ready")
    except Exception as e:
        logger.error(f"❌ Failed to create repo: {e}")
        return

    # 2. Upload Root README
    logger.info("Uploading main README.md...")
    root_readme = create_root_readme()
    with open("README_TEMP.md", "w") as f:
        f.write(root_readme)
    
    try:
        api.upload_file(
            path_or_fileobj="README_TEMP.md",
            path_in_repo="README.md",
            repo_id=REPO_ID,
            commit_message="Update main README with checkpoint list"
        )
        os.remove("README_TEMP.md")
        logger.info("✅ Main README uploaded")
    except Exception as e:
        logger.error(f"❌ Failed to upload root README: {e}")

    # 3. Upload Each Checkpoint to Subfolder
    for i, adapter in enumerate(ADAPTERS, 1):
        print(f"\n{'-'*60}")
        logger.info(f"Processing ({i}/{len(ADAPTERS)}): {adapter['subfolder']}")
        
        local_path = adapter['local_path']
        if not os.path.exists(local_path):
            logger.error(f"❌ Path not found: {local_path}")
            continue

        # Create subfolder README
        readme_content = create_subfolder_readme(adapter)
        readme_path = os.path.join(local_path, "README.md")
        
        # Backup existing README if it exists
        backup_path = None
        if os.path.exists(readme_path):
            backup_path = os.path.join(local_path, "README_backup.md")
            import shutil
            shutil.copy(readme_path, backup_path)
            
        with open(readme_path, "w") as f:
            f.write(readme_content)

        # Upload Folder
        try:
            logger.info(f"Uploading to folder: {adapter['subfolder']}...")
            api.upload_folder(
                folder_path=local_path,
                repo_id=REPO_ID,
                path_in_repo=adapter['subfolder'],
                allow_patterns=ALLOW_PATTERNS,
                ignore_patterns=IGNORE_PATTERNS,
                commit_message=f"Upload {adapter['subfolder']}"
            )
            logger.info("✅ Upload successful")
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
        
        # Restore backup README to keep local state clean
        # (Optional, but good practice if you run other scripts locally)
        if backup_path and os.path.exists(backup_path):
            import shutil
            shutil.move(backup_path, readme_path)

    print(f"\n{'='*80}")
    print(f"✅ All Done! View your repository here:")
    print(f"https://huggingface.co/{REPO_ID}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
