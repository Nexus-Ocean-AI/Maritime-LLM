# Upload 5 Checkpoints to Single Repository - Summary

## 📦 Target Repository
**Repo ID:** `naga080898/qwen7b-marine`

Instead of 5 separate repositories, we are organizing everything into **one repository with subfolders**. This is cleaner and easier to manage.

### Structure

```
naga080898/qwen7b-marine/
├── README.md  (Main documentation & table of contents)
├── phase1a-short-ckpt2157/
│   ├── adapter_model.safetensors
│   └── ...
├── phase1a-short-ckpt4314/
├── phase1a-short-ckpt6471/
├── phase1a-short-ckpt8628/  (Phase 1a Final)
└── phase1b-medium-ckpt872/
```

---

## 🚀 How to Upload

Since you are already logged in, simply run:

```bash
python upload_adapters_to_hf.py
```

The script will:
1. Create the repo `qwen7b-marine` automatically.
2. Generate and upload the main `README.md`.
3. Iterate through all 5 local checkpoints and upload them to their specific subfolders.

---

## 💻 Usage Example

To load a specific checkpoint from this unified repository:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Base Model
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 2. Load Specific Adapter (e.g., Phase 1a Final)
model = PeftModel.from_pretrained(
    base, 
    "naga080898/qwen7b-marine", 
    subfolder="phase1a-short-ckpt8628"
)
```

This keeps your HuggingFace profile clean and your maritime model versions organized in one place! 🚢
