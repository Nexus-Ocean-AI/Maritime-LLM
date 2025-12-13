# Adapter Management Summary

## Quick Reference for Your LoRA Adapters

### 📦 Your Trained Adapters

Based on your terminal output, you have two trained LoRA adapters:

1. **Phase 1a - Short Context**
   - Path: `qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628/`
   - Checkpoint: 8628 steps
   - Files: ~500MB

2. **Phase 1b - Medium Context**
   - Path: `qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872/`
   - Checkpoint: 872 steps
   - Files: ~500MB

---

## 🚀 Quick Start

### Option 1: Interactive Script (Easiest)

```bash
./quickstart_adapters.sh
```

This interactive script guides you through:
- Installing dependencies
- Setting up HuggingFace authentication
- Uploading adapters
- Merging adapters
- Uploading merged models

### Option 2: Direct Commands

**Upload Adapters:**
```bash
python upload_adapters_to_hf.py
```

**Merge Adapters (all):**
```bash
python merge_lora_adapters.py
```

**Merge Specific Adapter:**
```bash
python merge_lora_adapters.py --adapter phase1a
python merge_lora_adapters.py --adapter phase1b
```

**Merge and Upload:**
```bash
python merge_lora_adapters.py --upload
```

---

## 🛠️ Setup Steps

### 1. Prerequisites

```bash
# Install dependencies
pip install huggingface_hub transformers peft torch accelerate

# Login to HuggingFace
huggingface-cli login
```

### 2. Configure Scripts

**Both scripts need your HuggingFace username:**

Edit `upload_adapters_to_hf.py`:
```python
HF_USERNAME = "your_username"  # Line 18
```

Edit `merge_lora_adapters.py`:
```python
HF_USERNAME = "YOUR_HF_USERNAME"  # Line 48
```

### 3. Verify Checkpoint Paths

The scripts are already configured for your checkpoint paths:
- `qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628`
- `qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872`

**Important:** If running locally, ensure these directories exist or copy them from your server.

---

## 📋 What Each Script Does

### `upload_adapters_to_hf.py`

**Purpose:** Upload LoRA adapters to HuggingFace Hub

**What it uploads:**
- ✅ Adapter weights (`adapter_model.safetensors`)
- ✅ Adapter config (`adapter_config.json`)
- ✅ Tokenizer files
- ✅ Auto-generated model card
- ❌ Training states (optimizer, scheduler, etc.)

**Output:** 
- Creates repositories on HuggingFace
- URLs: `https://huggingface.co/your_username/qwen-maritime-7b-phase1a-short`

---

### `merge_lora_adapters.py`

**Purpose:** Merge LoRA weights into base model to create standalone models

**Process:**
1. Loads base Qwen 7B model
2. Loads your LoRA adapter
3. Merges adapter weights into base weights
4. Saves merged model locally
5. Optionally uploads to HuggingFace

**Output:**
- Local directories with merged models (~14GB each)
- `qwen-maritime-7b-merged-phase1a/`
- `qwen-maritime-7b-merged-phase1b/`

**Advantages of Merged Models:**
- No PEFT dependency needed
- Simpler deployment
- Standard transformers inference
- Optimal performance

---

## 💾 File Sizes

| Item | Size | Description |
|------|------|-------------|
| LoRA Adapter | ~500MB | Just the adapter weights |
| Merged Model | ~14GB | Full model with adapter merged |
| Base Model | ~14GB | Original Qwen 7B |

**Storage Planning:**
- To upload adapters: Need source checkpoints available
- To merge: Need ~14GB per merged model + base model in cache
- GPU Memory: ~16GB per model (can use CPU or 8-bit mode)

---

## 🔍 File Structure

### Adapter Checkpoint
```
checkpoint-8628/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights (~500MB)
├── tokenizer.json               # Tokenizer
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── chat_template.jinja
├── optimizer.pt                 # Training states (not uploaded)
├── scheduler.pt                 # (not uploaded)
└── trainer_state.json           # (not uploaded)
```

### Merged Model
```
qwen-maritime-7b-merged-phase1a/
├── config.json
├── generation_config.json
├── model-00001-of-00003.safetensors  # Sharded weights
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json
├── tokenizer files...
└── README.md                         # Auto-generated card
```

---

## 🎯 Use Cases

### When to Use Adapters

**Upload to HuggingFace:**
- Share your work with community
- Version control for experiments
- Easy loading: `PeftModel.from_pretrained(base, "username/adapter")`
- Storage efficient (~500MB)
- Can swap multiple adapters on same base

**Example:**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "username/qwen-maritime-7b-phase1a-short")
```

### When to Use Merged Models

**For production deployment:**
- Simpler code (no PEFT dependency)
- Standard transformers workflow
- Slightly better inference speed
- Self-contained (~14GB)

**Example:**
```python
from transformers import AutoModelForCausalLM

# Direct loading, no PEFT needed
model = AutoModelForCausalLM.from_pretrained("./qwen-maritime-7b-merged-phase1a")
```

---

## 🔧 Configuration Options

### Upload Script (`upload_adapters_to_hf.py`)

```python
# Repository visibility
PRIVATE_REPO = False  # Set True for private repos

# Files to include
ALLOW_PATTERNS = ["*.json", "*.safetensors", ...]

# Files to exclude
IGNORE_PATTERNS = ["optimizer.pt", "scheduler.pt", ...]
```

### Merge Script (`merge_lora_adapters.py`)

```python
# Memory optimization
USE_8BIT_FOR_MERGE = False  # Set True if low on VRAM
DEVICE_MAP = "auto"         # Or "cpu" for CPU-only

# Auto-upload after merge
UPLOAD_TO_HF = False        # Set True to auto-upload
```

---

## ⚠️ Important Notes

### Running on Server vs Local

**Your checkpoints are on a remote server** (based on terminal showing `root@ai-model`)

**Option A: Run on Server**
```bash
# SSH to server
ssh root@ai-model

# Navigate to repo
cd ~/Maritime-LLM

# Copy scripts from local
# (scp upload_adapters_to_hf.py, merge_lora_adapters.py)

# Run scripts on server
python upload_adapters_to_hf.py
python merge_lora_adapters.py
```

**Option B: Copy to Local**
```bash
# From local machine
scp -r root@ai-model:~/Maritime-LLM/qwen-maritime-longcontext-cpt ./
```

### GPU Memory Requirements

**For Merging:**
- GPU (recommended): ~16GB VRAM per model
- CPU (slower): ~32GB RAM

**If low on memory:**
```python
USE_8BIT_FOR_MERGE = True  # In merge_lora_adapters.py
```

---

## 📚 Documentation

- **Comprehensive Guide**: `ADAPTER_UPLOAD_MERGE_GUIDE.md`
- **This Summary**: `ADAPTER_MANAGEMENT_SUMMARY.md`
- **Upload Script**: `upload_adapters_to_hf.py`
- **Merge Script**: `merge_lora_adapters.py`
- **Quick Start**: `quickstart_adapters.sh`

---

## 🐛 Troubleshooting

**"Checkpoint not found"**
- Ensure running on server where checkpoints exist
- Or copy checkpoints to local machine first

**"Set your HuggingFace username"**
- Edit scripts and change `HF_USERNAME = "your_username"`

**"401 Unauthorized"**
- Run `huggingface-cli login`
- Or set `HF_TOKEN` environment variable

**"CUDA out of memory"**
- Set `USE_8BIT_FOR_MERGE = True`
- Or use `DEVICE_MAP = "cpu"`

See full troubleshooting section in `ADAPTER_UPLOAD_MERGE_GUIDE.md`

---

## ✅ Checklist

Before running scripts:

- [ ] HuggingFace account created
- [ ] HuggingFace CLI authentication (`huggingface-cli login`)
- [ ] Dependencies installed (`pip install huggingface_hub transformers peft torch`)
- [ ] Scripts configured with your username
- [ ] Checkpoint paths verified/accessible
- [ ] Sufficient disk space (~14GB per merge)
- [ ] GPU available (or configured for CPU)

---

## 🎓 Next Steps

1. **Upload adapters** to HuggingFace for sharing
2. **Merge adapters** locally for testing
3. **Compare** Phase 1a vs Phase 1b performance
4. **Deploy** best-performing model
5. **Continue training** with subsequent phases if needed

---

**Need help? See `ADAPTER_UPLOAD_MERGE_GUIDE.md` for detailed instructions!**
