# LoRA Adapter Upload and Merge Guide

This guide explains how to upload your trained LoRA adapters to HuggingFace Hub and merge them with the base model to create standalone models.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Upload Adapters to HuggingFace](#upload-adapters-to-huggingface)
3. [Merge Adapters with Base Model](#merge-adapters-with-base-model)
4. [Usage Examples](#usage-examples)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Install Required Dependencies

```bash
pip install huggingface_hub transformers peft torch
```

### 2. HuggingFace Authentication

You need to authenticate with HuggingFace to upload models:

**Option A: CLI Login (Recommended)**
```bash
huggingface-cli login
```

**Option B: Environment Variable**
```bash
export HF_TOKEN="your_huggingface_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Verify Checkpoint Availability

Make sure your checkpoint directories are available. Based on your terminal output:

- **Phase 1a**: `qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628/`
- **Phase 1b**: `qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872/`

If these are on a remote server, you'll need to either:
- Run the scripts on the server
- Copy the checkpoints to your local machine
- Update the paths in the scripts

---

## Upload Adapters to HuggingFace

### Step 1: Configure the Upload Script

Edit `upload_adapters_to_hf.py` and update:

```python
# Set your HuggingFace username
HF_USERNAME = "your_username"  # Change this!

# Configure adapters (already pre-configured for your checkpoints)
ADAPTERS = [
    {
        "name": "qwen-maritime-7b-phase1a-short",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628",
        ...
    },
    {
        "name": "qwen-maritime-7b-phase1b-medium",
        "local_path": "qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872",
        ...
    }
]
```

### Step 2: Run the Upload

```bash
python upload_adapters_to_hf.py
```

The script will:
- ✅ Create repositories on HuggingFace
- ✅ Generate model cards with usage instructions
- ✅ Upload adapter files (excluding training state files)
- ✅ Provide URLs to view your models

### What Gets Uploaded?

**Included:**
- `adapter_config.json` - Adapter configuration
- `adapter_model.safetensors` - Adapter weights
- `tokenizer.json`, `tokenizer_config.json` - Tokenizer files
- `vocab.json`, `merges.txt` - Vocabulary
- `chat_template.jinja` - Chat template
- Auto-generated `README.md` - Model card

**Excluded (training artifacts):**
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Learning rate scheduler
- `rng_state.pth` - Random state
- `trainer_state.json` - Training metrics
- `training_args.bin` - Training arguments

### Expected Output

```
================================================================================
HuggingFace Adapter Upload Tool
================================================================================

Will upload 2 adapter(s):
  1. qwen-maritime-7b-phase1a-short
     From: qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628
     To: your_username/qwen-maritime-7b-phase1a-short
  2. qwen-maritime-7b-phase1b-medium
     From: qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872
     To: your_username/qwen-maritime-7b-phase1b-medium

Proceed with upload? (y/n):
```

---

## Merge Adapters with Base Model

Merging creates standalone models that don't require PEFT at inference time.

### Step 1: Configure Paths (Optional)

The script `merge_lora_adapters.py` is already configured for your checkpoints. You can verify:

```python
ADAPTERS = {
    "phase1a": {
        "adapter_path": "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628",
        "output_path": "qwen-maritime-7b-merged-phase1a",
        ...
    },
    "phase1b": {
        "adapter_path": "qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872",
        "output_path": "qwen-maritime-7b-merged-phase1b",
        ...
    }
}
```

### Step 2: Run the Merge

**Merge all adapters:**
```bash
python merge_lora_adapters.py
```

**Merge specific adapter:**
```bash
# Merge only Phase 1a
python merge_lora_adapters.py --adapter phase1a

# Merge only Phase 1b
python merge_lora_adapters.py --adapter phase1b
```

**Merge and upload to HuggingFace:**
```bash
python merge_lora_adapters.py --upload
```

### What Happens During Merge?

```
[1/6] Loading base model: Qwen/Qwen2.5-7B-Instruct...
✅ Base model loaded (torch.bfloat16)

[2/6] Loading LoRA adapter from: checkpoint-8628...
✅ LoRA adapter loaded
   LoRA rank: 64
   LoRA alpha: 128
   Target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', ...]

[3/6] Merging LoRA weights into base model...
⏳ This may take a few minutes...
✅ Merge complete!

[4/6] Loading tokenizer...
✅ Tokenizer loaded

[5/6] Saving merged model to: qwen-maritime-7b-merged-phase1a...
✅ Merged model saved!

[6/6] Testing merged model with sample prompt...
Test Prompt: Explain the maritime safety protocols for
Generated: Explain the maritime safety protocols for cargo operations...
```

### Output Structure

After merging, you'll have:

```
qwen-maritime-7b-merged-phase1a/
├── config.json                    # Model configuration
├── generation_config.json         # Generation settings
├── model-00001-of-00003.safetensors  # Model weights (sharded)
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json   # Shard index
├── tokenizer.json                 # Tokenizer
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── README.md                      # Model card
```

**Size:** ~14GB per merged model

---

## Usage Examples

### Using Uploaded Adapter (from HuggingFace)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load your adapter from HuggingFace
model = PeftModel.from_pretrained(
    base_model, 
    "your_username/qwen-maritime-7b-phase1a-short"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "your_username/qwen-maritime-7b-phase1a-short"
)

# Generate
prompt = "Explain maritime safety protocols for cargo operations:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using Merged Model (Local)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model directly - no PEFT needed!
model = AutoModelForCausalLM.from_pretrained(
    "./qwen-maritime-7b-merged-phase1a",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen-maritime-7b-merged-phase1a"
)

# Generate
prompt = "What are the key considerations for vessel navigation?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using Merged Model (from HuggingFace)

After uploading the merged model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Even simpler - just load from HuggingFace
model = AutoModelForCausalLM.from_pretrained(
    "your_username/qwen-maritime-7b-merged-phase1a",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "your_username/qwen-maritime-7b-merged-phase1a"
)
```

---

## Comparison: Adapter vs Merged Model

| Feature | Adapter Model | Merged Model |
|---------|---------------|--------------|
| **File Size** | ~500MB | ~14GB |
| **Loading** | Requires base + adapter | Direct loading |
| **Dependencies** | Needs PEFT library | Standard transformers |
| **Inference Speed** | Slight overhead | Optimal |
| **Memory Usage** | Lower (efficient) | Higher (full model) |
| **Deployment** | More complex | Simpler |
| **Flexibility** | Can swap adapters | Single model |
| **Use Case** | Research, experimentation | Production deployment |

**When to use Adapter:**
- Storage is limited
- You want to try multiple adapters
- Research and experimentation
- Need to share multiple versions efficiently

**When to use Merged:**
- Production deployment
- Simplicity is priority
- Don't need PEFT flexibility
- Want optimal inference speed

---

## Troubleshooting

### Issue: "Local path does not exist"

**Problem:** Checkpoint directory not found

**Solutions:**
1. **If on remote server:** Run the scripts on the server where checkpoints exist
2. **If need local:** Copy checkpoints to your local machine:
   ```bash
   scp -r user@server:~/Maritime-LLM/qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628 ./
   ```
3. **Update paths:** Modify the `local_path` or `adapter_path` in the scripts

### Issue: "Please set your HuggingFace username"

**Problem:** Script has default username

**Solution:** Edit the script and change:
```python
HF_USERNAME = "YOUR_HF_USERNAME"
```
to:
```python
HF_USERNAME = "your_actual_username"
```

### Issue: "401 Client Error: Unauthorized"

**Problem:** Not authenticated with HuggingFace

**Solutions:**
1. Login via CLI:
   ```bash
   huggingface-cli login
   ```
2. Or set token:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

### Issue: "CUDA out of memory" during merge

**Problem:** GPU memory insufficient

**Solutions:**
1. Enable 8-bit mode:
   ```python
   USE_8BIT_FOR_MERGE = True
   ```
2. Use CPU:
   ```python
   DEVICE_MAP = "cpu"
   ```
3. Clear GPU cache before merging:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue: Merge is very slow

**Problem:** Large model, slow device

**Solutions:**
- Use GPU instead of CPU
- Be patient - merging 7B model takes 5-10 minutes on GPU
- Consider using a machine with more RAM/VRAM

### Issue: "Repository not found" when loading from HF

**Problem:** Repository doesn't exist or is private

**Solutions:**
1. Check repository URL: `https://huggingface.co/username/model-name`
2. If private, authenticate first
3. Verify repository name spelling

---

## Next Steps

After uploading and merging:

1. **Test your models** with maritime-specific prompts
2. **Share** your models with the community
3. **Document** which training phase works best for your use case
4. **Compare** Phase 1a vs Phase 1b performance
5. **Continue training** with later phases if needed

---

## Resources

- **HuggingFace Documentation**: https://huggingface.co/docs
- **PEFT Documentation**: https://huggingface.co/docs/peft
- **Transformers Documentation**: https://huggingface.co/docs/transformers

---

## Script Locations

- **Upload Script**: `upload_adapters_to_hf.py`
- **Merge Script**: `merge_lora_adapters.py`
- **Legacy Merge**: `merge_lora_adapter.py` (single adapter)

---

**Good luck with your maritime LLM deployment! 🚢**
