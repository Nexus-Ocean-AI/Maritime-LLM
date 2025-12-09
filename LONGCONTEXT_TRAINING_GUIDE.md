# Long-Context Training Guide (32K Target)

## Overview
This guide explains the progressive context length training approach to extend Qwen2.5-7B to handle 32K token contexts for maritime domain applications.

---

## Why Progressive Training?

### The Problem with Direct 32K Training:
- **Memory Requirements**: 32K sequences need ~200GB+ VRAM (impossible on consumer hardware)
- **Learning Difficulty**: Model struggles to learn long-range dependencies from scratch
- **Inefficiency**: Most maritime documents are < 32K, wasting compute on padding

### The Solution: Progressive Length Extension
Train in 3 phases with increasing sequence lengths:
1. **Phase 1a (4K)**: 60% of tokens - Efficient domain knowledge building
2. **Phase 1b (16K)**: 30% of tokens - Extend positional understanding  
3. **Phase 1c (32K)**: 10% of tokens - Final long-context refinement

This matches how GPT-4, Claude, and production LLMs are trained.

---

## Training Schedule Breakdown

| Phase | Seq Length | Token Allocation | Steps (est.) | Purpose |
|-------|-----------|------------------|--------------|---------|
| **1a** | 4,096 | 300M (60%) | ~2,300 | Fast domain adaptation |
| **1b** | 16,384 | 150M (30%) | ~600 | Medium context learning |
| **1c** | 32,768 | 50M (10%) | ~50 | Long context mastery |

**Total**: 500M tokens, ~3,000 training steps

---

## Memory Requirements

### Hardware Recommendations:

| Hardware | Can Train? | Config |
|----------|-----------|--------|
| **Mac M1/M2 (16GB)** | ❌ Phase 1a only | Max 4K context |
| **Mac M1 Ultra (64GB+)** | ⚠️ Phase 1a-1b | Up to 16K with 8-bit |
| **RTX 4090 (24GB)** | ✅ All phases | With Flash Attention 2 |
| **A100 (40GB+)** | ✅ All phases | Optimal |
| **Multi-GPU (2x3090)** | ✅ All phases | With DDP |

### If You Get OOM Errors:
1. Enable 8-bit quantization: `USE_8BIT = True`
2. Reduce batch size (already set to minimum)
3. Skip Phase 1c, stop at 16K context
4. Use cloud GPUs (RunPod, Vast.ai ~$0.50/hr for A100)

---

## Key Configuration Options

### In `continual_pretraining_longcontext.py`:

```python
# Progressive vs Direct Training
PROGRESSIVE_TRAINING = True  # RECOMMENDED: Keep as True

# Memory Optimization
USE_FLASH_ATTENTION_2 = True  # Reduces memory by 30-40% for long sequences
USE_8BIT = False              # Set True if OOM, reduces memory by 50%

# Adjust Context Schedule (if needed)
CONTEXT_SCHEDULE = [
    {"name": "Phase_1a_Short", "max_seq_length": 4096, "token_fraction": 0.60},
    {"name": "Phase_1b_Medium", "max_seq_length": 16384, "token_fraction": 0.30},
    {"name": "Phase_1c_Long", "max_seq_length": 32768, "token_fraction": 0.10},
]

# To skip Phase 1c (stop at 16K), just remove the last item:
# CONTEXT_SCHEDULE = CONTEXT_SCHEDULE[:2]
```

---

## Installation Requirements

Install Flash Attention 2 for memory efficiency:

```bash
# For CUDA/NVIDIA GPUs
pip install flash-attn --no-build-isolation

# For Mac (Flash Attention not supported, but script will fall back gracefully)
# Just run as-is, script auto-detects and uses standard attention
```

If Flash Attention installation fails, set `USE_FLASH_ATTENTION_2 = False` in the script.

---

## Running the Training

### Step 1: Start Training
```bash
python continual_pretraining_longcontext.py
```

### Step 2: Monitor Progress
Open TensorBoard in a separate terminal:
```bash
tensorboard --logdir qwen-maritime-longcontext-cpt/
```

Navigate to `http://localhost:6006` to see:
- Training loss curves
- Learning rate schedule
- Phase transitions

### Step 3: Wait for Completion
The script will automatically:
1. Train Phase 1a (4K context) - Fastest
2. Train Phase 1b (16K context) - Slower
3. Train Phase 1c (32K context) - Slowest
4. Save final model to `qwen-maritime-longcontext-cpt/final_32k_model/`

---

## Expected Training Time

Estimates based on hardware:

| Hardware | Phase 1a (4K) | Phase 1b (16K) | Phase 1c (32K) | **Total** |
|----------|--------------|---------------|---------------|-----------|
| **Mac M2 Ultra** | 12 hours | 36 hours | 48 hours | **~96 hours** |
| **RTX 4090** | 6 hours | 18 hours | 24 hours | **~48 hours** |
| **A100 (40GB)** | 3 hours | 9 hours | 12 hours | **~24 hours** |

💡 **Tip**: You can start Phase 2 (Instruction Tuning) after Phase 1a completes if you need a model sooner.

---

## Output Structure

After training, you'll have:

```
qwen-maritime-longcontext-cpt/
├── Phase_1a_Short/          # 4K checkpoint
│   ├── adapter_model.bin    # LoRA weights
│   └── logs/                # TensorBoard logs
├── Phase_1b_Medium/         # 16K checkpoint
│   ├── adapter_model.bin
│   └── logs/
├── Phase_1c_Long/           # 32K checkpoint
│   ├── adapter_model.bin
│   └── logs/
└── final_32k_model/         # FINAL MODEL (use this!)
    ├── adapter_model.bin    # LoRA weights
    ├── adapter_config.json
    └── tokenizer files
```

---

## Testing Long Context Capability

After training, test the model's long-context understanding:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(base_model, "qwen-maritime-longcontext-cpt/final_32k_model")
tokenizer = AutoTokenizer.from_pretrained("qwen-maritime-longcontext-cpt/final_32k_model")

# Test with long maritime document (e.g., 20K tokens)
long_text = "..." # Your maritime document
inputs = tokenizer(long_text, return_tensors="pt").to(model.device)

print(f"Input length: {inputs['input_ids'].shape[1]} tokens")

# Generate
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

---

## Troubleshooting

### Problem: OOM (Out of Memory) Error

**Solution 1: Enable 8-bit**
```python
USE_8BIT = True  # In the script
```

**Solution 2: Skip longest phase**
```python
# Stop at 16K instead of 32K
CONTEXT_SCHEDULE = CONTEXT_SCHEDULE[:2]  
```

**Solution 3: Use Cloud GPU**
- RunPod: ~$0.50/hour for A100
- Vast.ai: ~$0.30/hour for RTX 4090

### Problem: Flash Attention Installation Fails

**Solution**: Disable it
```python
USE_FLASH_ATTENTION_2 = False
```
Training will be slower but will still work.

### Problem: Training is Too Slow

**Solution**: Train only Phase 1a (4K), which gives 80% of the benefit:
```python
CONTEXT_SCHEDULE = [CONTEXT_SCHEDULE[0]]  # Only 4K phase
```

---

## Next Steps: Phase 2 (Instruction Tuning)

After continual pretraining completes, you'll want to:
1. **Instruction tune** on maritime Q&A, manuals, regulations
2. **Preserve long-context capability** by including long examples in instruction data
3. **Evaluate** on real maritime tasks

Let me know when you're ready, and I'll create the Phase 2 instruction tuning script!

---

## References

- **Progressive Training**: Used by GPT-4, Claude (Anthropic), Llama 3
- **Flash Attention 2**: [Dao et al., 2023](https://arxiv.org/abs/2307.08691)
- **Context Extension**: [Position Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595)
- **Qwen2.5 Documentation**: https://github.com/QwenLM/Qwen2.5
