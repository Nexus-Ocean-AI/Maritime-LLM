# Checkpoint Resumption Guide

## ✅ Changes Made

The training script now supports **checkpoint resumption** to continue training from any completed phase without starting over.

### 1. Fixed the ValueError Bug
- **Issue**: Training crashed with `ValueError: Unknown format code 'f' for object of type 'str'`
- **Fix**: Added type checking before formatting the loss value in the callback

### 2. Added Checkpoint Resumption
- **New Parameter**: `START_FROM_PHASE` (line 94)
- **Default**: `START_FROM_PHASE = 1` (trains from scratch)

## 🚀 How to Resume Training

Since Phase_1a_Short (22+ hours) completed successfully, you can now resume from Phase_1b_Medium:

### Step 1: Update the Configuration
Change line 94 in `continual_pretraining_longcontext.py`:

```python
START_FROM_PHASE = 2  # Start from Phase_1b_Medium
```

### Step 2: Verify Checkpoint Exists
On your remote machine, confirm this path exists:
```
qwen-maritime-longcontext-cpt/Phase_1a_Short/
```

This directory should contain:
- `adapter_config.json`
- `adapter_model.safetensors` (or `adapter_model.bin`)
- Other checkpoint files

### Step 3: Run the Script
```bash
python continual_pretraining_longcontext.py
```

## 📋 What Happens When Resuming

When you set `START_FROM_PHASE = 2`:

1. **Loads Base Model**: Loads the Qwen/Qwen2.5-7B-Instruct model
2. **Applies LoRA Config**: Applies the same LoRA configuration
3. **Loads Checkpoint**: Loads the trained weights from `Phase_1a_Short/`
4. **Skips Phase 1**: Automatically skips Phase_1a_Short
5. **Continues Training**: Starts training Phase_1b_Medium (16K context)
6. **Proceeds to Phase 3**: After Phase_1b_Medium completes, continues to Phase_1c_Long (32K context)

## 🎯 Resume Options

- `START_FROM_PHASE = 1` → Train from scratch (all 3 phases)
- `START_FROM_PHASE = 2` → Resume from Phase_1b_Medium (loads Phase_1a_Short checkpoint)
- `START_FROM_PHASE = 3` → Resume from Phase_1c_Long (loads Phase_1b_Medium checkpoint)

## ⚠️ Important Notes

1. **Checkpoint Must Exist**: The script will error if the checkpoint from the previous phase doesn't exist
2. **Remote Path**: Make sure the checkpoint is at the correct path on your remote machine
3. **Same Configuration**: Use the same LoRA config (rank, alpha, dropout) as when training the checkpoint
4. **Data Consistency**: The script will use the same data for all phases

## 📊 Expected Output

When resuming, you'll see:
```
🔄 Resuming from checkpoint: qwen-maritime-longcontext-cpt/Phase_1a_Short
📂 Skipping phases 1-1, starting from phase 2

Loading model from checkpoint in torch.bfloat16...
✅ Using Flash Attention 2 for maximum speed
Loading LoRA adapter from qwen-maritime-longcontext-cpt/Phase_1a_Short...
✅ Successfully loaded checkpoint from Phase_1a_Short

⏭️  Skipping Phase_1a_Short (already completed)
================================================================================
PHASE 2/3: Phase_1b_Medium
Sequence Length: 16,384 tokens
Training Epochs: 3
================================================================================
```

## 🎉 Benefits

- **No Data Loss**: Preserves 22+ hours of Phase 1 training
- **Flexible Resumption**: Can resume from any completed phase
- **Easy to Use**: Just change one parameter
- **Safe**: Validates checkpoint exists before loading
