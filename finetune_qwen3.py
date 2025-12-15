from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "unsloth/Qwen3-30B-A3B-Instruct-2507"  # BF16 version (non-FP8 to avoid kernel bug)
MAX_SEQ_LENGTH = 4096  # Increased for longer maritime documents
DTYPE = None # None for auto detection
LOAD_IN_4BIT = True  # 4-bit QLoRA - stable training with minimal accuracy loss

# Dataset configuration
# Assumes a JSONL file with 'query' and 'answer' fields.
# Update the path to your custom dataset.
DATASET_PATH = "processed_queries_20251216_014609.jsonl" 

# LoRA Config - Optimized for 80GB GPU
LORA_R = 64  # Increased from 16 for better learning capacity
LORA_ALPHA = 128  # 2x rank is a good scaling factor
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# -----------------------------------------------------------------------------
# 1. Load Model
# -----------------------------------------------------------------------------
print(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

# -----------------------------------------------------------------------------
# 2. Add LoRA Adapters
# -----------------------------------------------------------------------------
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_R,
    target_modules = TARGET_MODULES,
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# -----------------------------------------------------------------------------
# 3. Prepare Dataset
# -----------------------------------------------------------------------------
prompt = """<|im_start|>system
You are a highly knowledgeable maritime expert. You are provided with a query related to maritime regulations, safety standards, or operational manuals. Answer the query accurately and comprehensively.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    queries = examples.get("query", [])
    answers = examples.get("answer", [])
    
    texts = []
    for query, answer in zip(queries, answers):
        # Must add EOS_TOKEN
        text = prompt.format(query, answer) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

print(f"Loading dataset from {DATASET_PATH}...")
try:
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
    print("Dataset loaded and formatted.")
    print("First 5 Training Examples:")
    for i in range(5):
        print(f"--- Example {i+1} ---")
        print(dataset["text"][i])
        print("-------------------")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have a valid JSONL file at 'custom_dataset.jsonl' with 'query' and 'answer' fields.")
    # Create a dummy dataset for demonstration if file lacks
    if "data_files" in str(e) or "FileNotFound" in str(e):
        print("Creating dummy dataset for demonstration...")
        from datasets import Dataset
        dummy_data = {
            "query": ["What is the capital of France?", "Explain quantum entanglement."],
            "answer": ["Paris.", "It is a phenomenon where particles become correlated..."]
        }
        dataset = Dataset.from_dict(dummy_data)
        dataset = dataset.map(formatting_prompts_func, batched = True)

# -----------------------------------------------------------------------------
# 4. Train
# -----------------------------------------------------------------------------
print("Starting training...")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    args = SFTConfig(
        per_device_train_batch_size = 16,  # Increased for better gradient estimates (~55GB VRAM)
        gradient_accumulation_steps = 2,   # Effective batch size = 16 * 2 = 32
        warmup_ratio = 0.03,  # 3% of total steps for warmup
        num_train_epochs = 3,  # Full training on dataset (3 epochs)
        learning_rate = 2e-4,
        logging_steps = 10,
        save_steps = 500,  # Save checkpoints every 500 steps
        save_total_limit = 3,  # Keep last 3 checkpoints
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",  # Cosine schedule works better for longer training
        seed = 3407,
        output_dir = "outputs_qwen3_finetune",
        bf16 = True,  # Use BF16 for faster training on H100
    ),
)

trainer_stats = trainer.train()
print("Training complete.")

# -----------------------------------------------------------------------------
# 5. Save Model
# -----------------------------------------------------------------------------
print("Saving model...")
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
print("Model saved to 'lora_model directory'.")
