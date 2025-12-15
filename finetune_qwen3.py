# from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
# from trl import SFTTrainer, SFTConfig
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "unsloth/Qwen3-30B-A3B-Instruct-2507-FP8"
MAX_SEQ_LENGTH = 2048 # Adjust as needed
DTYPE = None # None for auto detection
LOAD_IN_4BIT = False # We want 8-bit/FP8, so we disable 4-bit loading

# Dataset configuration
# Assumes a JSONL file with 'query' and 'answer' fields.
# Update the path to your custom dataset.
DATASET_PATH = "processed_queries_20251216_014609.jsonl" 

# LoRA Config
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# -----------------------------------------------------------------------------
# 1. Load Model (SKIPPED FOR DATA CHECK)
# -----------------------------------------------------------------------------
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = MODEL_NAME,
#     max_seq_length = MAX_SEQ_LENGTH,
#     dtype = DTYPE,
#     load_in_4bit = LOAD_IN_4BIT,
# )

# -----------------------------------------------------------------------------
# 2. Add LoRA Adapters (SKIPPED)
# -----------------------------------------------------------------------------
# print("Adding LoRA adapters...")
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = LORA_R,
#     target_modules = TARGET_MODULES,
#     lora_alpha = LORA_ALPHA,
#     lora_dropout = LORA_DROPOUT,
#     bias = "none",
#     use_gradient_checkpointing = "unsloth", 
#     random_state = 3407,
#     use_rslora = False,
#     loftq_config = None,
# )

# -----------------------------------------------------------------------------
# 3. Prepare Dataset
# -----------------------------------------------------------------------------
prompt = """You are a highly knowledgeable maritime expert. You are provided with a query related to maritime regulations, safety standards, or operational manuals. Answer the query accurately and comprehensively.

### Query:
{}

### Answer:
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
    
    print("Data preparation check complete. Exiting before training.")
    sys.exit(0)

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
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Adjust for full training
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_qwen3_finetune",
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
