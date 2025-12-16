"""
Evaluate all 5 LoRA checkpoints on 100 random maritime queries.

This script:
1. Loads 100 random queries from the processed queries dataset
2. Evaluates all 5 checkpoints (4 from phase1a + 1 from phase1b)
3. Generates a CSV with ground truth and all checkpoint predictions

Checkpoints are loaded from HuggingFace: naga080898/qwen7b-marine

Usage:
    python evaluate_checkpoints.py
    
    # Use more/fewer samples
    python evaluate_checkpoints.py --num-samples 50
    
    # Set different random seed
    python evaluate_checkpoints.py --seed 42
"""

import os
import json
import random
import argparse
import logging
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Dataset
DATASET_PATH = "processed_queries_20251216_014609.jsonl"

# Base model and adapter repo
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO_ID = "naga080898/qwen7b-marine"

# All 5 checkpoints to evaluate
CHECKPOINTS = [
    {
        "name": "phase1a_ckpt2157",
        "subfolder": "phase1a-short-ckpt2157",
        "description": "Phase 1a Short Context (2,157 steps)"
    },
    {
        "name": "phase1a_ckpt4314",
        "subfolder": "phase1a-short-ckpt4314",
        "description": "Phase 1a Short Context (4,314 steps)"
    },
    {
        "name": "phase1a_ckpt6471",
        "subfolder": "phase1a-short-ckpt6471",
        "description": "Phase 1a Short Context (6,471 steps)"
    },
    {
        "name": "phase1a_ckpt8628",
        "subfolder": "phase1a-short-ckpt8628",
        "description": "Phase 1a Short Context (8,628 steps - Final)"
    },
    {
        "name": "phase1b_ckpt872",
        "subfolder": "phase1b-medium-ckpt872",
        "description": "Phase 1b Medium Context (872 steps)"
    },
]

# Generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # Low temperature for more deterministic outputs
DO_SAMPLE = False  # Greedy decoding for reproducibility

# Output
OUTPUT_DIR = "evaluation_results"


def load_random_queries(dataset_path: str, num_samples: int, seed: int) -> list:
    """
    Load random queries from the JSONL dataset.
    
    Args:
        dataset_path: Path to the JSONL file
        num_samples: Number of random samples to select
        seed: Random seed for reproducibility
        
    Returns:
        List of query dictionaries
    """
    logger.info(f"Loading queries from: {dataset_path}")
    
    # Load all queries
    all_queries = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_queries.append(json.loads(line))
    
    logger.info(f"Total queries in dataset: {len(all_queries)}")
    
    # Random sample
    random.seed(seed)
    selected_queries = random.sample(all_queries, min(num_samples, len(all_queries)))
    
    logger.info(f"Selected {len(selected_queries)} random queries (seed={seed})")
    
    return selected_queries


def format_prompt(query: str) -> str:
    """Format query using Qwen chat template."""
    return f"""<|im_start|>system
You are a highly knowledgeable maritime expert. You are provided with a query related to maritime regulations, safety standards, or operational manuals. Answer the query accurately and comprehensively.<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""


def generate_response(
    model,
    tokenizer,
    query: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE
) -> str:
    """
    Generate response from model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        query: Input query text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample or use greedy decoding
        
    Returns:
        Generated response text
    """
    prompt = format_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract only the generated part (after the prompt)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1].strip()
    else:
        # Fallback: remove the prompt portion
        response = full_response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
    
    return response


def load_model_with_adapter(base_model, adapter_repo_id: str, subfolder: str):
    """
    Load PEFT adapter on top of base model.
    
    Args:
        base_model: Pre-loaded base model
        adapter_repo_id: HuggingFace repo ID containing adapters
        subfolder: Subfolder within repo containing the specific checkpoint
        
    Returns:
        Model with adapter loaded
    """
    logger.info(f"Loading adapter from: {adapter_repo_id}/{subfolder}")
    
    model = PeftModel.from_pretrained(
        base_model,
        adapter_repo_id,
        subfolder=subfolder,
    )
    
    return model


def evaluate_checkpoint(
    base_model,
    tokenizer,
    checkpoint_config: dict,
    queries: list,
    adapter_repo_id: str
) -> list:
    """
    Evaluate a single checkpoint on all queries.
    
    Args:
        base_model: Pre-loaded base model
        tokenizer: Tokenizer
        checkpoint_config: Checkpoint configuration dict
        queries: List of query dicts
        adapter_repo_id: HuggingFace repo ID for adapters
        
    Returns:
        List of generated responses
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {checkpoint_config['description']}")
    logger.info(f"{'='*60}")
    
    # Load adapter
    model = load_model_with_adapter(
        base_model,
        adapter_repo_id,
        checkpoint_config["subfolder"]
    )
    model.eval()
    
    responses = []
    for query_data in tqdm(queries, desc=f"Evaluating {checkpoint_config['name']}"):
        query = query_data["query"]
        try:
            response = generate_response(model, tokenizer, query)
        except Exception as e:
            logger.error(f"Error generating response for query: {query[:50]}... Error: {e}")
            response = f"[ERROR: {str(e)}]"
        responses.append(response)
    
    # Clean up - unload adapter to free memory
    del model
    torch.cuda.empty_cache()
    
    return responses


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints on random maritime queries"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random queries to evaluate (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH,
        help=f"Path to dataset (default: {DATASET_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚢 MARITIME CHECKPOINT EVALUATION")
    print("=" * 80 + "\n")
    
    logger.info("Configuration:")
    logger.info(f"  • Samples: {args.num_samples}")
    logger.info(f"  • Seed: {args.seed}")
    logger.info(f"  • Dataset: {args.dataset}")
    logger.info(f"  • Base model: {BASE_MODEL_ID}")
    logger.info(f"  • Adapter repo: {ADAPTER_REPO_ID}")
    logger.info(f"  • Checkpoints: {len(CHECKPOINTS)}")
    
    # -------------------------------------------------------------------------
    # 1. Load Random Queries
    # -------------------------------------------------------------------------
    logger.info("\n[1/4] Loading random queries...")
    queries = load_random_queries(args.dataset, args.num_samples, args.seed)
    
    # -------------------------------------------------------------------------
    # 2. Load Base Model
    # -------------------------------------------------------------------------
    logger.info("\n[2/4] Loading base model...")
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Base model loaded: {BASE_MODEL_ID}")
    
    # -------------------------------------------------------------------------
    # 3. Evaluate Each Checkpoint
    # -------------------------------------------------------------------------
    logger.info("\n[3/4] Evaluating checkpoints...")
    
    results = {
        "query_id": [],
        "query": [],
        "ground_truth": [],
        "query_type": [],
        "difficulty": [],
    }
    
    # Add columns for each checkpoint
    for ckpt in CHECKPOINTS:
        results[ckpt["name"]] = []
    
    # Initialize with query data
    for i, query_data in enumerate(queries):
        results["query_id"].append(i)
        results["query"].append(query_data["query"])
        results["ground_truth"].append(query_data["answer"])
        results["query_type"].append(query_data.get("query_type", "unknown"))
        results["difficulty"].append(query_data.get("difficulty", "unknown"))
    
    # Evaluate each checkpoint
    for checkpoint in CHECKPOINTS:
        responses = evaluate_checkpoint(
            base_model,
            tokenizer,
            checkpoint,
            queries,
            ADAPTER_REPO_ID
        )
        results[checkpoint["name"]] = responses
    
    # -------------------------------------------------------------------------
    # 4. Save Results
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] Saving results...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"checkpoint_evaluation_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"checkpoint_evaluation_{timestamp}.json")
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ CSV saved: {csv_path}")
    
    # Save JSON (for easier inspection of long texts)
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"✅ JSON saved: {json_path}")
    
    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    
    logger.info(f"\nTotal queries evaluated: {len(queries)}")
    logger.info(f"Checkpoints evaluated: {len(CHECKPOINTS)}")
    
    # Query type distribution
    logger.info("\nQuery Type Distribution:")
    for qtype, count in df["query_type"].value_counts().items():
        logger.info(f"  • {qtype}: {count}")
    
    # Difficulty distribution
    logger.info("\nDifficulty Distribution:")
    for diff, count in df["difficulty"].value_counts().items():
        logger.info(f"  • {diff}: {count}")
    
    # Average response lengths
    logger.info("\nAverage Response Lengths (characters):")
    logger.info(f"  • Ground Truth: {df['ground_truth'].str.len().mean():.0f}")
    for ckpt in CHECKPOINTS:
        avg_len = df[ckpt["name"]].str.len().mean()
        logger.info(f"  • {ckpt['name']}: {avg_len:.0f}")
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print(f"Results saved to: {args.output_dir}/")
    print("=" * 80 + "\n")
    
    return df


if __name__ == "__main__":
    main()
