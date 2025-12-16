"""
Evaluate the SFT fine-tuned model on maritime queries.

This script:
1. Loads the SFT LoRA adapter from outputs_qwen2.5_7b_sft/final_model
2. Evaluates on random queries from the dataset
3. Optionally compares with base model (no adapter)
4. Saves results as CSV and JSON

Usage:
    python evaluate_sft_model.py
    
    # Use more/fewer samples
    python evaluate_sft_model.py --num-samples 50
    
    # Compare with base model
    python evaluate_sft_model.py --compare-base
    
    # Custom SFT model path
    python evaluate_sft_model.py --sft-model-path /path/to/sft/model
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

# Models
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL_PATH = "outputs_qwen2.5_7b_sft/final_model"

# Generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
DO_SAMPLE = False

# Output
OUTPUT_DIR = "evaluation_results"


def load_random_queries(dataset_path: str, num_samples: int, seed: int) -> list:
    """Load random queries from the JSONL dataset."""
    logger.info(f"Loading queries from: {dataset_path}")
    
    all_queries = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_queries.append(json.loads(line))
    
    logger.info(f"Total queries in dataset: {len(all_queries)}")
    
    random.seed(seed)
    selected_queries = random.sample(all_queries, min(num_samples, len(all_queries)))
    
    logger.info(f"Selected {len(selected_queries)} random queries (seed={seed})")
    
    return selected_queries


def format_prompt(query: str) -> str:
    """Format query using Qwen chat template (same as training)."""
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
    """Generate response from model."""
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
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1].strip()
    else:
        response = full_response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
    
    return response


def evaluate_model(
    model,
    tokenizer,
    queries: list,
    model_name: str
) -> list:
    """Evaluate a model on all queries."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")
    
    model.eval()
    
    responses = []
    for query_data in tqdm(queries, desc=f"Evaluating {model_name}"):
        query = query_data["query"]
        try:
            response = generate_response(model, tokenizer, query)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = f"[ERROR: {str(e)}]"
        responses.append(response)
    
    return responses


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Evaluate SFT model on maritime queries"
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
        "--sft-model-path",
        type=str,
        default=SFT_MODEL_PATH,
        help=f"Path to SFT model (default: {SFT_MODEL_PATH})"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also evaluate base model for comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚢 MARITIME SFT MODEL EVALUATION")
    print("=" * 80 + "\n")
    
    logger.info("Configuration:")
    logger.info(f"  • Samples: {args.num_samples}")
    logger.info(f"  • Seed: {args.seed}")
    logger.info(f"  • Dataset: {args.dataset}")
    logger.info(f"  • Base model: {BASE_MODEL_ID}")
    logger.info(f"  • SFT model: {args.sft_model_path}")
    logger.info(f"  • Compare with base: {args.compare_base}")
    
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
    # 3. Evaluate Models
    # -------------------------------------------------------------------------
    logger.info("\n[3/4] Evaluating models...")
    
    results = {
        "query_id": [],
        "query": [],
        "ground_truth": [],
        "query_type": [],
        "difficulty": [],
    }
    
    # Initialize with query data
    for i, query_data in enumerate(queries):
        results["query_id"].append(i)
        results["query"].append(query_data["query"])
        results["ground_truth"].append(query_data["answer"])
        results["query_type"].append(query_data.get("query_type", "unknown"))
        results["difficulty"].append(query_data.get("difficulty", "unknown"))
    
    # Evaluate base model if requested
    if args.compare_base:
        logger.info("\n📊 Evaluating BASE model (no fine-tuning)...")
        base_responses = evaluate_model(base_model, tokenizer, queries, "base_model")
        results["base_model"] = base_responses
    
    # Load and evaluate SFT model
    logger.info(f"\n📊 Loading SFT adapter from: {args.sft_model_path}")
    
    if not os.path.exists(args.sft_model_path):
        logger.error(f"❌ SFT model not found at: {args.sft_model_path}")
        logger.info("Please provide the correct path using --sft-model-path")
        return None
    
    sft_model = PeftModel.from_pretrained(
        base_model,
        args.sft_model_path,
    )
    logger.info("✅ SFT adapter loaded")
    
    sft_responses = evaluate_model(sft_model, tokenizer, queries, "sft_model")
    results["sft_model"] = sft_responses
    
    # Clean up
    del sft_model
    torch.cuda.empty_cache()
    
    # -------------------------------------------------------------------------
    # 4. Save Results
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] Saving results...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"sft_evaluation_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"sft_evaluation_{timestamp}.json")
    
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ CSV saved: {csv_path}")
    
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"✅ JSON saved: {json_path}")
    
    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    
    logger.info(f"\nTotal queries evaluated: {len(queries)}")
    
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
    if args.compare_base:
        logger.info(f"  • Base Model: {df['base_model'].str.len().mean():.0f}")
    logger.info(f"  • SFT Model: {df['sft_model'].str.len().mean():.0f}")
    
    # Show sample comparisons
    print("\n" + "=" * 80)
    print("📝 SAMPLE COMPARISONS (First 3 queries)")
    print("=" * 80)
    
    for i in range(min(3, len(queries))):
        print(f"\n{'─'*80}")
        print(f"Query {i+1}: {queries[i]['query'][:100]}...")
        print(f"\n🎯 Ground Truth:\n{results['ground_truth'][i][:300]}...")
        if args.compare_base:
            print(f"\n🔵 Base Model:\n{results['base_model'][i][:300]}...")
        print(f"\n🟢 SFT Model:\n{results['sft_model'][i][:300]}...")
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print(f"Results saved to: {args.output_dir}/")
    print("=" * 80 + "\n")
    
    return df


if __name__ == "__main__":
    main()
