"""
Evaluate the SFT V2 fine-tuned model on its test set.

This script:
1. Loads the SFT V2 model from the training output directory
2. Evaluates on the saved test_dataset.jsonl
3. Computes evaluation metrics
4. Saves results as CSV

Usage:
    python evaluate_sft_v2.py
    
    # Specify model directory
    python evaluate_sft_v2.py --model-dir outputs_qwen2.5_7b_sft_v2_20251217_105003
    
    # Limit samples
    python evaluate_sft_v2.py --num-samples 50
"""

import os
import json
import argparse
import logging
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Default paths
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_DIR = "outputs_qwen2.5_7b_sft_v2_20251217_105003"

# Generation settings
MAX_NEW_TOKENS = 4096  # Increased to allow comprehensive answers
TEMPERATURE = 0.1
DO_SAMPLE = False


def load_test_dataset(test_path: str, num_samples: int = None) -> list:
    """Load test dataset from JSONL file."""
    logger.info(f"Loading test dataset from: {test_path}")
    
    queries = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    logger.info(f"Total test samples: {len(queries)}")
    
    if num_samples and num_samples < len(queries):
        queries = queries[:num_samples]
        logger.info(f"Using first {num_samples} samples for evaluation")
    
    return queries


def format_prompt(query: str) -> str:
    """Format query using Qwen chat template (same as training)."""
    return f"""<|im_start|>system
You are a highly knowledgeable maritime expert with extensive experience in maritime regulations, safety standards, vessel operations, and maritime law. 

When answering queries:
- Provide comprehensive, detailed explanations
- Include relevant regulations, standards, and references (IMO, SOLAS, MARPOL, etc.)
- Use proper formatting with headers, bullet points, and numbered lists where appropriate
- Include practical examples and real-world applications
- Explain technical terms and concepts thoroughly
- Cover all aspects of the question without omitting important details
- Structure your response logically with clear sections

Do NOT provide brief or summarized answers. Always aim for thorough, expert-level responses.<|im_end|>
<|im_start|>user
{query}

Please provide a detailed and comprehensive answer.<|im_end|>
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


def compute_metrics(generated: str, reference: str) -> dict:
    """Compute ROUGE and BLEU metrics."""
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)
    
    # BLEU score
    smoothing = SmoothingFunction().method1
    reference_tokens = reference.lower().split()
    generated_tokens = generated.lower().split()
    
    try:
        bleu = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
    except:
        bleu = 0.0
    
    return {
        'rouge1_f': rouge_scores['rouge1'].fmeasure,
        'rouge2_f': rouge_scores['rouge2'].fmeasure,
        'rougeL_f': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu,
    }



def evaluate_model(model, tokenizer, queries: list) -> list:
    """Evaluate model on all queries."""
    logger.info(f"Evaluating on {len(queries)} test samples...")
    
    model.eval()
    results = []
    
    for query_data in tqdm(queries, desc="Generating responses"):
        query = query_data["query"]
        reference = query_data["answer"]
        
        try:
            generated = generate_response(model, tokenizer, query)
            metrics = compute_metrics(generated, reference)
            
            result = {
                "query": query,
                "reference_answer": reference,
                "generated_answer": generated,
                "query_type": query_data.get("query_type", "unknown"),
                "difficulty": query_data.get("difficulty", "unknown"),
                "ref_length": len(reference),
                "gen_length": len(generated),
                **metrics
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            result = {
                "query": query,
                "reference_answer": reference,
                "generated_answer": f"[ERROR: {str(e)}]",
                "query_type": query_data.get("query_type", "unknown"),
                "difficulty": query_data.get("difficulty", "unknown"),
                "ref_length": len(reference),
                "gen_length": 0,
                "rouge1_f": 0.0,
                "rouge2_f": 0.0,
                "rougeL_f": 0.0,
                "bleu": 0.0,
            }
        
        results.append(result)
    
    return results



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SFT V2 model on test set"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to model output directory (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚢 MARITIME SFT V2 MODEL EVALUATION")
    print("=" * 80 + "\n")
    
    # Paths
    sft_model_path = os.path.join(args.model_dir, "final_model")
    test_dataset_path = os.path.join(args.model_dir, "test_dataset.jsonl")
    
    # Validate paths
    if not os.path.exists(sft_model_path):
        logger.error(f"❌ Model not found: {sft_model_path}")
        return
    
    if not os.path.exists(test_dataset_path):
        logger.error(f"❌ Test dataset not found: {test_dataset_path}")
        return
    
    logger.info("Configuration:")
    logger.info(f"  • Model directory: {args.model_dir}")
    logger.info(f"  • SFT model: {sft_model_path}")
    logger.info(f"  • Test dataset: {test_dataset_path}")
    
    # -------------------------------------------------------------------------
    # 1. Load Test Dataset
    # -------------------------------------------------------------------------
    logger.info("\n[1/4] Loading test dataset...")
    queries = load_test_dataset(test_dataset_path, args.num_samples)
    
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
    # 3. Load SFT Adapter and Evaluate
    # -------------------------------------------------------------------------
    logger.info("\n[3/4] Loading SFT adapter and evaluating...")
    
    sft_model = PeftModel.from_pretrained(base_model, sft_model_path)
    logger.info(f"✅ SFT adapter loaded from: {sft_model_path}")
    
    results = evaluate_model(sft_model, tokenizer, queries)
    
    # Clean up
    del sft_model
    del base_model
    torch.cuda.empty_cache()
    
    # -------------------------------------------------------------------------
    # 4. Save Results
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] Saving results...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.model_dir)
    csv_path = os.path.join(args.output_dir, f"eval_{model_name}_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Results saved: {csv_path}")
    
    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    
    logger.info(f"\nTotal queries evaluated: {len(results)}")
    
    # Overall metrics
    logger.info("\n📈 Overall Metrics:")
    logger.info(f"  • ROUGE-1 F1: {df['rouge1_f'].mean():.4f}")
    logger.info(f"  • ROUGE-2 F1: {df['rouge2_f'].mean():.4f}")
    logger.info(f"  • ROUGE-L F1: {df['rougeL_f'].mean():.4f}")
    logger.info(f"  • BLEU: {df['bleu'].mean():.4f}")
    
    # Response lengths
    logger.info("\n📏 Average Response Lengths:")
    logger.info(f"  • Reference: {df['ref_length'].mean():.0f} chars")
    logger.info(f"  • Generated: {df['gen_length'].mean():.0f} chars")
    
    # Metrics by query type
    if 'query_type' in df.columns:
        logger.info("\n📊 ROUGE-L by Query Type:")
        for qtype in df['query_type'].unique():
            subset = df[df['query_type'] == qtype]
            logger.info(f"  • {qtype}: {subset['rougeL_f'].mean():.4f} (n={len(subset)})")
    
    # Metrics by difficulty
    if 'difficulty' in df.columns:
        logger.info("\n📊 ROUGE-L by Difficulty:")
        for diff in df['difficulty'].unique():
            subset = df[df['difficulty'] == diff]
            logger.info(f"  • {diff}: {subset['rougeL_f'].mean():.4f} (n={len(subset)})")
    
    # Show sample comparisons
    print("\n" + "=" * 80)
    print("📝 SAMPLE COMPARISONS (First 3 queries)")
    print("=" * 80)
    
    for i in range(min(3, len(results))):
        print(f"\n{'─'*80}")
        print(f"Query {i+1}: {results[i]['query'][:100]}...")
        print(f"\n🎯 Reference Answer (first 300 chars):")
        print(f"  {results[i]['reference_answer'][:300]}...")
        print(f"\n🤖 Generated Answer (first 300 chars):")
        print(f"  {results[i]['generated_answer'][:300]}...")
        print(f"\n📈 Scores: ROUGE-L={results[i]['rougeL_f']:.3f}, BLEU={results[i]['bleu']:.3f}")
    
    # Save summary
    summary = {
        "model_dir": args.model_dir,
        "num_samples": len(results),
        "timestamp": timestamp,
        "metrics": {
            "rouge1_f_mean": float(df['rouge1_f'].mean()),
            "rouge2_f_mean": float(df['rouge2_f'].mean()),
            "rougeL_f_mean": float(df['rougeL_f'].mean()),
            "bleu_mean": float(df['bleu'].mean()),
        }
    }
    
    summary_path = os.path.join(args.output_dir, f"eval_summary_{model_name}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n✅ Summary saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print(f"Results saved to: {csv_path}")
    print("=" * 80 + "\n")
    
    return df


if __name__ == "__main__":
    main()
