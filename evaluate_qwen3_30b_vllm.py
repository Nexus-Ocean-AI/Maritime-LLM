"""
Evaluate Qwen3-30B SFT model using vLLM for fast batch inference.

vLLM provides much faster inference compared to HuggingFace transformers.
This script evaluates both base model and fine-tuned model with LoRA adapter.

Usage:
    python evaluate_qwen3_30b_vllm.py
    
    # Specify model directory
    python evaluate_qwen3_30b_vllm.py --model-dir outputs_qwen3_finetune_v2_20251218_093000
    
    # Limit samples for quick testing
    python evaluate_qwen3_30b_vllm.py --num-samples 50
    
    # Skip base model comparison
    python evaluate_qwen3_30b_vllm.py --no-compare-base
    
    # Multi-GPU inference
    python evaluate_qwen3_30b_vllm.py --tensor-parallel-size 2

Requirements:
    pip install vllm rouge-score nltk
"""

import os
import json
import argparse
import logging
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
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

# Qwen3-30B base model (same as used in training)
BASE_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct"

DEFAULT_MODEL_DIR = "outputs_qwen3_finetune_v2"

MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.1
TOP_P = 0.95


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
    """Format query using Qwen3 chat template (same as training)."""
    return """<|im_start|>system
You are a highly knowledgeable maritime expert. You are provided with a query related to maritime regulations, safety standards, or operational manuals. Answer the query accurately and comprehensively.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
""".format(query)


def compute_metrics(generated: str, reference: str) -> dict:
    """Compute ROUGE and BLEU metrics."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)
    
    smoothing = SmoothingFunction().method1
    reference_tokens = reference.lower().split()
    generated_tokens = generated.lower().split()
    
    try:
        bleu = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
    except Exception:
        bleu = 0.0
    
    return {
        'rouge1_f': rouge_scores['rouge1'].fmeasure,
        'rouge2_f': rouge_scores['rouge2'].fmeasure,
        'rougeL_f': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu,
    }


def extract_response_text(output) -> str:
    """Extract generated text from vLLM output."""
    if output.outputs:
        return output.outputs[0].text.strip()
    return ""


def find_model_dir(model_dir: str) -> str:
    """Find the correct model directory, handling timestamped directories."""
    if os.path.exists(model_dir):
        return model_dir
    
    # Try to find a directory starting with the given prefix
    parent_dir = os.path.dirname(model_dir) or "."
    base_name = os.path.basename(model_dir)
    
    if os.path.exists(parent_dir):
        matching_dirs = sorted([
            d for d in os.listdir(parent_dir)
            if d.startswith(base_name) and os.path.isdir(os.path.join(parent_dir, d))
        ])
        
        if matching_dirs:
            latest = os.path.join(parent_dir, matching_dirs[-1])
            logger.info(f"Found matching directory: {latest}")
            return latest
    
    return model_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-30B SFT model using vLLM")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Path to model output directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--no-compare-base", action="store_true",
                        help="Skip base model comparison")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                        help="GPU memory utilization for vLLM (default: 0.95 for 30B model)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length (default: 4096)")
    parser.add_argument("--max-lora-rank", type=int, default=64,
                        help="Maximum LoRA rank (default: 64)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_ID,
                        help=f"Base model ID (default: {BASE_MODEL_ID})")
    
    args = parser.parse_args()
    compare_base = not args.no_compare_base
    
    print("\n" + "=" * 80)
    print("🚢 QWEN3-30B MARITIME SFT EVALUATION (vLLM - Fast Batch Inference)")
    print("=" * 80 + "\n")
    
    # -------------------------------------------------------------------------
    # Validate Paths
    # -------------------------------------------------------------------------
    model_dir = find_model_dir(args.model_dir)
    sft_model_path = os.path.join(model_dir, "final_model")
    test_dataset_path = os.path.join(model_dir, "test_dataset.jsonl")
    
    if not os.path.exists(sft_model_path):
        logger.warning(f"final_model not found: {sft_model_path}")
        # Try to find any checkpoint
        if os.path.exists(model_dir):
            checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
                sft_model_path = os.path.join(model_dir, latest_checkpoint)
                logger.info(f"Using checkpoint instead: {sft_model_path}")
            else:
                logger.error("No checkpoints found")
                return
        else:
            logger.error(f"Model directory not found: {model_dir}")
            return
    
    if not os.path.exists(test_dataset_path):
        logger.error(f"❌ Test dataset not found: {test_dataset_path}")
        return
    
    logger.info("Configuration:")
    logger.info(f"  • Base model: {args.base_model}")
    logger.info(f"  • SFT adapter: {sft_model_path}")
    logger.info(f"  • Test dataset: {test_dataset_path}")
    logger.info(f"  • Compare with base: {compare_base}")
    logger.info(f"  • GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"  • Max model length: {args.max_model_len}")
    logger.info(f"  • Tensor parallel size: {args.tensor_parallel_size}")
    
    # -------------------------------------------------------------------------
    # [1/5] Load Test Dataset
    # -------------------------------------------------------------------------
    logger.info("\n[1/5] Loading test dataset...")
    queries = load_test_dataset(test_dataset_path, args.num_samples)
    prompts = [format_prompt(q["query"]) for q in queries]
    logger.info(f"Prepared {len(prompts)} prompts for batch inference")
    
    # -------------------------------------------------------------------------
    # [2/5] Initialize vLLM with LoRA Support
    # -------------------------------------------------------------------------
    logger.info("\n[2/5] Initializing vLLM engine with LoRA support...")
    logger.info("Note: Loading Qwen3-30B may take a few minutes...")
    
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    
    logger.info("✅ vLLM engine initialized successfully")
    
    # -------------------------------------------------------------------------
    # [3/5] Evaluate Base Model (if requested)
    # -------------------------------------------------------------------------
    base_results = {}
    if compare_base:
        logger.info("\n[3/5] Running batch inference with BASE model (no LoRA)...")
        
        base_outputs = llm.generate(prompts, sampling_params)
        
        logger.info(f"✅ Base model generated {len(base_outputs)} responses")
        
        for i, output in enumerate(base_outputs):
            generated_text = extract_response_text(output)
            reference = queries[i]["answer"]
            metrics = compute_metrics(generated_text, reference)
            
            base_results[i] = {
                "base_answer": generated_text,
                "base_length": len(generated_text),
                "base_rouge1_f": metrics['rouge1_f'],
                "base_rouge2_f": metrics['rouge2_f'],
                "base_rougeL_f": metrics['rougeL_f'],
                "base_bleu": metrics['bleu'],
            }
    else:
        logger.info("\n[3/5] Skipping base model evaluation (--no-compare-base)")
    
    # -------------------------------------------------------------------------
    # [4/5] Evaluate SFT Model with LoRA Adapter
    # -------------------------------------------------------------------------
    logger.info("\n[4/5] Running batch inference with SFT model (LoRA adapter)...")
    
    # Create LoRA request for the fine-tuned adapter
    lora_request = LoRARequest(
        lora_name="maritime_qwen3_sft",
        lora_int_id=1,
        lora_path=sft_model_path,
    )
    
    sft_outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )
    
    logger.info(f"✅ SFT model generated {len(sft_outputs)} responses")
    
    # Process SFT results
    all_results = []
    for i, output in enumerate(sft_outputs):
        generated_text = extract_response_text(output)
        reference = queries[i]["answer"]
        query_data = queries[i]
        metrics = compute_metrics(generated_text, reference)
        
        result = {
            "query": query_data["query"],
            "reference_answer": reference,
            "sft_answer": generated_text,
            "query_type": query_data.get("query_type", "unknown"),
            "difficulty": query_data.get("difficulty", "unknown"),
            "ref_length": len(reference),
            "sft_length": len(generated_text),
            "sft_rouge1_f": metrics['rouge1_f'],
            "sft_rouge2_f": metrics['rouge2_f'],
            "sft_rougeL_f": metrics['rougeL_f'],
            "sft_bleu": metrics['bleu'],
        }
        
        # Add base results if available
        if i in base_results:
            result.update(base_results[i])
        
        all_results.append(result)
    
    # -------------------------------------------------------------------------
    # [5/5] Save Results
    # -------------------------------------------------------------------------
    logger.info("\n[5/5] Saving results...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_dir)
    csv_path = os.path.join(args.output_dir, f"eval_qwen3_30b_vllm_{model_name}_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Results saved: {csv_path}")
    
    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY - BASE vs FINE-TUNED (Qwen3-30B)")
    print("=" * 80)
    
    logger.info(f"\nTotal queries evaluated: {len(all_results)}")
    
    # Compute aggregate metrics
    sft_rouge1 = df['sft_rouge1_f'].mean()
    sft_rouge2 = df['sft_rouge2_f'].mean()
    sft_rougeL = df['sft_rougeL_f'].mean()
    sft_bleu = df['sft_bleu'].mean()
    sft_len = df['sft_length'].mean()
    ref_len = df['ref_length'].mean()
    
    print("\n" + "-" * 80)
    print("📈 METRICS COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Base Model':>15} {'SFT Model':>15} {'Improvement':>15}")
    print("-" * 80)
    
    if compare_base:
        base_rouge1 = df['base_rouge1_f'].mean()
        base_rouge2 = df['base_rouge2_f'].mean()
        base_rougeL = df['base_rougeL_f'].mean()
        base_bleu = df['base_bleu'].mean()
        base_len = df['base_length'].mean()
        
        print(f"{'ROUGE-1 F1':<20} {base_rouge1:>15.4f} {sft_rouge1:>15.4f} {sft_rouge1-base_rouge1:>+15.4f}")
        print(f"{'ROUGE-2 F1':<20} {base_rouge2:>15.4f} {sft_rouge2:>15.4f} {sft_rouge2-base_rouge2:>+15.4f}")
        print(f"{'ROUGE-L F1':<20} {base_rougeL:>15.4f} {sft_rougeL:>15.4f} {sft_rougeL-base_rougeL:>+15.4f}")
        print(f"{'BLEU':<20} {base_bleu:>15.4f} {sft_bleu:>15.4f} {sft_bleu-base_bleu:>+15.4f}")
        print("-" * 80)
        print(f"{'Avg Length':<20} {base_len:>15.0f} {sft_len:>15.0f} {sft_len-base_len:>+15.0f}")
        print(f"{'Reference Length':<20} {ref_len:>15.0f}")
    else:
        base_rouge1 = base_rouge2 = base_rougeL = base_bleu = base_len = 0
        print(f"{'ROUGE-1 F1':<20} {'N/A':>15} {sft_rouge1:>15.4f}")
        print(f"{'ROUGE-2 F1':<20} {'N/A':>15} {sft_rouge2:>15.4f}")
        print(f"{'ROUGE-L F1':<20} {'N/A':>15} {sft_rougeL:>15.4f}")
        print(f"{'BLEU':<20} {'N/A':>15} {sft_bleu:>15.4f}")
        print("-" * 80)
        print(f"{'Avg Gen Length':<20} {sft_len:>15.0f}")
        print(f"{'Reference Length':<20} {ref_len:>15.0f}")
    
    # Show sample comparisons
    print("\n" + "=" * 80)
    print("📝 SAMPLE COMPARISONS (First 3 queries)")
    print("=" * 80)
    
    for i in range(min(3, len(all_results))):
        result = all_results[i]
        print(f"\n{'─'*80}")
        query_preview = result['query'][:100] + "..." if len(result['query']) > 100 else result['query']
        print(f"Query {i+1}: {query_preview}")
        print(f"\n🎯 Reference Answer (first 300 chars):")
        ref_preview = result['reference_answer'][:300] + "..." if len(result['reference_answer']) > 300 else result['reference_answer']
        print(f"  {ref_preview}")
        
        if compare_base and 'base_answer' in result:
            print(f"\n🔵 Base Model (first 300 chars) [ROUGE-L={result['base_rougeL_f']:.3f}]:")
            base_preview = result['base_answer'][:300] + "..." if len(result['base_answer']) > 300 else result['base_answer']
            print(f"  {base_preview}")
        
        print(f"\n🟢 SFT Model (first 300 chars) [ROUGE-L={result['sft_rougeL_f']:.3f}]:")
        sft_preview = result['sft_answer'][:300] + "..." if len(result['sft_answer']) > 300 else result['sft_answer']
        print(f"  {sft_preview}")
    
    # Save summary JSON
    summary = {
        "model_dir": model_dir,
        "num_samples": len(all_results),
        "timestamp": timestamp,
        "base_model": args.base_model,
        "inference_engine": "vLLM",
        "model_type": "Qwen3-30B",
        "sft_metrics": {
            "rouge1_f_mean": float(sft_rouge1),
            "rouge2_f_mean": float(sft_rouge2),
            "rougeL_f_mean": float(sft_rougeL),
            "bleu_mean": float(sft_bleu),
            "avg_length": float(sft_len),
        }
    }
    
    if compare_base:
        summary["base_metrics"] = {
            "rouge1_f_mean": float(base_rouge1),
            "rouge2_f_mean": float(base_rouge2),
            "rougeL_f_mean": float(base_rougeL),
            "bleu_mean": float(base_bleu),
            "avg_length": float(base_len),
        }
        summary["improvement"] = {
            "rouge1_f": float(sft_rouge1 - base_rouge1),
            "rouge2_f": float(sft_rouge2 - base_rouge2),
            "rougeL_f": float(sft_rougeL - base_rougeL),
            "bleu": float(sft_bleu - base_bleu),
        }
    
    summary_path = os.path.join(args.output_dir, f"eval_qwen3_30b_summary_{model_name}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n✅ Summary saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print(f"Results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 80 + "\n")
    
    return df


if __name__ == "__main__":
    main()
