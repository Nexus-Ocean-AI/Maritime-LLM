"""
Evaluate SFT V2 model using vLLM for fast batch inference.

vLLM provides much faster inference compared to HuggingFace transformers.
This script evaluates both base model and fine-tuned model with LoRA adapter.

Usage:
    python evaluate_sft_v2_vllm.py
    
    # Specify model directory
    python evaluate_sft_v2_vllm.py --model-dir outputs_qwen2.5_7b_sft_v2_20251217_105003
    
    # Limit samples for quick testing
    python evaluate_sft_v2_vllm.py --num-samples 50
    
    # Skip base model comparison
    python evaluate_sft_v2_vllm.py --no-compare-base

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

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_DIR = "outputs_qwen2.5_7b_sft_v2_20251217_105003"

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
    """Format query using Qwen chat template."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT V2 model using vLLM")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Path to model output directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--no-compare-base", action="store_true",
                        help="Skip base model comparison")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM (default: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Maximum model context length (default: 8192)")
    parser.add_argument("--max-lora-rank", type=int, default=64,
                        help="Maximum LoRA rank (default: 64)")
    
    args = parser.parse_args()
    compare_base = not args.no_compare_base
    
    print("\n" + "=" * 80)
    print("🚢 MARITIME SFT V2 MODEL EVALUATION (vLLM - Fast Batch Inference)")
    print("=" * 80 + "\n")
    
    # -------------------------------------------------------------------------
    # Validate Paths
    # -------------------------------------------------------------------------
    sft_model_path = os.path.join(args.model_dir, "final_model")
    test_dataset_path = os.path.join(args.model_dir, "test_dataset.jsonl")
    
    if not os.path.exists(sft_model_path):
        logger.error(f"❌ Model not found: {sft_model_path}")
        return
    
    if not os.path.exists(test_dataset_path):
        logger.error(f"❌ Test dataset not found: {test_dataset_path}")
        return
    
    logger.info("Configuration:")
    logger.info(f"  • Base model: {BASE_MODEL_ID}")
    logger.info(f"  • SFT adapter: {sft_model_path}")
    logger.info(f"  • Test dataset: {test_dataset_path}")
    logger.info(f"  • Compare with base: {compare_base}")
    logger.info(f"  • GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"  • Max model length: {args.max_model_len}")
    
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
    
    llm = LLM(
        model=BASE_MODEL_ID,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
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
        lora_name="maritime_sft",
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
    model_name = os.path.basename(args.model_dir)
    csv_path = os.path.join(args.output_dir, f"eval_vllm_{model_name}_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Results saved: {csv_path}")
    
    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY - BASE vs FINE-TUNED (vLLM)")
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
        print(f"Query {i+1}: {result['query'][:100]}...")
        print(f"\n🎯 Reference Answer (first 300 chars):")
        print(f"  {result['reference_answer'][:300]}...")
        
        if compare_base and 'base_answer' in result:
            print(f"\n🔵 Base Model (first 300 chars) [ROUGE-L={result['base_rougeL_f']:.3f}]:")
            print(f"  {result['base_answer'][:300]}...")
        
        print(f"\n🟢 SFT Model (first 300 chars) [ROUGE-L={result['sft_rougeL_f']:.3f}]:")
        print(f"  {result['sft_answer'][:300]}...")
    
    # Save summary JSON
    summary = {
        "model_dir": args.model_dir,
        "num_samples": len(all_results),
        "timestamp": timestamp,
        "base_model": BASE_MODEL_ID,
        "inference_engine": "vLLM",
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
    
    summary_path = os.path.join(args.output_dir, f"eval_vllm_summary_{model_name}_{timestamp}.json")
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
