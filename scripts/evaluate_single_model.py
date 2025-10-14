"""
Evaluate a single model using lm-evaluation-harness.

Usage:
    python scripts/evaluate_single_model.py --model smollm --tasks boolq --limit 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lm_eval_wrapper import ModelEvaluator, LMEvalWrapper
from src.utils import setup_logging, get_timestamp

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single model")
    parser.add_argument("--model", required=True, 
                       choices=["phi3", "gemma2b", "qwen", "tinyllama", "smollm"])
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output_dir", default="results/raw")
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("\n" + "="*60)
    print("MODEL EVALUATION WITH LM-EVAL-HARNESS")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Limit: {args.limit}")
    print("="*60 + "\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(device=args.device, batch_size=args.batch_size)
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.evaluate_model(
        model_key=args.model,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit
    )
    
    # Print summary
    wrapper = LMEvalWrapper(evaluator.MODELS[args.model], args.device)
    wrapper.print_summary(results)
    
    # Save results
    timestamp = get_timestamp()
    output_path = Path(args.output_dir) / f"{args.model}_{timestamp}.json"
    wrapper.save_results(results, output_path)
    
    print(f"\n✓ Results saved to: {output_path}")

if __name__ == "__main__":
    main()