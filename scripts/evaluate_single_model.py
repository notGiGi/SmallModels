"""
Evaluate a single model using lm-evaluation-harness.
"""

import argparse
import sys
import warnings
import os
from pathlib import Path


warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lm_eval_wrapper import ModelEvaluator, LMEvalWrapper
from src.utils import setup_logging, get_timestamp
import logging

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
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    
    args = parser.parse_args()
    
  
    if args.verbose:
        setup_logging(level=logging.INFO)
    else:
        setup_logging(level=logging.WARNING)
    
    # Clean header
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.limit if args.limit else 'all'}")
    print("="*60)
    
    # Create evaluator
    evaluator = ModelEvaluator(device=args.device, batch_size=args.batch_size)
    
    # Run evaluation
    print("\nStarting evaluation...")
   
    results = evaluator.evaluate_model(
        model_key=args.model,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit
    )
    


    print("RESULTS")

    
    for task in args.tasks:
        task_results = results["results"].get(task, {})
        acc = task_results.get("acc,none", task_results.get("acc_norm,none", 0))
        print(f"{task.upper():20s}: {acc:.2%} ({acc*100:.1f}%)")
    
    print("="*60)
    
    # Save results
    timestamp = get_timestamp()
    output_path = Path(args.output_dir) / f"{args.model}_{timestamp}.json"
    
    wrapper = LMEvalWrapper(evaluator.MODELS[args.model], args.device)
    wrapper.save_results(results, output_path)
    
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
