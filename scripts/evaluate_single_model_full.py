"""
Evaluate a single model on all tasks - optimized for Colab sessions.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lm_eval_wrapper import ModelEvaluator, LMEvalWrapper, get_all_tasks
from src.utils import setup_logging
import logging

def main():
    parser = argparse.ArgumentParser(description="Evaluate single model - full evaluation")
    parser.add_argument("--model", required=True, 
                       choices=["phi3", "gemma2b", "qwen", "tinyllama", "smollm"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=None, 
                       help="Batch size (auto if not specified)")
    parser.add_argument("--output_dir", default="results/full")
    parser.add_argument("--tasks", nargs="+", default=None,
                       help="Tasks to run (default: all)")
    
    args = parser.parse_args()
    
    setup_logging(level=logging.WARNING)
    
    # Auto batch size based on model
    if args.batch_size is None:
        batch_sizes = {
            "smollm": 16,
            "tinyllama": 8,
            "qwen": 4,
            "gemma2b": 4,
            "phi3": 2
        }
        args.batch_size = batch_sizes.get(args.model, 4)
    
    # Tasks
    tasks = args.tasks if args.tasks else get_all_tasks()
    
    print("\n" + "="*60)
    print("🔬 FULL MODEL EVALUATION")
    print("="*60)
    print(f"📦 Model: {args.model}")
    print(f"📋 Tasks: {', '.join(tasks)}")
    print(f"💻 Device: {args.device}")
    print(f"📊 Batch size: {args.batch_size}")
    print("="*60 + "\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(device=args.device, batch_size=args.batch_size)
    
    start_time = datetime.now()
    print(f"⏱️  Started: {start_time.strftime('%H:%M:%S')}\n")
    
    # Evaluate
    try:
        results = evaluator.evaluate_model(
            model_key=args.model,
            tasks=tasks,
            limit=None  
        )
        
        elapsed = datetime.now() - start_time
        

        print("\n" + "="*60)
        print("✅ RESULTS")
        print("="*60)
        
        for task in tasks:
            task_results = results["results"].get(task, {})
            
            
            acc = (task_results.get("acc,none") or 
                   task_results.get("acc_norm,none") or
                   task_results.get("pass@1") or
                   task_results.get("exact_match,none") or
                   0)
            
            print(f"{task:20s}: {acc:.2%}")
        
        print("="*60)
        print(f"⏱️  Time elapsed: {elapsed}")
        print("="*60 + "\n")
        
        # Save
        output_path = Path(args.output_dir) / f"{args.model}_full.json"
        wrapper = LMEvalWrapper(evaluator.MODELS[args.model], args.device)
        wrapper.save_results(results, output_path)
        
        print(f"💾 Saved: {output_path}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())