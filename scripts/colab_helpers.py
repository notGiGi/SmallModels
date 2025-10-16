import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lm_eval_wrapper import ModelEvaluator, LMEvalWrapper


def evaluate_and_save(model_key, batch_size, tasks=None, output_dir="results"):
    """
    Evaluate a single model and save/download results.
    
    Args:
        model_key: Model identifier (smollm, tinyllama, qwen, gemma2b, phi3)
        batch_size: Batch size for evaluation
        tasks: List of tasks (default: all safe tasks without code execution)
        output_dir: Output directory for results
    
    Returns:
        dict: Evaluation results
    """
    # Default tasks (safe, no code execution)
    if tasks is None:
        tasks = [
            "boolq",
            "hellaswag", 
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "piqa",
            "gsm8k"
        ]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_key}")
    print(f"Batch size: {batch_size}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"{'='*60}\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(device="cuda", batch_size=batch_size)
    
    # Start timer
    start_time = datetime.now()
    
    # Evaluate
    try:
        results = evaluator.evaluate_model(
            model_key=model_key,
            tasks=tasks,
            limit=None  # Full evaluation
        )
        
        elapsed = datetime.now() - start_time
        
        # Print results
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        
        for task in tasks:
            task_results = results["results"].get(task, {})
            
            # Get accuracy (different tasks have different metric names)
            acc = (task_results.get("acc,none") or 
                   task_results.get("acc_norm,none") or
                   task_results.get("pass@1") or
                   task_results.get("exact_match,none") or
                   0)
            
            print(f"  {task:20s}: {acc:.2%}")
        
        print(f"{'='*60}")
        print(f"⏱️  Time elapsed: {elapsed}")
        print(f"{'='*60}\n")
        
        # Save results
        filename = f"{output_dir}/{model_key}_full.json"
        wrapper = LMEvalWrapper("dummy", "cuda")
        wrapper.save_results(results, filename)
        
        print(f"💾 Saved: {filename}")
        
        try:
            from google.colab import files
            print(f"📥 Downloading {filename}...")
            files.download(filename)
        except ImportError:
            # Not in Colab, skip download
            pass
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def get_model_config():
    """
    Get recommended batch sizes for each model.
    
    Returns:
        dict: Model configurations
    """
    return {
        "smollm": {
            "batch_size": 16,
            "params": "360M",
            "estimated_time": "25-30 min"
        },
        "tinyllama": {
            "batch_size": 8,
            "params": "1.1B",
            "estimated_time": "45-60 min"
        },
        "qwen": {
            "batch_size": 4,
            "params": "1.5B",
            "estimated_time": "90-120 min"
        },
        "gemma2b": {
            "batch_size": 4,
            "params": "2.5B",
            "estimated_time": "120-150 min"
        },
        "phi3": {
            "batch_size": 2,
            "params": "3.8B",
            "estimated_time": "150-180 min"
        }
    }


def print_model_info():
    """Print information about available models."""
    configs = get_model_config()
    
    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)
    
    for model, config in configs.items():
        print(f"\n{model.upper()}")
        print(f"  Parameters: {config['params']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Est. time:  {config['estimated_time']}")
    
    print("="*60 + "\n")


def evaluate_model_by_name(model_name):
    """
    Convenience function to evaluate a model with recommended settings.
    
    Args:
        model_name: Model name (smollm, tinyllama, qwen, gemma2b, phi3)
    
    Returns:
        dict: Evaluation results
    """
    configs = get_model_config()
    
    if model_name not in configs:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available: {list(configs.keys())}")
        return None
    
    config = configs[model_name]
    batch_size = config["batch_size"]
    
    print(f"📊 Evaluating {model_name.upper()} ({config['params']})")
    print(f"⏱️  Estimated time: {config['estimated_time']}")
    
    return evaluate_and_save(model_name, batch_size)