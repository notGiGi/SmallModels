"""
Helper functions for Kaggle evaluation.
Optimized for speed, reliability, and taxonomy data collection.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lm_eval_wrapper import ModelEvaluator, LMEvalWrapper


def get_model_config():
    """
    Get optimized configurations for each model.
    
    Returns:
        dict: Model configurations with optimized batch sizes
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
            "estimated_time": "60-90 min"
        },
        "qwen": {
            "batch_size": 8,  # Optimized: was 4
            "params": "1.5B",
            "estimated_time": "180-210 min"  # ~3h with GSM8K
        },
        "gemma2b": {
            "batch_size": 8,  # Optimized: was 4
            "params": "2.5B",
            "estimated_time": "150-180 min"  # ~2.5-3h
        },
        "phi3": {
            "batch_size": 4,  # Keep at 4 (large model)
            "params": "3.8B",
            "estimated_time": "210-240 min"  # ~3.5-4h
        }
    }


def get_all_tasks():
    """
    Get all evaluation tasks INCLUDING GSM8K.
    
    GSM8K is included because:
    - Gemma-2B shows emergence (~20% expected)
    - Phi-3 shows exceptional performance (~85% expected)
    - Critical for identifying capability thresholds
    
    Returns:
        list: All task names
    """
    return [
        "boolq",
        "hellaswag", 
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "piqa",
        "gsm8k"  # Included for emergence analysis
    ]


def evaluate_and_save(model_key, batch_size, tasks=None, output_dir="results"):
    """
    Evaluate a model with optimized saving for Kaggle.
    
    Saves in multiple locations to prevent data loss:
    1. /kaggle/working/ (visible in Output panel)
    2. results/ subdirectory (backup)
    3. CSV format (easy recovery)
    4. Manual print (copy-paste backup)
    
    Args:
        model_key: Model identifier
        batch_size: Batch size for evaluation
        tasks: List of tasks (default: all tasks)
        output_dir: Output directory
    
    Returns:
        dict: Evaluation results
    """
    if tasks is None:
        tasks = get_all_tasks()
    
    config = get_model_config()[model_key]
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluating {model_key.upper()} ({config['params']})")
    print(f"⏱️  Estimated time: {config['estimated_time']}")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"{'='*60}\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(device="cuda", batch_size=batch_size)
    
    # Start timer
    start_time = datetime.now()
    
    try:
        # Evaluate with samples for taxonomy
        print("🔬 Running evaluation (with sample logging for taxonomy)...\n")
        results = evaluator.evaluate_model(
            model_key=model_key,
            tasks=tasks,
            limit=None,
            log_samples=True  # CRITICAL: Enable samples for taxonomy
        )
        
        elapsed = datetime.now() - start_time
        
        # ===== MULTIPLE SAVE STRATEGIES =====
        
        # 1. Save to /kaggle/working (visible in Output panel)
        kaggle_path = f"/kaggle/working/{model_key}_full.json"
        save_json(results, kaggle_path)
        print(f"💾 Saved: {kaggle_path}")
        
        # 2. Save to results/ (backup)
        os.makedirs(output_dir, exist_ok=True)
        backup_path = f"{output_dir}/{model_key}_full.json"
        save_json(results, backup_path)
        print(f"💾 Backup: {backup_path}")
        
        # 3. Save CSV (easy to recover)
        csv_path = f"/kaggle/working/{model_key}_results.csv"
        save_as_csv(results, tasks, csv_path)
        print(f"📊 CSV: {csv_path}")
        
        # 4. Check if samples were saved
        if 'samples' in results and results['samples']:
            n_samples = sum(len(results['samples'].get(task, [])) for task in tasks)
            print(f"✅ Samples saved: {n_samples} total across tasks")
            
            # Save samples separately (easier to load)
            samples_path = f"/kaggle/working/{model_key}_samples.json"
            save_json(results['samples'], samples_path)
            print(f"💾 Samples: {samples_path}")
        else:
            print(f"⚠️  WARNING: No samples in results! Check log_samples setting.")
        
        # ===== PRINT RESULTS =====
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        
        for task in tasks:
            task_results = results["results"].get(task, {})
            
            # Get accuracy metric
            acc = (task_results.get("acc,none") or 
                   task_results.get("acc_norm,none") or
                   task_results.get("exact_match,none") or
                   task_results.get("pass@1") or 0)
            
            print(f"  {task:20s}: {acc:.2%}")
        
        print(f"{'='*60}")
        print(f"⏱️  Time elapsed: {elapsed}")
        print(f"{'='*60}\n")
        
        # ===== MANUAL BACKUP (copy-paste recovery) =====
        
        print(f"\n{'='*60}")
        print("MANUAL BACKUP - Copy these if needed:")
        print(f"{'='*60}")
        for task in tasks:
            task_results = results["results"].get(task, {})
            acc = (task_results.get("acc,none") or 
                   task_results.get("acc_norm,none") or
                   task_results.get("exact_match,none") or 0)
            print(f'"{task}": {acc:.4f},')
        print(f"{'='*60}\n")
        
        print("✅ Evaluation complete!")
        print("📥 Download files from Output panel →")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def save_json(data, filepath):
    """Save data as JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_as_csv(results, tasks, filepath):
    """Save results in CSV format."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Metric', 'Value'])
        
        for task in tasks:
            task_results = results["results"].get(task, {})
            for metric, value in task_results.items():
                if not metric.endswith('_stderr'):
                    writer.writerow([task, metric, value])


def evaluate_model_by_name(model_name):
    """
    Convenience function to evaluate with recommended settings.
    
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
    
    return evaluate_and_save(model_name, batch_size)


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
    
    print("\n" + "="*60)
    print("All models include GSM8K for emergence analysis")
    print("="*60 + "\n")