import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


class LMEvalWrapper:
    """
    Wrapper for lm-evaluation-harness.
    Provides a clean interface for our research project.
    """
    
    TASK_MAPPING = {
        "boolq": "boolq",
        "triviaqa": "triviaqa",
        "hellaswag": "hellaswag",
        "gsm8k": "gsm8k",
        "humaneval": "humaneval",
        # Additional useful tasks
        "arc_easy": "arc_easy",
        "arc_challenge": "arc_challenge",
        "winogrande": "winogrande",
        "piqa": "piqa",
        "mmlu": "mmlu"
    }
    
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 8):
        """
        Initialize evaluator.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ("cuda" or "cpu")
            batch_size: Batch size for evaluation
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Initializing LMEval for {model_name}")
        logger.info(f"Device: {device}, Batch size: {batch_size}")
    
    def evaluate(
        self,
        tasks: List[str],
        num_fewshot: int = 0,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Evaluate model on specified tasks.
        
        Args:
            tasks: List of task names (use our naming convention)
            num_fewshot: Number of few-shot examples (0 = zero-shot)
            limit: Limit number of examples per task (None = all)
            
        Returns:
            Dictionary with results
        """
        
        eval_tasks = [self.TASK_MAPPING.get(task, task) for task in tasks]
        
        logger.info(f"Evaluating on tasks: {eval_tasks}")
        logger.info(f"Few-shot: {num_fewshot}, Limit: {limit}")
        
        try:
           
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={self.model_name},dtype=auto",
                tasks=eval_tasks,
                num_fewshot=num_fewshot,
                batch_size=self.batch_size,
                device=self.device,
                limit=limit,
                write_out=False  # Don't write intermediate files
            )
            
            logger.info("Evaluation complete")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def evaluate_single_task(
        self,
        task: str,
        num_fewshot: int = 0,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Evaluate on a single task.
        
        Args:
            task: Task name
            num_fewshot: Number of few-shot examples
            limit: Limit number of examples
            
        Returns:
            Results dictionary for the task
        """
        results = self.evaluate(
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit
        )
        
        
        eval_task = self.TASK_MAPPING.get(task, task)
        return results["results"].get(eval_task, {})
    
    def format_results(self, results: Dict) -> Dict[str, Dict]:
        """
        Format results into a cleaner structure.
        
        Args:
            results: Raw results from lm-eval
            
        Returns:
            Formatted results
        """
        formatted = {}
        
        for task, metrics in results["results"].items():
            formatted[task] = {
                "accuracy": metrics.get("acc,none", metrics.get("acc_norm,none", 0)),
                "all_metrics": metrics
            }
        
        return formatted
    def save_results(self, results: Dict, output_path: str):
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert non-JSON-serializable objects
        def make_serializable(obj):
            """Convert numpy/torch types to Python types."""
            import numpy as np
            
            # Handle numpy types
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            
            # Handle dtype objects (the problematic one)
            elif hasattr(obj, 'dtype') and isinstance(obj.dtype, np.dtype):
                return str(obj)
            
            # Handle torch tensors
            elif hasattr(obj, 'item'):
                try:
                    return obj.item()
                except:
                    return str(obj)
            
            # Recursively handle containers
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            
            # Handle other types
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Last resort: convert to string
                return str(obj)
        
        # Clean results
        clean_results = make_serializable(results)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """
        Print a summary of results.
        
        Args:
            results: Results dictionary
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        formatted = self.format_results(results)
        
        for task, metrics in formatted.items():
            acc = metrics["accuracy"]
            print(f"\n{task.upper()}")
            print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        print("\n" + "="*60)


class ModelEvaluator:
    """
    High-level interface for evaluating our 5 models.
    """
    
    MODELS = {
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "gemma2b": "google/gemma-2b-it",
        "qwen": "Qwen/Qwen2-1.5B-Instruct",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "smollm": "HuggingFaceTB/SmolLM-360M-Instruct"
    }
    
    def __init__(self, device: str = "cuda", batch_size: int = 8):
        """
        Initialize model evaluator.
        
        Args:
            device: Device to use
            batch_size: Batch size for evaluation
        """
        self.device = device
        self.batch_size = batch_size
    
    def evaluate_model(self, model_key: str, tasks: List[str], 
                   limit: Optional[int] = None, log_samples: bool = True):
        """
        Evaluate model on specified tasks.
        
        Args:
            model_key: Model identifier (smollm, tinyllama, qwen, etc.)
            tasks: List of task names
            limit: Limit number of samples (None = all)
            log_samples: Whether to save individual samples (for taxonomy)
        
        Returns:
            dict: Evaluation results with samples
        """
        import lm_eval
        
        # Get model path from model_key
        model_configs = {
            'smollm': 'HuggingFaceTB/SmolLM-360M-Instruct',
            'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'qwen': 'Qwen/Qwen-1_5B',
            'gemma2b': 'google/gemma-2b',
            'phi3': 'microsoft/Phi-3-mini-4k-instruct'
        }
        
        if model_key not in model_configs:
            raise ValueError(f"Unknown model_key: {model_key}. Available: {list(model_configs.keys())}")
        
        model_path = model_configs[model_key]
        
        print(f"🔬 Evaluating {model_key} ({model_path}) on {len(tasks)} tasks...")
        if log_samples:
            print("📝 Sample logging ENABLED (for taxonomy analysis)")
        
        results = lm_eval.simple_evaluate(
            model='hf',
            model_args=f'pretrained={model_path},dtype=auto',
            tasks=tasks,
            num_fewshot=0,
            batch_size=self.batch_size,
            device=self.device,
            limit=limit,
            log_samples=log_samples,
        )
        
        # Verify samples were saved
        if log_samples:
            if 'samples' in results and results['samples']:
                n_samples = sum(len(results['samples'].get(task, [])) for task in tasks)
                print(f"✅ Captured {n_samples} samples for taxonomy")
            else:
                print(f"⚠️  WARNING: log_samples=True but no samples in results!")
        
        return results
    
    def evaluate_all_models(
        self,
        tasks: List[str],
        num_fewshot: int = 0,
        limit: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Evaluate all 5 models.
        
        Args:
            tasks: List of tasks
            num_fewshot: Few-shot examples
            limit: Limit examples per task
            save_dir: Directory to save results (optional)
            
        Returns:
            Dictionary mapping model_key -> results
        """
        all_results = {}
        
        for model_key in self.MODELS.keys():
            try:
                results = self.evaluate_model(
                    model_key=model_key,
                    tasks=tasks,
                    num_fewshot=num_fewshot,
                    limit=limit
                )
                
                all_results[model_key] = results
                
                
                if save_dir:
                    save_path = Path(save_dir) / f"{model_key}_results.json"
                    evaluator = LMEvalWrapper(self.MODELS[model_key], self.device)
                    evaluator.save_results(results, save_path)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_key}: {e}")
                all_results[model_key] = {"error": str(e)}
        
        return all_results


def get_all_tasks():
    """
    Get all available tasks for evaluation.
    
    Returns:
        list: List of all task names
    """
    return [
        "boolq",
        "hellaswag", 
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "piqa",
        "gsm8k"   
    ]


def get_task_groups():
    """
    Get tasks organized by type.
    
    Returns:
        dict: Dictionary of task groups
    """
    return {
        "reasoning": ["boolq", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"],
        "math": ["gsm8k"],
        "all": get_all_tasks() 
    }