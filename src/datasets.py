from datasets import load_dataset
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:

    DATASETS = {
        "boolq": {
            "name": "google/boolq",
            "split": "validation",
            "description": "Yes/No questions"
        },
        "triviaqa": {
            "name": "trivia_qa",
            "split": "validation",
            "subset": "unfiltered",
            "description": "Factual knowledge questions"
        },
        "hellaswag": {
            "name": "Rowan/hellaswag",
            "split": "validation",
            "description": "Common sense reasoning"
        },
        "gsm8k": {
            "name": "openai/gsm8k",
            "split": "test",
            "subset": "main",
            "description": "Grade school math"
        },
        "humaneval": {
            "name": "openai/openai_humaneval",
            "split": "test",
            "description": "Code generation"
        }
    }
    
    def load_dataset(
        self, 
        task: str, 
        n_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> List[Dict]:
        """
        Load dataset for a specific task.
        
        Args:
            task: Task name (e.g., "boolq")
            n_samples: Number of samples to load (None = all)
            shuffle: Whether to shuffle dataset
            seed: Random seed for shuffling
            
        Returns:
            List of dataset examples
        """
        if task not in self.DATASETS:
            raise ValueError(f"Unknown task: {task}. Choose from: {list(self.DATASETS.keys())}")
            
        config = self.DATASETS[task]
        logger.info(f"Loading {task}: {config['description']}")
        
       
        dataset_name = config["name"]
        split = config["split"]
        subset = config.get("subset")
        
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
            
        
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
            
        
        if n_samples is not None:
            dataset = dataset.select(range(min(n_samples, len(dataset))))
            
        logger.info(f"  Loaded {len(dataset)} samples")
        
        
        examples = [dict(example) for example in dataset]
        return examples
    
    def format_prompt(self, task: str, example: Dict) -> str:
        """
        Format example into prompt for model.
        
        Args:
            task: Task name
            example: Dataset example
            
        Returns:
            Formatted prompt string
        """
        if task == "boolq":
            return self._format_boolq(example)
        elif task == "triviaqa":
            return self._format_triviaqa(example)
        elif task == "hellaswag":
            return self._format_hellaswag(example)
        elif task == "gsm8k":
            return self._format_gsm8k(example)
        elif task == "humaneval":
            return self._format_humaneval(example)
        else:
            raise ValueError(f"No formatter for task: {task}")
    
    def _format_boolq(self, example: Dict) -> str:
        """Format BoolQ example."""
        passage = example["passage"]
        question = example["question"]
        return f"Passage: {passage}\n\nQuestion: {question}\nAnswer (Yes or No):"
    
    def _format_triviaqa(self, example: Dict) -> str:
        """Format TriviaQA example."""
        question = example["question"]
        return f"Question: {question}\nAnswer:"
    
    def _format_hellaswag(self, example: Dict) -> str:
        """Format HellaSwag example."""
        ctx = example["ctx"]
        endings = example["endings"]
        prompt = f"Complete the following:\n{ctx}\n\nChoices:\n"
        for i, ending in enumerate(endings):
            prompt += f"{i+1}. {ending}\n"
        prompt += "\nBest completion (1-4):"
        return prompt
    
    def _format_gsm8k(self, example: Dict) -> str:
        """Format GSM8K example."""
        question = example["question"]
        return f"Question: {question}\nLet's solve this step by step.\nAnswer:"
    
    def _format_humaneval(self, example: Dict) -> str:
        """Format HumanEval example."""
        prompt = example["prompt"]
        return f"{prompt}\n# Complete the function"