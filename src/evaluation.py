import torch
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate models on various tasks."""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize evaluator.
        
        Args:
            model: Language model
            tokenizer: Model tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate(
        self,
        task: str,
        examples: List[Dict],
        max_new_tokens: int = 50,
        batch_size: int = 1
    ) -> List[Dict]:
        """
        Evaluate model on task examples.
        
        Args:
            task: Task name
            examples: List of examples to evaluate
            max_new_tokens: Max tokens to generate
            batch_size: Batch size for evaluation
            
        Returns:
            List of results with predictions
        """
        logger.info(f"Evaluating on {task} ({len(examples)} examples)")
        
        results = []
        
        for example in tqdm(examples, desc=f"Evaluating {task}"):
            try:
                
                prediction = self.generate(
                    example["prompt"],
                    max_new_tokens=max_new_tokens
                )
                
                
                result = {
                    "task": task,
                    "prompt": example["prompt"],
                    "prediction": prediction,
                    "ground_truth": example.get("answer"),
                    "example": example
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating example: {e}")
                continue
                
        logger.info(f"Evaluated {len(results)}/{len(examples)} examples")
        return results
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        
        response = generated[len(prompt):].strip()
        
        return response