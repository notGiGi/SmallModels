import torch
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class InteractiveEvaluator:
    """
    Interactive evaluator for qualitative analysis and debugging.
    Not for computing final metrics (use lm-eval for that).
    """
    
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
        
        logger.info(f"InteractiveEvaluator initialized on {device}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate text from prompt.
        Useful for quick testing and exploration.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample
            
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
                do_sample=do_sample or temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
       
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        
        response = generated[len(prompt):].strip()
        
        return response
    
    def analyze_failures(
        self,
        examples: List[Dict],
        max_examples: int = 50,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate detailed responses for failure analysis.
        
        Args:
            examples: List of examples with 'prompt' and 'answer'
            max_examples: Maximum examples to analyze
            save_path: Optional path to save results
            
        Returns:
            List of detailed results for manual inspection
        """
        logger.info(f"Analyzing {min(len(examples), max_examples)} examples for failures")
        
        results = []
        
        for i, example in enumerate(tqdm(examples[:max_examples], desc="Analyzing")):
            try:
                
                prediction = self.generate(
                    example['prompt'],
                    max_new_tokens=100
                )
                
               
                result = {
                    'index': i,
                    'prompt': example['prompt'],
                    'prediction': prediction,
                    'ground_truth': example.get('answer', 'N/A'),
                    'metadata': example.get('metadata', {})
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error on example {i}: {e}")
                continue
        
      
        if save_path:
            import json
            from pathlib import Path
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {save_path}")
        
        return results
    
    def compare_prompts(
        self,
        base_prompt: str,
        variations: List[str],
        max_new_tokens: int = 50
    ) -> Dict[str, str]:
        """
        Compare different prompt formulations.
        Useful for understanding prompt sensitivity.
        
        Args:
            base_prompt: Base prompt template
            variations: List of variations to try
            max_new_tokens: Tokens to generate
            
        Returns:
            Dict mapping variation -> response
        """
        results = {}
        
        print("\n" + "="*60)
        print("PROMPT COMPARISON")
        print("="*60)
        
        for i, variation in enumerate(variations, 1):
            full_prompt = base_prompt.format(variation=variation)
            response = self.generate(full_prompt, max_new_tokens=max_new_tokens)
            
            results[variation] = response
            
            print(f"\n[Variation {i}]: {variation}")
            print(f"Response: {response}")
        
        print("="*60)
        
        return results
    
    def inspect_examples(
        self,
        examples: List[Dict],
        n: int = 5
    ):
        """
        Pretty-print examples for manual inspection.
        
        Args:
            examples: List of examples
            n: Number to display
        """
        print("\n" + "="*60)
        print(f"SHOWING {n} EXAMPLES")
        print("="*60)
        
        for i, ex in enumerate(examples[:n], 1):
            print(f"\n[Example {i}]")
            print("-"*60)
        
            prediction = self.generate(ex['prompt'], max_new_tokens=50)
            
            print(f"Prompt:\n  {ex['prompt'][:200]}...")
            print(f"\nModel Output:\n  {prediction}")
            print(f"\nGround Truth:\n  {ex.get('answer', 'N/A')}")
            print("-"*60)


class FailureCategorizer:
    """
    Categorize failure modes for systematic analysis.
    Use this AFTER getting results from lm-eval.
    """
    
    CATEGORIES = {
        'factual_error': 'Model got the fact wrong',
        'reasoning_error': 'Logical/reasoning mistake',
        'format_error': 'Correct answer, wrong format',
        'hallucination': 'Made up information',
        'ambiguous': 'Answer is unclear/ambiguous',
        'context_error': 'Misunderstood the context',
        'calculation_error': 'Math/calculation mistake'
    }
    
    def categorize_failure(
        self,
        prompt: str,
        prediction: str,
        ground_truth: str
    ) -> str:
        """
        Manually categorize a single failure.
        Returns category key.
        
        In practice, you'll do this manually for 200-300 examples.
        This is just a helper structure.
        """
        print("\n" + "="*60)
        print("CATEGORIZE FAILURE")
        print("="*60)
        print(f"Prompt: {prompt[:200]}...")
        print(f"\nPrediction: {prediction}")
        print(f"Ground Truth: {ground_truth}")
        print("\nCategories:")
        
        for i, (key, desc) in enumerate(self.CATEGORIES.items(), 1):
            print(f"  {i}. {key}: {desc}")
        
        print("\n0. Skip")
        print("="*60)
        
        choice = input("Select category (0-7): ").strip()
        
        try:
            choice_num = int(choice)
            if choice_num == 0:
                return "skip"
            category_key = list(self.CATEGORIES.keys())[choice_num - 1]
            return category_key
        except:
            return "unknown"
    
    def create_taxonomy(
        self,
        categorized_results: List[Dict]
    ) -> Dict:
        """
        Create failure taxonomy from categorized results.
        
        Args:
            categorized_results: List of results with 'category' field
            
        Returns:
            Summary statistics
        """
        from collections import Counter
        
        categories = [r['category'] for r in categorized_results if r.get('category')]
        counter = Counter(categories)
        
        total = len(categorized_results)
        taxonomy = {
            'total_examples': total,
            'categories': {}
        }
        
        for category, count in counter.items():
            taxonomy['categories'][category] = {
                'count': count,
                'percentage': count / total * 100
            }
        
        return taxonomy