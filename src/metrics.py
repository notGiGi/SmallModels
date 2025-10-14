from typing import Dict, List
import re
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate metrics for different task types."""
    
    def calculate_metrics(
        self, 
        task: str, 
        results: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate metrics for evaluation results.
        
        Args:
            task: Task name
            results: List of evaluation results
            
        Returns:
            Dict of metrics
        """
        if task == "boolq":
            return self._metrics_boolq(results)
        elif task == "triviaqa":
            return self._metrics_triviaqa(results)
        elif task == "hellaswag":
            return self._metrics_hellaswag(results)
        elif task == "gsm8k":
            return self._metrics_gsm8k(results)
        elif task == "humaneval":
            return self._metrics_humaneval(results)
        else:
            logger.warning(f"No metrics defined for task: {task}")
            return {}
    
    def _metrics_boolq(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for BoolQ (binary classification)."""
        correct = 0
        total = len(results)
        
        for result in results:
            pred = self._parse_bool(result["prediction"])
            true = result["ground_truth"]
            
            if pred == true:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _metrics_triviaqa(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for TriviaQA (exact match + F1)."""
        exact_matches = 0
        total = len(results)
        
        for result in results:
            pred = result["prediction"].lower().strip()
            true_answers = result["ground_truth"]["value"]
            
            
            if isinstance(true_answers, str):
                true_answers = [true_answers]
            
            for answer in true_answers:
                if self._normalize_answer(pred) == self._normalize_answer(answer):
                    exact_matches += 1
                    break
        
        accuracy = exact_matches / total if total > 0 else 0.0
        
        return {
            "exact_match": accuracy,
            "correct": exact_matches,
            "total": total
        }
    
    def _metrics_hellaswag(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for HellaSwag (multiple choice)."""
        correct = 0
        total = len(results)
        
        for result in results:
            pred = self._parse_choice(result["prediction"])
            true = result["ground_truth"]
            
            if pred == true:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _metrics_gsm8k(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for GSM8K (math problems)."""
        correct = 0
        total = len(results)
        
        for result in results:
            pred_num = self._extract_number(result["prediction"])
            true_num = self._extract_number(result["ground_truth"])
            
            if pred_num is not None and true_num is not None:
                if abs(pred_num - true_num) < 1e-4:  # Float comparison
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _metrics_humaneval(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for HumanEval (code generation)."""

        logger.warning("HumanEval metrics require code execution (not implemented)")
        return {
            "pass@1": 0.0,
            "total": len(results)
        }
    

    
    def _parse_bool(self, text: str) -> bool:
        """Parse Yes/No from text."""
        text = text.lower().strip()
        if any(word in text[:10] for word in ["yes", "true", "correct"]):
            return True
        if any(word in text[:10] for word in ["no", "false", "incorrect"]):
            return False
        return False  
    
    def _parse_choice(self, text: str) -> int:
        """Parse choice number (1-4) from text."""
        match = re.search(r'\b([1-4])\b', text)
        if match:
            return int(match.group(1))
        return 0  # Invalid
    
    def _extract_number(self, text: str) -> float:
        """Extract number from text."""

        match = re.search(r'-?\d+\.?\d*', str(text))
        if match:
            try:
                return float(match.group())
            except:
                return None
        return None
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""

        text = text.lower().strip()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()