import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MetricsAnalyzer:
    """
    Analyze and extend lm-eval results.
    Provides additional metrics beyond standard accuracy.
    """
    
    def summarize_results(self, lmeval_results: Dict) -> pd.DataFrame:
        """
        Convert lm-eval results to clean DataFrame.
        
        Args:
            lmeval_results: Results from lm-eval
            
        Returns:
            DataFrame with task-level metrics
        """
        rows = []
        
        for task, metrics in lmeval_results["results"].items():
            row = {
                "task": task,
                "accuracy": metrics.get("acc,none", metrics.get("acc_norm,none", 0)),
                "stderr": metrics.get("acc_stderr,none", 0),
                "num_samples": metrics.get("alias", task) 
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models across tasks.
        
        Args:
            results_dict: Dict mapping model_name -> lm-eval results
            
        Returns:
            Comparison DataFrame
        """
        all_data = {}
        
        for model_name, results in results_dict.items():
            df = self.summarize_results(results)
            all_data[model_name] = df.set_index("task")["accuracy"]
        
        comparison_df = pd.DataFrame(all_data)
        comparison_df = comparison_df.round(4)
        
        return comparison_df
    
    def calculate_relative_performance(
        self, 
        results_dict: Dict[str, Dict],
        baseline_model: str
    ) -> pd.DataFrame:
        """
        Calculate performance relative to baseline.
        
        Args:
            results_dict: Dict of model results
            baseline_model: Name of baseline model
            
        Returns:
            DataFrame with relative performance
        """
        comparison = self.compare_models(results_dict)
        
        if baseline_model not in comparison.columns:
            raise ValueError(f"Baseline model {baseline_model} not found")
        
        baseline = comparison[baseline_model]
        relative = comparison.div(baseline, axis=0) - 1  
        
        return relative.round(4)
    
    def identify_failure_patterns(
        self,
        model_results: Dict,
        threshold: float = 0.5
    ) -> Dict[str, List[str]]:
        """
        Identify tasks where model performs poorly.
        
        Args:
            model_results: Results from lm-eval
            threshold: Accuracy threshold for "failure"
            
        Returns:
            Dict categorizing tasks by performance
        """
        df = self.summarize_results(model_results)
        
        patterns = {
            "strong": df[df["accuracy"] >= 0.7]["task"].tolist(),
            "moderate": df[(df["accuracy"] >= threshold) & (df["accuracy"] < 0.7)]["task"].tolist(),
            "weak": df[df["accuracy"] < threshold]["task"].tolist()
        }
        
        return patterns
    
    def create_heatmap_data(
        self,
        results_dict: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Create data for capability heatmap visualization.
        
        Args:
            results_dict: Dict of model results
            
        Returns:
            DataFrame suitable for heatmap plotting
        """
        comparison = self.compare_models(results_dict)
        
        heatmap_data = comparison.T
        
        return heatmap_data
    
    def print_summary(self, results_dict: Dict[str, Dict]):
        """
        Print formatted summary of results.
        
        Args:
            results_dict: Dict of model results
        """
        comparison = self.compare_models(results_dict)
        
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(comparison.to_string())
        print("="*70)
        
    
        print("\nBEST MODEL PER TASK:")
        for task in comparison.index:
            best_model = comparison.loc[task].idxmax()
            best_score = comparison.loc[task].max()
            print(f"  {task:20s}: {best_model:15s} ({best_score:.2%})")