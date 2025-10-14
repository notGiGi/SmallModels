import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_results(
    results: List[Dict],
    output_path: str,
    format: str = "json"
):
    """
    Save evaluation results to file.
    
    Args:
        results: List of result dictionaries
        output_path: Output file path
        format: Output format ("json" or "csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {output_path}")
        
    elif format == "csv":
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(file_path: str) -> List[Dict]:
    """
    Load evaluation results from file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        List of result dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    if file_path.suffix == ".json":
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        results = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(results)} results from {file_path}")
    return results


def print_metrics(metrics: Dict[str, float], task: str = ""):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        task: Task name (optional)
    """
    if task:
        print(f"\n{'='*50}")
        print(f"Metrics for {task.upper()}")
        print(f"{'='*50}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)