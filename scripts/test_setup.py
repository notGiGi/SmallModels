import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelLoader
from src.utils import setup_logging, print_metrics
import torch

def main():
    setup_logging()
    
    print("\n" + "="*60)
    print("TESTING SMALL MODELS RESEARCH SETUP")
    print("="*60)
    
    
    print("\n1. PyTorch:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    
    print("\n2. ModelLoader:")
    loader = ModelLoader()
    print(f"   Device: {loader.device}")
    print(f"   Available models: {list(loader.MODELS.keys())}")
    
    
    if torch.cuda.is_available():
        print("\n3. GPU Memory:")
        mem_info = loader.check_gpu_memory()
        print(f"   Total: {mem_info['total_gb']:.2f} GB")
        print(f"   Free: {mem_info['free_gb']:.2f} GB")
        
        print("\n4. Model VRAM Estimates:")
        for model_key in loader.MODELS.keys():
            vram = loader.estimate_vram(model_key)
            vram_4bit = loader.estimate_vram(model_key, load_in_4bit=True)
            fits = "✓" if vram < mem_info['free_gb'] else "✗"
            print(f"   {model_key:12s}: {vram:.2f}GB (4-bit: {vram_4bit:.2f}GB) {fits}")
    
    print("\n" + "="*60)
    print("✓ Setup test complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python scripts/test_setup.py")
    print("  2. Try loading a small model")
    print("  3. Run first evaluation")

if __name__ == "__main__":
    main()