import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage small language models with memory optimization."""
    

    MODELS = {
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "gemma2b": "google/gemma-2b-it",
        "qwen": "Qwen/Qwen2-1.5B-Instruct",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "smollm": "HuggingFaceTB/SmolLM-360M-Instruct"
    }
    
    
    PARAM_COUNTS = {
        "phi3": 3.8,
        "gemma2b": 2.5,
        "qwen": 1.5,
        "tinyllama": 1.1,
        "smollm": 0.36
    }
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"ModelLoader initialized with device: {self.device}")
        
    def load_model(
        self, 
        model_key: str, 
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}. Choose from: {list(self.MODELS.keys())}")
            
        model_name = self.MODELS[model_key]
        
        logger.info(f"Loading {model_key} ({model_name})")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  4-bit: {load_in_4bit}, 8-bit: {load_in_8bit}, CAREFUL! WE DONT NEED TO MAKE QUANTIZATIONS")
        
        
        vram = self.estimate_vram(model_key, load_in_4bit, load_in_8bit)
        logger.info(f"  Estimated VRAM: {vram:.2f} GB")
        
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        
        if load_in_4bit and self.device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        elif load_in_8bit and self.device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            
        logger.info(f"Model loaded successfully")
        return model, tokenizer
    
    def estimate_vram(
        self, 
        model_key: str, 
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ) -> float:

        params = self.PARAM_COUNTS.get(model_key, 1.0)
        
        if load_in_4bit:
      
            vram = (params * 0.5) + 0.5
        elif load_in_8bit:
          
            vram = params + 0.5
        else:
       
            vram = (params * 2) + 1.0
            
        return vram
    
    def check_gpu_memory(self) -> dict:
        if not torch.cuda.is_available():
            return {"available": False}
            
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        free = total - allocated
        
        return {
            "available": True,
            "total_gb": total,
            "allocated_gb": allocated,
            "free_gb": free
        }