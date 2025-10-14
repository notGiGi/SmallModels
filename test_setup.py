import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("Testing Small Models Research Setup")
print("=" * 60)
print()

# Check Python version
print(f"✓ Python version: {sys.version.split()[0]}")

# Check GPU
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    print("⚠ No GPU found - will use CPU (slower)")

print()
print("Testing model download (this will take a minute)...")

try:
    # Test with smallest model
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("✓ Tokenizer loaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ Model loaded")
    
    # Test generation
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"✓ Generation works: '{result}'")
    
    print()
    print("=" * 60)
    print("🎉 All tests passed! You're ready to start.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Read QUICK_START.md")
    print("2. Run: python evaluate_models.py --setup")
    print("3. Start your first evaluation!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Troubleshooting:")
    print("- Make sure you have internet connection")
    print("- Try running again (downloads can be flaky)")
    print("- Check HuggingFace status: https://status.huggingface.co/")
