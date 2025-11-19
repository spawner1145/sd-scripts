
import torch
from transformers import AutoModel, AutoTokenizer

def check_qwen_output():
    model_name = "Sakiko-Prompt-Gen-v1.0"
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt")
    
    print("Input keys:", inputs.keys())
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Output keys:", outputs.keys())
    print("Has attention_mask in output?", hasattr(outputs, "attention_mask"))

if __name__ == "__main__":
    check_qwen_output()
