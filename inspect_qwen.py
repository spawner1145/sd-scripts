
from transformers import AutoModel, AutoConfig
import torch

def inspect_qwen():
    print("Inspecting Qwen structure...")
    try:
        # config = AutoConfig.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
        # model = AutoModel.from_config(config, trust_remote_code=True)
        model = AutoModel.from_pretrained("Sakiko-Prompt-Gen-v1.0", trust_remote_code=True)
        
        print("Model class:", model.__class__.__name__)
        
        modules = set()
        for name, module in model.named_modules():
            modules.add(module.__class__.__name__)
            
        print("Modules found:", modules)
        
        # Check specific layers
        for name, module in model.named_modules():
            if "layers.0" in name:
                print(f"{name}: {module.__class__.__name__}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_qwen()
