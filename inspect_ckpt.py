
from safetensors.torch import load_file
import os

def inspect_checkpoint():
    filename = "dummy_lsunet.safetensors"
    size = os.path.getsize(filename)
    print(f"File size: {size / 1024 / 1024:.2f} MB")
    
    try:
        sd = load_file(filename)
        print(f"Total keys: {len(sd)}")
        
        # Count params in SD
        total_params = 0
        for k, v in sd.items():
            total_params += v.numel()
            
        print(f"Total params in file: {total_params:,}")
        
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    inspect_checkpoint()
