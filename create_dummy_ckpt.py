
import torch
from safetensors.torch import save_file
from library import susanoo_models

def create_dummy_checkpoint():
    print("Creating dummy LSUNet model...")
    model = susanoo_models.LSUNet()
    
    print("Saving dummy checkpoint to dummy_lsunet.safetensors...")
    state_dict = model.state_dict()
    save_file(state_dict, "dummy_lsunet.safetensors")
    print("Done.")

if __name__ == "__main__":
    create_dummy_checkpoint()
