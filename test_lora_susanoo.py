
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from networks import lora_susanoo
from library import susanoo_models
from transformers import CLIPTextModel, CLIPConfig

def test_lora_creation():
    print("Testing LoRA creation...")
    
    # Mock models
    # We need to initialize LSUNet with minimal args to avoid errors if any
    # LSUNet __init__ takes **kwargs but uses global constants from susanoo_models
    # We might need to set them or mock them if they are not set
    
    try:
        unet = susanoo_models.LSUNet()
    except Exception as e:
        print(f"Failed to create LSUNet: {e}")
        return

    # text_encoder = CLIPTextModel(CLIPConfig(hidden_size=768, num_hidden_layers=1, num_attention_heads=4, vocab_size=1000))
    # Mock Qwen
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained("Sakiko-Prompt-Gen-v1.0", trust_remote_code=True)
    # Reduce size for testing
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    
    text_encoder = AutoModel.from_config(config, trust_remote_code=True)
    
    vae = None 
    
    # Create network
    try:
        network = lora_susanoo.create_network(
            multiplier=1.0,
            network_dim=4,
            network_alpha=1.0,
            vae=vae,
            text_encoder=text_encoder,
            unet=unet
        )
        
        print("Network created successfully.")
        print(f"Network class: {type(network)}")
        
        lora_layers = list(network.unet_loras)
        print(f"Number of LoRA layers in UNet: {len(lora_layers)}")
        
        if len(lora_layers) > 0:
            print("LoRA layers found!")
        else:
            print("No LoRA layers found. Check UNET_TARGET_REPLACE_MODULE.")
            
    except Exception as e:
        print(f"Failed to create network: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lora_creation()
