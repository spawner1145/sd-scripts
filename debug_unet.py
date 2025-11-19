
import torch
import sys
import os
from library import susanoo_models

def debug_unet_modules():
    print("Debugging UNet modules...")
    unet = susanoo_models.LSUNet()
    
    target_modules = ["LSTransformer2DModel", "ResnetBlock2D", "Downsample2D", "Upsample2D", "LSUNetBlock", "LSConv", "CrossAttention", "FeedForward"]
    
    found_targets = set()
    
    for name, module in unet.named_modules():
        if module.__class__.__name__ in target_modules:
            found_targets.add(module.__class__.__name__)
            print(f"Found target module: {name} ({module.__class__.__name__})")
            
            # Check children
            for child_name, child_module in module.named_modules():
                is_linear = child_module.__class__.__name__ == "Linear"
                is_conv2d = child_module.__class__.__name__ == "Conv2d"
                if is_linear or is_conv2d:
                    print(f"  -> Child: {child_name} ({child_module.__class__.__name__})")
                    
    print(f"Found targets: {found_targets}")

if __name__ == "__main__":
    debug_unet_modules()
