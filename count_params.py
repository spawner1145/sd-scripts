
import torch
from library import susanoo_models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_model():
    print("Initializing LSUNet...")
    model = susanoo_models.LSUNet()
    
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Estimated Size (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"Estimated Size (BF16): {total_params * 2 / 1024 / 1024:.2f} MB")
    
    # Check specific blocks
    print("\nChecking Level 2 Transformer Blocks:")
    # Accessing via internal structure (might be fragile)
    # input_blocks[4] is level 1 downsample
    # input_blocks[5] is level 2 block 0
    
    # Let's just iterate and find LSTransformer2DModel
    transformer_count = 0
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, susanoo_models.LSTransformer2DModel):
            transformer_count += 1
            layer_count += len(module.transformer_blocks)
            # print(f"  {name}: {len(module.transformer_blocks)} layers")

    print(f"Total LSTransformer2DModel modules: {transformer_count}")
    print(f"Total Transformer Layers: {layer_count}")

if __name__ == "__main__":
    analyze_model()
