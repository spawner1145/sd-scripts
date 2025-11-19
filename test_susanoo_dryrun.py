
import torch
import sys
import os

# Add current directory to path so we can import library
sys.path.append(os.getcwd())

from library import susanoo_models
from library.susanoo_models import LSUNet, LSUNetBlock, AdaLayerNorm

def test_ada_layer_norm():
    print("Testing AdaLayerNorm...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    dim = 320
    time_embed_dim = 1280
    
    ada_norm = AdaLayerNorm(time_embed_dim, dim).to(device, dtype)
    
    x = torch.randn(2, 100, dim, device=device, dtype=dtype)
    t_emb = torch.randn(2, time_embed_dim, device=device, dtype=dtype)
    
    out = ada_norm(x, t_emb)
    
    assert out.shape == x.shape
    print("AdaLayerNorm test passed!")

def test_lsunet_block_with_ada():
    print("Testing LSUNetBlock with AdaLayerNorm...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    dim = 320
    heads = 8
    head_dim = 40
    context_dim = 1024
    time_embed_dim = 1280
    
    block = LSUNetBlock(dim, heads, head_dim, context_dim, time_embed_dim=time_embed_dim).to(device, dtype)
    
    # Mock LSConv to avoid Triton if needed, but let's try running it
    # If Triton fails, it should fallback to pytorch_ska
    
    x = torch.randn(2, 64, dim, device=device, dtype=dtype) # N=64 (8x8)
    context = torch.randn(2, 77, context_dim, device=device, dtype=dtype)
    t_emb = torch.randn(2, time_embed_dim, device=device, dtype=dtype)
    
    out = block(x, context=context, timestep=t_emb)
    
    assert out.shape == x.shape
    print("LSUNetBlock test passed!")

def test_lsunet_full():
    print("Testing full LSUNet forward pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "cpu": dtype = torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    model = LSUNet().to(device, dtype)
    
    # Input: (B, 16, H, W)
    # Susanoo uses 16 channels
    B = 1
    C = 16
    H = 64
    W = 64
    
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)
    timesteps = torch.tensor([500], device=device)
    context = torch.randn(B, 77, 1024, device=device, dtype=dtype)
    
    # Forward
    try:
        out = model(x, timesteps=timesteps, context=context)
        print(f"Output shape: {out.shape}")
        assert out.shape == (B, 16, H, W)
        print("LSUNet full forward pass passed!")
    except Exception as e:
        print(f"LSUNet forward failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ada_layer_norm()
    test_lsunet_block_with_ada()
    test_lsunet_full()
