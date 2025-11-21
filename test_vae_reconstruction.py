
import torch
import numpy as np
from PIL import Image, ImageDraw
import os
import sys
from library import susanoo_utils
from library.susanoo_train_utils import decode_latents_with_vae

def create_geometric_image(width=1024, height=1024):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    # Red Circle
    draw.ellipse([100, 100, 400, 400], fill="red", outline="black")
    
    # Blue Rectangle
    draw.rectangle([500, 100, 900, 400], fill="blue", outline="black")
    
    # Green Triangle
    draw.polygon([(250, 600), (100, 900), (400, 900)], fill="green", outline="black")
    
    # Yellow Circle inside Blue Rectangle
    draw.ellipse([600, 150, 800, 350], fill="yellow", outline="black")
    
    return img

def preprocess_image(image, device, dtype):
    # Convert to tensor, normalize to [-1, 1]
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = img_np * 2.0 - 1.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    return tensor.to(device, dtype)

def postprocess_image(tensor):
    # Denormalize from [-1, 1] to [0, 255]
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2
    tensor = tensor.permute(0, 2, 3, 1).cpu().float().numpy()
    img_np = (tensor * 255).astype(np.uint8)[0]
    return Image.fromarray(img_np)

def test_vae_reconstruction():
    print("Testing VAE Reconstruction...")
    
    vae_path = "Pad Flux EQ v2 B1.safetensors"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"Device: {device}, Dtype: {dtype}")
    
    # 1. Create Image
    print("Creating geometric image...")
    original_img = create_geometric_image()
    original_img.save("vae_test_original.png")
    
    # 2. Load VAE
    print(f"Loading VAE from {vae_path}...")
    if not os.path.exists(vae_path):
        print(f"Error: VAE path {vae_path} not found.")
        return

    vae = susanoo_utils.load_vae(vae_path, dtype, device)
    vae.to(device, dtype) # Ensure model is on the correct device
    vae.eval()
    
    # 3. Preprocess
    input_tensor = preprocess_image(original_img, device, dtype)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # 4. Encode & Decode
    print("Encoding and Decoding...")
    with torch.no_grad():
        # Encode
        latents = vae.encode(input_tensor)
        print(f"Latents shape: {latents.shape}")
        
        # Decode via original API
        reconstructed_direct = vae.decode(latents)
        
        # Decode via Susanoo helper (manual scaling + decoder)
        reconstructed_helper = decode_latents_with_vae(vae, latents)
        print(f"Reconstruction tensor shape: {reconstructed_helper.shape}")
        
        # Compare both decoding paths to ensure logic parity
        diff = (reconstructed_direct - reconstructed_helper).float()
        max_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()
        print(f"Direct vs helper decode — max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
        if max_diff > 5e-4:
            print("Warning: decode helper diverges from original vae.decode beyond tolerance!")
        
        reconstructed_tensor = reconstructed_helper
        
    # 5. Postprocess & Save
    reconstructed_img = postprocess_image(reconstructed_tensor)
    reconstructed_img.save("vae_test_reconstructed.png")
    
    # Simple reconstruction quality metrics
    l1 = (input_tensor - reconstructed_tensor).abs().mean().item()
    mse = torch.mean((input_tensor - reconstructed_tensor) ** 2).item()
    print(f"Reconstruction L1 loss: {l1:.6f}, MSE: {mse:.6f}")
    
    print("Done! Saved 'vae_test_original.png' and 'vae_test_reconstructed.png'.")

if __name__ == "__main__":
    test_vae_reconstruction()
