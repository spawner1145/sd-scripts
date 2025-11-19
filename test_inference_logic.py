
import torch
import sys
import os
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

from susanoo_minimal_inference import generate_image
from library import susanoo_models, susanoo_utils
import library.strategy_susanoo as strategy_susanoo

def test_inference_loop():
    print("Testing inference loop logic with REAL VAE and Text Encoder...")
    
    # Paths provided by user
    text_encoder_path = "Sakiko-Prompt-Gen-v1.0"
    vae_path = "Pad Flux EQ v2 B1.safetensors"
    
    # Mock args
    args = MagicMock()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dtype = "bf16"
    
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"Device: {device}, Dtype: {dtype}")

    # 1. LSUNet (Random Init - No checkpoint yet)
    print("Initializing LSUNet (Random Weights)...")
    model = susanoo_models.LSUNet().to(device, dtype)
    model.eval()
    
    # 2. Load Real VAE
    print(f"Loading VAE from {vae_path}...")
    if os.path.exists(vae_path):
        vae = susanoo_utils.load_vae(vae_path, dtype, device)
        vae.to(device, dtype) # Ensure it's on the right device
        vae.eval()
    else:
        print(f"Warning: VAE path {vae_path} not found. Using Mock.")
        vae = MagicMock()
        vae.decode.return_value = torch.randn(1, 3, 1024, 1024, device=device, dtype=dtype)

    # 3. Load Real Text Encoder
    print(f"Loading Text Encoder from {text_encoder_path}...")
    if os.path.exists(text_encoder_path):
        text_encoder = susanoo_utils.load_text_encoder(text_encoder_path, dtype, device)
        text_encoder.to(device, dtype) # Ensure it's on the right device
    else:
        print(f"Warning: Text Encoder path {text_encoder_path} not found. Using Mock.")
        text_encoder = MagicMock()
        def mock_text_encoder_forward(input_ids, **kwargs):
            batch_size = input_ids.shape[0]
            output = MagicMock()
            output.last_hidden_state = torch.randn(batch_size, 77, 1024, device=device, dtype=dtype)
            return output
        text_encoder.side_effect = mock_text_encoder_forward

    # 4. Tokenizer Strategy
    # If text encoder exists, we assume it has tokenizer. 
    # We don't need to mock strategy if we have the real path.
    tokenizer_path = text_encoder_path if os.path.exists(text_encoder_path) else "dummy_path"
    
    # If we are using real text encoder, we should probably NOT mock the strategy 
    # unless the tokenizer is missing.
    # But generate_image creates the strategy internally.
    # If the path is valid, it should work.
    
    try:
        img = generate_image(
            args,
            model,
            vae,
            text_encoder,
            tokenizer_path,
            None, # text_projection
            "A beautiful sunset",
            "",
            None,
            64, # width (small for speed)
            64, # height
            2, # steps
            4.0, # guidance_scale
            42, # seed
            device,
            dtype
        )
        print("Inference loop finished successfully!")
        print(f"Generated image size: {img.size}")
        
    except Exception as e:
        print(f"Inference loop failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_loop()
