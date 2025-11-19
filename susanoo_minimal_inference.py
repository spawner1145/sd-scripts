# Minimum Inference Code for Susanoo

import argparse
import math
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
import time
from typing import Callable, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from accelerate import init_empty_weights

from library import device_utils
from library.device_utils import init_ipex, get_preferred_device

init_ipex()

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import susanoo_models, susanoo_utils, strategy_susanoo
from library.susanoo_train_utils import get_lin_function, time_shift

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def generate_image(
    args,
    model: susanoo_models.LSUNet,
    vae: susanoo_models.AutoEncoder,
    text_encoder,
    tokenizer_path: str,
    text_projection,
    prompt: str,
    negative_prompt: str,
    system_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 1. Prepare Latents
    # Susanoo: 16 channels, downsample 8
    latent_height = height // 8
    latent_width = width // 8
    
    latents = torch.randn(
        (1, 16, latent_height, latent_width),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed)
    )

    # 2. Encode Text
    prompts = [prompt]
    if guidance_scale > 1.0:
        prompts = [negative_prompt or "", prompt]

    # Use Strategy for Tokenization
    # Max length set to 512 as requested
    tokenize_strategy = strategy_susanoo.SusanooTokenizeStrategy(
        tokenizer_path, 
        max_length=512, 
        system_prompt=system_prompt
    )
    
    tokens = tokenize_strategy.tokenize(prompts)
    input_ids = tokens[0].to(device)
    attention_mask = tokens[1].to(device)
    
    with torch.no_grad():
        encoder_outputs = text_encoder(input_ids, attention_mask=attention_mask)
        context = encoder_outputs.last_hidden_state.to(dtype)
        
        if text_projection is not None:
            context = text_projection(context)

    if args.offload:
        logger.info("Moving Text Encoder to CPU and Model to GPU...")
        text_encoder.to("cpu")
        if text_projection is not None:
            text_projection.to("cpu")
        torch.cuda.empty_cache()
        model.to(device)

    # 3. Schedule
    # Flux shift logic
    # image_seq_len for Flux is (h//2)*(w//2).
    # For Susanoo, let's use the same logic as in train_utils: (H/16)*(W/16)
    # latent_height = H/8.
    # So (latent_height // 2) * (latent_width // 2)
    image_seq_len = (latent_height // 2) * (latent_width // 2)
    timesteps = get_schedule(steps, image_seq_len, shift=True)

    # 4. Denoise
    with torch.no_grad():
        # Use autocast for mixed precision (especially if model is fp8 or bf16)
        # We use the device type for autocast (e.g. 'cuda')
        autocast_device = device.type if device.type != "mps" else "cpu" # MPS autocast might differ, but usually 'cuda' or 'cpu'
        
        for i, (t_curr, t_prev) in enumerate(zip(tqdm(timesteps[:-1]), timesteps[1:])):
            t_vec = torch.full((latents.shape[0],), t_curr, dtype=dtype, device=device)
            t_input = t_vec * 1000.0
            
            # Prepare inputs for CFG
            if guidance_scale > 1.0:
                # Duplicate latents for (uncond, cond)
                latents_input = torch.cat([latents, latents], dim=0)
                t_input = torch.cat([t_input, t_input], dim=0)
            else:
                latents_input = latents

            # Model Prediction
            with torch.autocast(device_type=autocast_device, dtype=dtype):
                model_pred = model(latents_input, t_input, context, context_mask=attention_mask)
            
            # CFG Guidance
            if guidance_scale > 1.0:
                pred_uncond, pred_text = torch.chunk(model_pred, 2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
            else:
                v_pred = model_pred

            # Euler Step
            dt = t_prev - t_curr
            latents = latents + dt * v_pred

    if args.offload:
        logger.info("Moving Model to CPU and VAE to GPU...")
        model.to("cpu")
        torch.cuda.empty_cache()
        vae.to(device)

    # 5. Decode
    # Flux VAE decode
    with torch.no_grad():
        image = vae.decode(latents)
    
    image = image.clamp(-1, 1)
    image = (image + 1) / 2
    image = image.permute(0, 2, 3, 1).cpu().float().numpy()
    image = (image * 255).astype(np.uint8)[0]
    return Image.fromarray(image)

def main():
    parser = argparse.ArgumentParser(description="Susanoo Minimal Inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to Susanoo LSUNet checkpoint")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to Flux VAE checkpoint")
    parser.add_argument("--text_encoder_path", type=str, default="Qwen/Qwen1.5-0.5B", help="Path to Qwen model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to Qwen tokenizer")
    parser.add_argument("--text_projection_path", type=str, default=None, help="Path to Text Projection checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative Prompt")
    parser.add_argument("--system_prompt", type=str, default=None, help="System Prompt")
    parser.add_argument("--output", type=str, default="output.png", help="Output filename")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", help="Data type for inference: fp16, bf16, float32, fp8")
    parser.add_argument("--offload", action="store_true", help="Offload models to CPU when not in use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}
    if hasattr(torch, "float8_e4m3fn"):
        dtype_map["fp8"] = torch.float8_e4m3fn
    
    dtype = dtype_map.get(args.dtype)
    if dtype is None:
        if args.dtype == "fp8":
            logger.warning("FP8 requested but torch.float8_e4m3fn is not available. Falling back to bf16.")
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {args.dtype}")

    # VAE and Text Encoder are usually better in BF16/FP16 even if Model is FP8
    # Also, latents are generated in this aux_dtype to avoid VAE decoding issues
    aux_dtype = torch.bfloat16 if args.dtype == "fp8" else dtype
    if args.dtype == "fp8" and not torch.cuda.is_bf16_supported():
         aux_dtype = torch.float16
    
    logger.info(f"Device: {device}, Model Dtype: {dtype}, Aux Dtype: {aux_dtype}")
    
    # Load Models
    logger.info("Loading models...")
    
    # 1. LSUNet
    if args.offload:
        model = susanoo_utils.load_lsunet(args.ckpt_path, dtype, "cpu")
    else:
        model = susanoo_utils.load_lsunet(args.ckpt_path, dtype, device)
    model.eval()
    
    # 2. VAE
    if args.offload:
        vae = susanoo_utils.load_vae(args.vae_path, aux_dtype, "cpu")
    else:
        vae = susanoo_utils.load_vae(args.vae_path, aux_dtype, device)
        vae.to(device, dtype=aux_dtype)
    vae.eval()
    
    # 3. Text Encoder (Qwen)
    if args.offload:
        text_encoder = susanoo_utils.load_text_encoder(args.text_encoder_path, aux_dtype, "cpu")
    else:
        text_encoder = susanoo_utils.load_text_encoder(args.text_encoder_path, aux_dtype, device)
        text_encoder.to(device, dtype=aux_dtype)
    
    # 4. Tokenizer
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.text_encoder_path
    # tokenizer = susanoo_utils.load_tokenizer(tokenizer_path) # Handled by strategy
    
    # 5. Text Projection (Optional)
    text_projection = None
    if args.text_projection_path:
        text_projection = susanoo_utils.load_text_projection(args.text_projection_path, dtype, device)
    
    # Generate
    logger.info("Generating image...")
    
    # Offloading Logic
    if args.offload:
        logger.info("Offloading enabled. Moving Text Encoder to GPU...")
        text_encoder.to(device)
        if text_projection is not None:
            text_projection.to(device)

    img = generate_image(
        args,
        model,
        vae,
        text_encoder,
        tokenizer_path,
        text_projection,
        args.prompt,
        args.negative_prompt,
        args.system_prompt,
        args.width,
        args.height,
        args.steps,
        args.guidance_scale,
        args.seed,
        device,
        aux_dtype # Use aux_dtype for latents and scheduling
    )
    
    img.save(args.output)
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

