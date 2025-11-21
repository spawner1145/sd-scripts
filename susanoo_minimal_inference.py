# Minimum Inference Code for Susanoo

import argparse
import math
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
import time
from typing import Optional
import importlib
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from accelerate import init_empty_weights
from safetensors.torch import load_file

from library import device_utils
from library.device_utils import init_ipex, get_preferred_device
import networks.lora_susanoo as lora_susanoo

init_ipex()

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import susanoo_models, susanoo_utils, strategy_susanoo
from library.susanoo_train_utils import build_inference_schedule, prepare_initial_latents, decode_latents_with_vae, load_latents_from_path

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
    model_prediction_type: str,
):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 1. Prepare Latents
    latents_override = None
    if getattr(args, "latents_path", None):
        latents_override = load_latents_from_path(args.latents_path)

    latents = prepare_initial_latents(
        height,
        width,
        dtype,
        device,
        seed=seed,
        latents=latents_override,
    )

    # 2. Encode Text
    prompts = [prompt]
    if guidance_scale > 1.0:
        prompts = [negative_prompt or "", prompt]

    # Use Strategy for Tokenization
    # Max length set to 512 as requested
    tokenize_strategy = strategy_susanoo.SusanooTokenizeStrategy(
        tokenizer_path, 
        max_length=args.max_token_length, 
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.to(device)

    # 3. Schedule
    # Use Flux schedule logic
    # Flux uses (H/16)*(W/16) for image_seq_len (packed latents)
    image_seq_len = math.ceil(height / 16) * math.ceil(width / 16)
    timesteps = build_inference_schedule(
        steps,
        image_seq_len,
        timestep_sampling=args.timestep_sampling,
        discrete_flow_shift=args.discrete_flow_shift,
        sigmoid_scale=args.sigmoid_scale,
    )
    
    # 4. Denoise
    with torch.no_grad():
        autocast_device = device.type if device.type != "mps" else "cpu"  # torch.autocast doesn't fully support mps
        use_autocast = dtype in (torch.float16, torch.bfloat16)

        for i, (t_curr, t_next) in enumerate(zip(tqdm(timesteps[:-1]), timesteps[1:])):
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
            autocast_ctx = torch.autocast(device_type=autocast_device, dtype=dtype) if use_autocast else nullcontext()
            with autocast_ctx:
                model_pred = model(latents_input, t_input, context, context_mask=attention_mask)
            
            # CFG Guidance
            if guidance_scale > 1.0:
                pred_uncond, pred_text = torch.chunk(model_pred, 2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
            else:
                v_pred = model_pred

            if model_prediction_type == "sigma_scaled":
                sigma_val = max(t_curr, 1e-5)
                v_pred = (latents - v_pred) / sigma_val

            # Euler Step
            dt = t_next - t_curr
            latents = latents + dt * v_pred
            
            if i % 5 == 0:
                logger.info(f"Step {i}: Latents Mean={latents.mean().item():.4f}, Std={latents.std().item():.4f}, Min={latents.min().item():.4f}, Max={latents.max().item():.4f}")

    if args.offload:
        logger.info("Moving Model to CPU and VAE to GPU...")
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vae.to(device)

    # 5. Decode
    # Flux VAE decode
    logger.info(f"Decoding latents: Shape={latents.shape}, Mean={latents.mean().item():.4f}, Std={latents.std().item():.4f}")
    
    # Manual scaling fix attempt (if model output is unscaled)
    # latents = latents / latents.std() * 0.3611
    
    with torch.no_grad():
        image = decode_latents_with_vae(vae, latents)
        image = image[:, :, :height, :width]
    
    logger.info(f"Decoded Image: Mean={image.mean().item():.4f}, Std={image.std().item():.4f}, Min={image.min().item():.4f}, Max={image.max().item():.4f}")
    
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
    parser.add_argument("--latents_path", type=str, default=None, help="Optional path to precomputed latents (.pt/.npy/.npz). If omitted, random latents are used")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32", "fp8"], help="Dtype")
    parser.add_argument("--offload", action="store_true", help="Offload models to CPU when not in use")
    parser.add_argument("--lora_weights", type=str, nargs="*", default=[], help="LoRA weights, can be multiple. Format: path or path;multiplier")
    parser.add_argument("--merge_lora_weights", action="store_true", help="Merge LoRA weights to model")
    parser.add_argument("--max_token_length", type=int, default=512, help="Max token length for tokenizer")
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0, help="Shift value for FlowMatchEulerDiscreteScheduler")
    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
        default="shift",
        help="Match the training timestep sampling strategy",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for sigmoid timestep sampling",
    )
    parser.add_argument(
        "--model_prediction_type",
        choices=["raw", "additive", "sigma_scaled"],
        default="raw",
        help="Match the training prediction type so inference interprets the model output correctly",
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}
    if hasattr(torch, "float8_e4m3fn"):
        dtype_map["fp8"] = torch.float8_e4m3fn
    
    dtype = dtype_map.get(args.dtype)
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if args.dtype == "fp8" and dtype is fp8_dtype:
        logger.warning("FP8 execution is experimental for Susanoo. Falling back to bf16 for computation.")
        dtype = torch.bfloat16
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
    vae = susanoo_utils.load_vae(args.vae_path, dtype, device)
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
        text_projection.eval()
    
    # Load LoRA
    lora_models = []
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        weights_sd = load_file(weights_file)
        lora_model, _ = lora_susanoo.create_network_from_weights(
            multiplier, 
            weights_file, 
            vae, 
            text_encoder, 
            model, 
            weights_sd, 
            True
        )

        if args.merge_lora_weights:
            lora_model.merge_to(text_encoder, model, weights_sd, dtype, device)
        else:
            lora_model.apply_to(text_encoder, model)
            info = lora_model.load_state_dict(weights_sd, strict=False)
            logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
            lora_model.to(device)
            lora_model.set_multiplier(multiplier)
            lora_model.eval()

        lora_models.append(lora_model)

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
        aux_dtype,  # Use aux_dtype for latents and scheduling
        args.model_prediction_type,
    )
    
    img.save(args.output)
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

