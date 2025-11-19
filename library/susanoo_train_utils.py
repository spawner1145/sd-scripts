import argparse
import os
import torch
import math
import time
from safetensors.torch import save_file
import logging
from diffusers import FlowMatchEulerDiscreteScheduler
import numpy as np
from PIL import Image
from accelerate import PartialState
from library import strategy_base, train_util
from library.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)

def add_susanoo_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None)
    parser.add_argument("--no_half_vae", action="store_true")
    parser.add_argument("--vae", type=str, default=None, help="Path to Flux VAE checkpoint file")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to Qwen model folder")
    parser.add_argument("--lsunet_path", type=str, default=None, help="Path to LSUNet checkpoint file")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to prepend to captions")
    parser.add_argument("--text_projection", action="store_true", help="Force use of text projection layer")
    
    # Flux-like arguments
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0, help="Discrete flow shift for the Euler Discrete Scheduler")
    parser.add_argument("--model_prediction_type", choices=["raw", "additive", "sigma_scaled"], default="raw", help="How to interpret and process the model prediction")
    parser.add_argument("--timestep_sampling", choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"], default="uniform", help="Method to sample timesteps")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Scale factor for sigmoid timestep sampling")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for training (if applicable)")
    parser.add_argument("--sample_prompts", type=str, default=None, help="Path to file with prompts to sample")
    parser.add_argument("--sample_every_n_steps", type=int, default=None, help="Sample every n steps")
    parser.add_argument("--sample_every_n_epochs", type=int, default=None, help="Sample every n epochs")
    
    # Advanced Loss Weighting (SD3/Flux style)
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["none", "sigma_sqrt", "cosmap", "logit_normal", "mode"], help="Loss weighting scheme")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit-normal weighting")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Std for logit-normal weighting")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale for mode weighting")
    
    # Text Encoder Caching
    parser.add_argument("--cache_text_encoder_outputs", action="store_true", help="Cache text encoder outputs to memory")
    parser.add_argument("--cache_text_encoder_outputs_to_disk", action="store_true", help="Cache text encoder outputs to disk")
    parser.add_argument("--text_encoder_batch_size", type=int, default=1, help="Batch size for text encoder caching")

def save_susanoo_model_on_train_end(args, save_dtype, epoch, global_step, unet, text_projection=None):
    save_susanoo_model(args, epoch, global_step, None, unet, text_projection, is_final=True, save_dtype=save_dtype)

def save_susanoo_model(args, epoch, global_step, accelerator, unet, text_projection=None, is_final=False, save_dtype=None, metadata=None):
    if args.output_dir is None:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    
    if accelerator is not None:
        unet_to_save = accelerator.unwrap_model(unet)
        if text_projection is not None:
            proj_to_save = accelerator.unwrap_model(text_projection)
    else:
        unet_to_save = unet
        proj_to_save = text_projection

    state_dict = unet_to_save.state_dict()
    
    if save_dtype is not None:
        for key in list(state_dict.keys()):
            v = state_dict[key]
            v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    if is_final:
        filename = f"susanoo_final.safetensors"
    else:
        filename = f"susanoo_epoch{epoch}_step{global_step}.safetensors"
        
    path = os.path.join(args.output_dir, filename)
    
    logger.info(f"Saving model to {path}")
    save_file(state_dict, path, metadata=metadata)
    
    if text_projection is not None:
        proj_state_dict = proj_to_save.state_dict()
        if save_dtype is not None:
            for key in list(proj_state_dict.keys()):
                v = proj_state_dict[key]
                v = v.detach().clone().to("cpu").to(save_dtype)
                proj_state_dict[key] = v
                
        if is_final:
            proj_filename = f"susanoo_proj_final.safetensors"
        else:
            proj_filename = f"susanoo_proj_epoch{epoch}_step{global_step}.safetensors"
        proj_path = os.path.join(args.output_dir, proj_filename)
        logger.info(f"Saving projection to {proj_path}")
        save_file(proj_state_dict, proj_path, metadata=metadata)

def sample_images(accelerator, args, epoch, global_step, unet, vae, text_encoder, text_projection=None):
    if args.sample_prompts is None:
        return
    
    if not os.path.exists(args.sample_prompts):
        logger.warning(f"Sample prompts file not found: {args.sample_prompts}")
        return

    logger.info(f"Sampling images at step {global_step}...")
    
    # Use train_util to load prompts with options
    prompts = train_util.load_prompts(args.sample_prompts)
    
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # Distributed sampling
    distributed_state = PartialState()
    
    # Unwrap models
    unet = accelerator.unwrap_model(unet)
    if text_projection is not None:
        text_projection = accelerator.unwrap_model(text_projection)
    
    # Switch to eval
    unet.eval()
    if text_projection is not None:
        text_projection.eval()
    if vae is not None:
        vae.to(accelerator.device)
        vae.eval()

    # Save RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()

    if distributed_state.num_processes <= 1:
        for prompt_dict in prompts:
            sample_image_inference(
                accelerator, args, unet, vae, text_encoder, text_projection,
                save_dir, prompt_dict, epoch, global_step
            )
    else:
        per_process_prompts = []
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i::distributed_state.num_processes])
            
        with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
            for prompt_dict in prompt_dict_lists[0]:
                sample_image_inference(
                    accelerator, args, unet, vae, text_encoder, text_projection,
                    save_dir, prompt_dict, epoch, global_step
                )

    # Restore RNG state
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
        
    # Restore train mode
    unet.train()
    if text_projection is not None:
        text_projection.train()
        
    clean_memory_on_device(accelerator.device)

def sample_image_inference(
    accelerator, args, unet, vae, text_encoder, text_projection,
    save_dir, prompt_dict, epoch, global_step
):
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt")
    width = prompt_dict.get("width", 1024)
    height = prompt_dict.get("height", 1024)
    sample_steps = prompt_dict.get("sample_steps", 20)
    scale = prompt_dict.get("scale", 1.0) # Guidance scale (CFG)
    seed = prompt_dict.get("seed")
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()
            
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Height: {height}, Width: {width}, Steps: {sample_steps}, Scale: {scale}, Seed: {seed}")
    
    # Strategies
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    
    with torch.no_grad():
        # 1. Text Encoding
        input_ids, attention_mask = tokenize_strategy.tokenize(prompt)
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device).float()
        
        encoder_hidden_states = text_encoder(input_ids).last_hidden_state.to(unet.dtype)
        
        if text_projection is not None:
            encoder_hidden_states = text_projection(encoder_hidden_states)

        # 2. Latents Initialization
        latents = torch.randn(
            (1, 16, height // 8, width // 8), 
            device=accelerator.device, 
            dtype=unet.dtype,
            generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None
        )
        
        # 3. Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        scheduler.set_timesteps(sample_steps)
        
        # 4. Denoising Loop
        for t in scheduler.timesteps:
            # Model prediction
            model_pred = unet(latents, t, encoder_hidden_states, context_mask=attention_mask)
            
            # Step
            latents = scheduler.step(model_pred, t, latents).prev_sample

        # 5. Decode
        if vae is not None:
            image = vae.decode(latents)
            
            # Post-process
            image = image.clamp(-1, 1)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1).cpu().numpy() # (B, H, W, C)
            image = (image * 255).astype(np.uint8)[0]
            
            img_pil = Image.fromarray(image)
            
            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
            seed_suffix = "" if seed is None else f"_{seed}"
            i: int = prompt_dict.get("enum", 0)
            img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
            img_path = os.path.join(save_dir, img_filename)
            
            img_pil.save(img_path)
            logger.info(f"Saved sample to {img_path}")
            
            # WandB
            if "wandb" in [tracker.name for tracker in accelerator.trackers]:
                import wandb
                wandb_tracker = accelerator.get_tracker("wandb")
                wandb_tracker.log({f"sample_{i}": wandb.Image(img_path, caption=prompt)}, step=global_step)


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

def get_noisy_model_input_and_timesteps(args, noise, latents, device):
    bs = latents.shape[0]
    
    # Advanced Sampling with Density (SD3/Flux style)
    if hasattr(args, "weighting_scheme") and args.weighting_scheme in ["logit_normal", "mode"]:
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bs,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        # Map u [0, 1] to timesteps [0, 1]
        # Note: Flux uses discrete timesteps from scheduler, but here we use continuous [0, 1]
        # If we want to match Flux exactly, we might need to map to discrete steps, but continuous is fine for Flow Matching usually.
        timesteps = u.to(device)
    else:
        # Timestep Sampling
        if args.timestep_sampling == "sigma" or args.timestep_sampling == "uniform":
            # Simple Uniform Sampling t \in [0, 1]
            timesteps = torch.rand((bs,), device=device)
        elif args.timestep_sampling == "sigmoid":
            # Sigmoid Sampling (used in some advanced configs)
            t = torch.randn((bs,), device=device)
            timesteps = torch.sigmoid(t * args.sigmoid_scale)
        elif args.timestep_sampling == "shift":
            # Shift Sampling (Simple shift)
            t = torch.rand((bs,), device=device)
            timesteps = (t * args.discrete_flow_shift) / (1 + (args.discrete_flow_shift - 1) * t)
        elif args.timestep_sampling == "flux_shift":
            # Flux Shift Sampling (Logit-Normal with shift)
            t = torch.sigmoid(torch.randn((bs,), device=device))
            timesteps = (t * args.discrete_flow_shift) / (1 + (args.discrete_flow_shift - 1) * t)
        else:
            timesteps = torch.rand((bs,), device=device)

    # Interpolate (Optimal Transport Path / Linear)
    # x_t = (1 - t) * x_0 + t * x_1
    # sigmas = t in this formulation
    sigmas = timesteps.view(bs, 1, 1, 1)
    noisy_latents = (1 - sigmas) * latents + sigmas * noise
    
    return noisy_latents, timesteps, sigmas

def apply_model_prediction_type(args, model_pred, noisy_latents, sigmas):
    weighting = None
    if args.model_prediction_type == "additive":
        # model_pred is additive noise
        model_pred = model_pred + noisy_latents
    elif args.model_prediction_type == "sigma_scaled":
        # model_pred is sigma scaled
        model_pred = model_pred * (-sigmas) + noisy_latents
        
        # Apply weighting if specified (SD3 style)
        if hasattr(args, "weighting_scheme") and args.weighting_scheme != "none":
             weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)

    return model_pred, weighting
    
    if not os.path.exists(args.sample_prompts):
        logger.warning(f"Sample prompts file not found: {args.sample_prompts}")
        return

    logger.info(f"Sampling images at step {global_step}...")
    
    # Use train_util to load prompts with options
    prompts = train_util.load_prompts(args.sample_prompts)
    
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # Distributed sampling
    distributed_state = PartialState()
    
    # Unwrap models
    unet = accelerator.unwrap_model(unet)
    if text_projection is not None:
        text_projection = accelerator.unwrap_model(text_projection)
    
    # Switch to eval
    unet.eval()
    if text_projection is not None:
        text_projection.eval()
    if vae is not None:
        vae.to(accelerator.device)
        vae.eval()

    # Save RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()

    if distributed_state.num_processes <= 1:
        for prompt_dict in prompts:
            sample_image_inference(
                accelerator, args, unet, vae, text_encoder, text_projection,
                save_dir, prompt_dict, epoch, global_step
            )
    else:
        per_process_prompts = []
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i::distributed_state.num_processes])
            
        with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
            for prompt_dict in prompt_dict_lists[0]:
                sample_image_inference(
                    accelerator, args, unet, vae, text_encoder, text_projection,
                    save_dir, prompt_dict, epoch, global_step
                )

    # Restore RNG state
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
        
    # Restore train mode
    unet.train()
    if text_projection is not None:
        text_projection.train()
        
    clean_memory_on_device(accelerator.device)

def sample_image_inference(
    accelerator, args, unet, vae, text_encoder, text_projection,
    save_dir, prompt_dict, epoch, global_step
):
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt")
    width = prompt_dict.get("width", 1024)
    height = prompt_dict.get("height", 1024)
    sample_steps = prompt_dict.get("sample_steps", 20)
    scale = prompt_dict.get("scale", 1.0) # Guidance scale (CFG)
    seed = prompt_dict.get("seed")
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()
            
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Height: {height}, Width: {width}, Steps: {sample_steps}, Scale: {scale}, Seed: {seed}")
    
    # Strategies
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    
    with torch.no_grad():
        # 1. Text Encoding
        input_ids, attention_mask = tokenize_strategy.tokenize(prompt)
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device).float()
        
        encoder_hidden_states = text_encoder(input_ids).last_hidden_state.to(unet.dtype)
        
        if text_projection is not None:
            encoder_hidden_states = text_projection(encoder_hidden_states)

        # 2. Latents Initialization
        latents = torch.randn(
            (1, 16, height // 8, width // 8), 
            device=accelerator.device, 
            dtype=unet.dtype,
            generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None
        )
        
        # 3. Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        scheduler.set_timesteps(sample_steps)
        
        # 4. Denoising Loop
        for t in scheduler.timesteps:
            # Model prediction
            model_pred = unet(latents, t, encoder_hidden_states, context_mask=attention_mask)
            
            # Step
            latents = scheduler.step(model_pred, t, latents).prev_sample

        # 5. Decode
        if vae is not None:
            # Assuming vae.decode handles standard (B, C, H, W)
            # Flux VAE decode expects z to be scaled back? 
            # Flux VAE decode: z = (z / scale_factor) + shift_factor
            # But flux_models.AutoEncoder.decode does NOT apply this automatically if it's just a wrapper around decoder.
            # Let's check flux_models.py or flux_utils.py usage.
            # In flux_train.py: latents = vae.encode(imgs) -> this returns posterior.sample() which IS scaled.
            # So decode should expect scaled latents.
            # flux_models.AutoEncoder.decode calls self.decoder(z)
            # We need to check if decoder handles scaling.
            # Usually in SD-Scripts, VAE encode/decode are symmetric.
            
            image = vae.decode(latents)
            
            # Post-process
            image = image.clamp(-1, 1)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1).cpu().numpy() # (B, H, W, C)
            image = (image * 255).astype(np.uint8)[0]
            
            img_pil = Image.fromarray(image)
            
            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
            seed_suffix = "" if seed is None else f"_{seed}"
            i: int = prompt_dict.get("enum", 0)
            img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
            img_path = os.path.join(save_dir, img_filename)
            
            img_pil.save(img_path)
            logger.info(f"Saved sample to {img_path}")
            
            # WandB
            if "wandb" in [tracker.name for tracker in accelerator.trackers]:
                import wandb
                wandb_tracker = accelerator.get_tracker("wandb")
                wandb_tracker.log({f"sample_{i}": wandb.Image(img_path, caption=prompt)}, step=global_step)


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

def get_noisy_model_input_and_timesteps(args, noise, latents, device):
    bs = latents.shape[0]
    
    # Advanced Sampling with Density (SD3/Flux style)
    if hasattr(args, "weighting_scheme") and args.weighting_scheme in ["logit_normal", "mode"]:
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bs,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        # Map u [0, 1] to timesteps [0, 1]
        # Note: Flux uses discrete timesteps from scheduler, but here we use continuous [0, 1]
        # If we want to match Flux exactly, we might need to map to discrete steps, but continuous is fine for Flow Matching usually.
        timesteps = u.to(device)
    else:
        # Timestep Sampling
        if args.timestep_sampling == "sigma" or args.timestep_sampling == "uniform":
            # Simple Uniform Sampling t \in [0, 1]
            timesteps = torch.rand((bs,), device=device)
        elif args.timestep_sampling == "sigmoid":
            # Sigmoid Sampling (used in some advanced configs)
            t = torch.randn((bs,), device=device)
            timesteps = torch.sigmoid(t * args.sigmoid_scale)
        elif args.timestep_sampling == "shift":
            # Shift Sampling (Simple shift)
            t = torch.rand((bs,), device=device)
            timesteps = (t * args.discrete_flow_shift) / (1 + (args.discrete_flow_shift - 1) * t)
        elif args.timestep_sampling == "flux_shift":
            # Flux Shift Sampling (Logit-Normal with shift)
            t = torch.sigmoid(torch.randn((bs,), device=device))
            timesteps = (t * args.discrete_flow_shift) / (1 + (args.discrete_flow_shift - 1) * t)
        else:
            timesteps = torch.rand((bs,), device=device)

    # Interpolate (Optimal Transport Path / Linear)
    # x_t = (1 - t) * x_0 + t * x_1
    # sigmas = t in this formulation
    sigmas = timesteps.view(bs, 1, 1, 1)
    noisy_latents = (1 - sigmas) * latents + sigmas * noise
    
    return noisy_latents, timesteps, sigmas

def apply_model_prediction_type(args, model_pred, noisy_latents, sigmas):
    weighting = None
    if args.model_prediction_type == "additive":
        # model_pred is additive noise
        model_pred = model_pred + noisy_latents
    elif args.model_prediction_type == "sigma_scaled":
        # model_pred is sigma scaled
        model_pred = model_pred * (-sigmas) + noisy_latents
        
        # Apply weighting if specified (SD3 style)
        if hasattr(args, "weighting_scheme") and args.weighting_scheme != "none":
             weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)

    return model_pred, weighting
