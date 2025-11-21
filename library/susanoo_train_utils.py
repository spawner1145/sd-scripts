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


def _should_sample(args, epoch, global_step):
    if args.sample_prompts is None:
        return False

    if global_step == 0:
        return bool(getattr(args, "sample_at_first", False))

    every_n_steps = getattr(args, "sample_every_n_steps", None)
    every_n_epochs = getattr(args, "sample_every_n_epochs", None)

    if every_n_steps is None and every_n_epochs is None:
        return False

    if every_n_epochs is not None:
        if epoch is None or epoch % every_n_epochs != 0:
            return False
        return True

    if every_n_steps is None:
        return False

    if global_step % every_n_steps != 0:
        return False

    # Avoid double-sampling at epoch boundaries when both triggers are enabled elsewhere
    if epoch is not None:
        return False

    return True


def _get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _has_vae_scaling(vae):
    return hasattr(vae, "scale_factor") and hasattr(vae, "shift_factor")


def load_latents_from_path(latents_path: str):
    if latents_path is None:
        return None
    ext = os.path.splitext(latents_path)[1].lower()
    if ext in {".pt", ".pth"}:
        return torch.load(latents_path, map_location="cpu")
    if ext == ".npz":
        data = np.load(latents_path)
        key = "latents" if "latents" in data.files else list(data.files)[0]
        return torch.from_numpy(data[key])
    if ext == ".npy":
        arr = np.load(latents_path)
        return torch.from_numpy(arr)
    raise ValueError(f"Unsupported latents file format: {latents_path}")


def prepare_initial_latents(height, width, dtype, device, seed=None, latents=None):
    latent_h = max(1, height // 8)
    latent_w = max(1, width // 8)
    shape = (1, 16, latent_h, latent_w)
    if latents is not None:
        latents = torch.as_tensor(latents, dtype=dtype)
        if latents.ndim == 4:
            pass
        elif latents.ndim == 3:
            latents = latents.unsqueeze(0)
        else:
            raise ValueError("Latents tensor must be 3D or 4D")
        if latents.shape[1:] != shape[1:]:
            raise ValueError(f"Latents shape {latents.shape} does not match expected {shape}")
        return latents.to(device=device, dtype=dtype)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def decode_latents_with_vae(vae, latents):
    latents = latents.to(dtype=vae.dtype)
    if _has_vae_scaling(vae):
        latents = latents / vae.scale_factor + vae.shift_factor
    if hasattr(vae, "decoder"):
        return vae.decoder(latents)
    return vae.decode(latents)

def add_susanoo_train_arguments(parser: argparse.ArgumentParser):
    # parser.add_argument("--tokenizer_cache_dir", type=str, default=None)
    if not any(action.dest == 'no_half_vae' for action in parser._actions):
        parser.add_argument("--no_half_vae", action="store_true", help="Do not use half precision for VAE")
    
    if not any(action.dest == 'vae' for action in parser._actions):
        parser.add_argument("--vae", type=str, default=None, help="Path to Flux VAE checkpoint file")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to Qwen model folder")
    parser.add_argument("--lsunet_path", type=str, default=None, help="Path to LSUNet checkpoint file")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to prepend to captions")
    parser.add_argument("--text_projection", action="store_true", help="Force use of text projection layer")
    
    # Flux-like arguments
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0, help="Discrete flow shift for the Euler Discrete Scheduler")
    parser.add_argument("--model_prediction_type", choices=["raw", "additive", "sigma_scaled"], default="raw", help="How to interpret and process the model prediction. 'raw' predicts v (velocity) directly. 'sigma_scaled' predicts x0 but requires weighting_scheme='sigma_sqrt' to be equivalent to v-prediction.")
    parser.add_argument("--timestep_sampling", choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"], default="shift", help="Method to sample timesteps")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Scale factor for sigmoid timestep sampling")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for training (if applicable)")
    # parser.add_argument("--sample_prompts", type=str, default=None, help="Path to file with prompts to sample") # Conflict with train_util
    # parser.add_argument("--sample_every_n_steps", type=int, default=None, help="Sample every n steps") # Conflict with train_util
    # parser.add_argument("--sample_every_n_epochs", type=int, default=None, help="Sample every n epochs") # Conflict with train_util
    
    # Advanced Loss Weighting (SD3/Flux style)
    parser.add_argument("--weighting_scheme", type=str, default="none", choices=["none", "sigma_sqrt", "cosmap", "logit_normal", "mode"], help="Loss weighting scheme")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit-normal weighting")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Std for logit-normal weighting")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale for mode weighting")
    
    # Text Encoder Caching
    parser.add_argument("--cache_text_encoder_outputs", action="store_true", help="Cache text encoder outputs to memory")
    parser.add_argument("--cache_text_encoder_outputs_to_disk", action="store_true", help="Cache text encoder outputs to disk")
    parser.add_argument("--text_encoder_batch_size", type=int, default=1, help="Batch size for text encoder caching")
    
    # Optimizer
    parser.add_argument("--blockwise_fused_optimizers", action="store_true", help="Use blockwise fused optimizers")

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

def sample_images(
    accelerator,
    args,
    epoch,
    global_step,
    unet,
    vae,
    text_encoder,
    text_projection=None,
    sample_prompts_te_outputs=None,
):
    if not _should_sample(args, epoch, global_step):
        return

    if vae is None:
        logger.warning("VAE is not available. Skipping sampling.")
        return
    if text_encoder is None and not sample_prompts_te_outputs:
        logger.warning("Text encoder is not available. Skipping sampling.")
        return
    if not os.path.exists(args.sample_prompts):
        logger.warning(f"Sample prompts file not found: {args.sample_prompts}")
        return

    logger.info(f"Sampling images at step {global_step}...")

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    distributed_state = PartialState()

    unet = accelerator.unwrap_model(unet)
    if text_projection is not None:
        text_projection = accelerator.unwrap_model(text_projection)

    unet.eval()
    if text_projection is not None:
        text_projection.eval()

    te_original_device = _get_model_device(text_encoder) if text_encoder is not None else None
    vae_original_device = _get_model_device(vae)

    if text_encoder is not None and te_original_device != accelerator.device:
        text_encoder.to(accelerator.device)
    if vae_original_device != accelerator.device:
        vae.to(accelerator.device)
    vae.eval()

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    def _run_sampling(prompt_iterable):
        for prompt_dict in prompt_iterable:
            sample_image_inference(
                accelerator,
                args,
                unet,
                vae,
                text_encoder,
                text_projection,
                save_dir,
                prompt_dict,
                epoch,
                global_step,
                sample_prompts_te_outputs,
            )

    if distributed_state.num_processes <= 1:
        _run_sampling(prompts)
    else:
        per_process_prompts = [prompts[i::distributed_state.num_processes] for i in range(distributed_state.num_processes)]
        with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
            _run_sampling(prompt_dict_lists[0])

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    unet.train()
    if text_projection is not None:
        text_projection.train()

    if text_encoder is not None and te_original_device != accelerator.device:
        text_encoder.to(te_original_device)
    if vae_original_device != accelerator.device:
        vae.to(vae_original_device)

    clean_memory_on_device(accelerator.device)

def sample_image_inference(
    accelerator,
    args,
    unet,
    vae,
    text_encoder,
    text_projection,
    save_dir,
    prompt_dict,
    epoch,
    global_step,
    sample_prompts_te_outputs=None,
):
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt")
    width = prompt_dict.get("width", 1024)
    height = prompt_dict.get("height", 1024)
    sample_steps = prompt_dict.get("sample_steps", 20)
    scale = prompt_dict.get("scale", 1.0) # Guidance scale (CFG)
    seed = prompt_dict.get("seed")

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    do_cfg = scale > 1.0
    negative_prompt = negative_prompt or ""
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()
            
    logger.info(f"Prompt: {prompt}")
    if do_cfg:
        logger.info(f"Negative prompt: {negative_prompt}")
    logger.info(f"Height: {height}, Width: {width}, Steps: {sample_steps}, Scale: {scale}, Seed: {seed}")
    
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()

    def _to_tensor(data, device, dtype=None):
        if isinstance(data, torch.Tensor):
            tensor = data.detach().clone()
        else:
            tensor = torch.as_tensor(data)
        tensor = tensor.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    def _encode_prompt(text):
        cached = None
        if sample_prompts_te_outputs is not None and text in sample_prompts_te_outputs:
            cached = sample_prompts_te_outputs[text]
        if cached is not None:
            hidden_states = _to_tensor(cached[0], accelerator.device, unet.dtype)
            attention_mask = _to_tensor(cached[2], accelerator.device, torch.float32)
            return hidden_states, attention_mask
        if text_encoder is None:
            raise ValueError("Text encoder is required to encode prompts when no cached outputs are available.")
        tokens = tokenize_strategy.tokenize(text)
        input_ids = tokens[0].to(accelerator.device)
        attention_mask = tokens[1].to(accelerator.device).float()
        with torch.no_grad():
            hidden_states = text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state.to(unet.dtype)
        return hidden_states, attention_mask

    latents_override = None
    if "latents" in prompt_dict and prompt_dict["latents"] is not None:
        latents_override = prompt_dict["latents"]
    else:
        latents_path = prompt_dict.get("latents_path") or prompt_dict.get("latents_npz")
        if latents_path:
            latents_override = load_latents_from_path(latents_path)

    with torch.no_grad():
        prompt_hidden, prompt_mask = _encode_prompt(prompt)
        if do_cfg:
            neg_hidden, neg_mask = _encode_prompt(negative_prompt)
            encoder_hidden_states = torch.cat([neg_hidden, prompt_hidden], dim=0)
            attention_mask = torch.cat([neg_mask, prompt_mask], dim=0)
        else:
            encoder_hidden_states = prompt_hidden
            attention_mask = prompt_mask

        if text_projection is not None:
            encoder_hidden_states = text_projection(encoder_hidden_states)

        if do_cfg:
            encoder_hidden_states_uncond, encoder_hidden_states = torch.chunk(encoder_hidden_states, 2, dim=0)
            attention_mask_uncond, attention_mask = torch.chunk(attention_mask, 2, dim=0)
        else:
            encoder_hidden_states_uncond = None
            attention_mask_uncond = None

        # 2. Latents Initialization
        latents = prepare_initial_latents(
            height,
            width,
            unet.dtype,
            accelerator.device,
            seed=seed,
            latents=latents_override,
        )
        
        # 3. Scheduler: follow training timestep sampling strategy
        image_seq_len = (height // 16) * (width // 16)
        timesteps = build_inference_schedule(
            sample_steps,
            image_seq_len,
            timestep_sampling=getattr(args, "timestep_sampling", "shift"),
            discrete_flow_shift=getattr(args, "discrete_flow_shift", 3.0),
            sigmoid_scale=getattr(args, "sigmoid_scale", 1.0),
        )
        
        # 4. Denoise
        # timesteps is list[float], convert to tensor for loop if needed, but here we iterate
        
        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((latents.shape[0],), t_curr, dtype=unet.dtype, device=accelerator.device)
            t_input = t_vec * 1000.0

            if do_cfg:
                latents_input = torch.cat([latents, latents], dim=0)
                t_uncond = torch.cat([t_input, t_input], dim=0)
                context = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states], dim=0)
                context_mask = torch.cat([attention_mask_uncond, attention_mask], dim=0)
                model_pred = unet(latents_input, t_uncond, context, context_mask=context_mask)
                pred_uncond, pred_text = torch.chunk(model_pred, 2, dim=0)
                v_pred = pred_uncond + scale * (pred_text - pred_uncond)
            else:
                model_pred = unet(latents, t_input, encoder_hidden_states, context_mask=attention_mask)
                v_pred = model_pred

            if args.model_prediction_type == "sigma_scaled":
                sigma_val = max(t_curr, 1e-5)
                v_pred = (latents - v_pred) / sigma_val

            dt = t_next - t_curr
            latents = latents + dt * v_pred

        # 5. Decode
        if vae is not None:
            image = decode_latents_with_vae(vae, latents)
            
            # Post-process
            image = image.clamp(-1, 1)
            image = (image + 1) / 2
            image = image.permute(0, 2, 3, 1).float().cpu().numpy() # (B, H, W, C)
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

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _apply_discrete_flow_shift(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    shift = max(shift, 1.0)
    return (sigmas * shift) / (1 + (shift - 1) * sigmas)


def _normal_icdf(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(1e-6, 1 - 1e-6)
    return math.sqrt(2.0) * torch.erfinv(2 * u - 1)


def _compute_flux_mu(image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    return get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)

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
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def build_inference_schedule(
    num_steps: int,
    image_seq_len: int,
    timestep_sampling: str = "shift",
    discrete_flow_shift: float = 3.0,
    sigmoid_scale: float = 1.0,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> list[float]:
    """Create a deterministic inference schedule that mirrors the training sampler."""

    device = torch.device("cpu")
    eps = 1e-4
    fractions = torch.linspace(1 - eps, eps, num_steps + 1, device=device)

    if timestep_sampling == "uniform":
        timesteps = fractions
    elif timestep_sampling == "sigmoid":
        sigmas = torch.sigmoid(_normal_icdf(fractions) * sigmoid_scale)
        timesteps = sigmas
    elif timestep_sampling == "shift":
        sigmas = torch.sigmoid(_normal_icdf(fractions) * sigmoid_scale)
        timesteps = _apply_discrete_flow_shift(sigmas, discrete_flow_shift)
    elif timestep_sampling == "flux_shift":
        sigmas = torch.sigmoid(_normal_icdf(fractions) * sigmoid_scale)
        mu = _compute_flux_mu(image_seq_len, base_shift, max_shift)
        timesteps = time_shift(mu, 1.0, sigmas)
    else:
        mu = _compute_flux_mu(image_seq_len, base_shift, max_shift)
        timesteps = time_shift(mu, 1.0, fractions)

    return timesteps.clamp(0.0, 1.0).tolist()


def scale_timesteps_to_scheduler_range(timesteps: torch.Tensor, num_train_timesteps: int) -> torch.LongTensor:
    if timesteps.dtype.is_floating_point:
        scaled = torch.round(timesteps * (num_train_timesteps - 1))
    else:
        scaled = timesteps
    scaled = scaled.clamp(0, num_train_timesteps - 1)
    return scaled.to(dtype=torch.long)

def get_noisy_model_input_and_timesteps(args, noise, latents, device, dtype=torch.float32):
    bs, _, h, w = latents.shape
    image_seq_len = (h // 2) * (w // 2)

    # Timestep Sampling
    if args.timestep_sampling == "uniform":
        timesteps = torch.rand((bs,), device=device)
    elif args.timestep_sampling in {"sigmoid", "shift", "flux_shift"}:
        base = torch.randn((bs,), device=device)
        sigmas = torch.sigmoid(base * args.sigmoid_scale)
        if args.timestep_sampling == "sigmoid":
            timesteps = sigmas
        elif args.timestep_sampling == "shift":
            timesteps = _apply_discrete_flow_shift(sigmas, args.discrete_flow_shift)
        else:
            mu = _compute_flux_mu(image_seq_len)
            timesteps = time_shift(mu, 1.0, sigmas)
    else:
        # "sigma" or fallback
        if hasattr(args, "weighting_scheme") and args.weighting_scheme in ["logit_normal", "mode"]:
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bs,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            timesteps = u.to(device)
        else:
            timesteps = torch.rand((bs,), device=device)

    # Interpolate (Optimal Transport Path / Linear)
    # x_t = (1 - t) * x_0 + t * x_1
    # sigmas = t in this formulation
    sigmas = timesteps.view(bs, 1, 1, 1)
    
    if hasattr(args, "ip_noise_gamma") and args.ip_noise_gamma:
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if hasattr(args, "ip_noise_gamma_random_strength") and args.ip_noise_gamma_random_strength:
            ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
        else:
            ip_noise_gamma = args.ip_noise_gamma
        noisy_latents = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
    else:
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
    
    return noisy_latents.to(dtype), timesteps.to(dtype), sigmas

def apply_model_prediction_type(args, model_pred, noisy_latents, sigmas):
    weighting = None
    if args.model_prediction_type == "additive":
        # model_pred is additive noise
        model_pred = model_pred + noisy_latents
    elif args.model_prediction_type == "sigma_scaled":
        # model_pred is x0
        # In Flow Matching: x_t = (1-t)x_0 + t*x_1
        # v = x_1 - x_0
        # x_t = x_0 + t*v  => v = (x_t - x_0) / t
        # We want to convert x0_pred to v_pred to match the target (v)
        # v_pred = (noisy_latents - model_pred) / sigmas
        
        # Avoid division by zero
        sigmas = sigmas.clamp(min=1e-5)
        model_pred = (noisy_latents - model_pred) / sigmas
        
        # Apply weighting if specified (SD3 style)
        if hasattr(args, "weighting_scheme") and args.weighting_scheme != "none":
             weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)

    return model_pred, weighting
