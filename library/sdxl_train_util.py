import argparse
import json
import math
import os
import random
import time
import toml
from typing import Optional

import torch
from PIL import Image
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import CLIPTokenizer
from library import model_util, sdxl_model_util, train_util, sdxl_original_unet, sdxl_ae_util, flux_utils
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# DEFAULT_NOISE_OFFSET = 0.0357


def load_target_model(args, accelerator, model_version: str, weight_dtype):
    model_dtype = match_mixed_precision(args, weight_dtype)  # prepare fp16/bf16
    use_flux_ae = getattr(args, "sdxl_ae", False) or getattr(args, "sdxl_ae_path", None) is not None
    flux_ae_path = getattr(args, "sdxl_ae_path", None)
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            (
                load_stable_diffusion_format,
                text_encoder1,
                text_encoder2,
                vae,
                unet,
                logit_scale,
                ckpt_info,
            ) = _load_target_model(
                args.pretrained_model_name_or_path,
                args.vae,
                model_version,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                model_dtype,
                args.disable_mmap_load_safetensors,
                use_flux_ae,
                flux_ae_path,
            )

            # work on low-ram device
            if args.lowram:
                text_encoder1.to(accelerator.device)
                text_encoder2.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def _load_target_model(
    name_or_path: str,
    vae_path: Optional[str],
    model_version: str,
    weight_dtype,
    device="cpu",
    model_dtype=None,
    disable_mmap=False,
    use_flux_ae: bool = False,
    flux_ae_path: Optional[str] = None,
):
    # model_dtype only work with full fp16/bf16
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers

    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        (
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_model_util.load_models_from_sdxl_checkpoint(
            model_version,
            name_or_path,
            device,
            model_dtype,
            disable_mmap,
            use_flux_ae,
            sdxl_ae_util.FLUX_VAE_LATENT_CHANNELS if use_flux_ae else None,
        )
    else:
        # Diffusers model is loaded to CPU
        from diffusers import StableDiffusionXLPipeline

        variant = "fp16" if weight_dtype == torch.float16 else None
        logger.info(f"load Diffusers pretrained models: {name_or_path}, variant={variant}")
        try:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    name_or_path, torch_dtype=model_dtype, variant=variant, tokenizer=None
                )
            except EnvironmentError as ex:
                if variant is not None:
                    logger.info("try to load fp32 model")
                    pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, variant=None, tokenizer=None)
                else:
                    raise ex
        except EnvironmentError as ex:
            logger.error(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex

        text_encoder1 = pipe.text_encoder
        text_encoder2 = pipe.text_encoder_2

        # convert to fp32 for cache text_encoders outputs
        if text_encoder1.dtype != torch.float32:
            text_encoder1 = text_encoder1.to(dtype=torch.float32)
        if text_encoder2.dtype != torch.float32:
            text_encoder2 = text_encoder2.to(dtype=torch.float32)

        vae = pipe.vae if not use_flux_ae else None
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        state_dict = sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl(unet.state_dict())
        if use_flux_ae:
            sdxl_ae_util.enable_flux_vae_unet_channels()
            state_dict = sdxl_ae_util.upgrade_unet_state_dict_for_flux(state_dict)
        with init_empty_weights():
            unet = sdxl_original_unet.SdxlUNet2DConditionModel()  # overwrite unet
        sdxl_model_util._load_state_dict_on_device(unet, state_dict, device=device, dtype=model_dtype)
        logger.info("U-Net converted to original U-Net")

        logit_scale = None
        ckpt_info = None

    # VAEを読み込む
    if use_flux_ae:
        target_flux_ae = flux_ae_path if flux_ae_path is not None else vae_path
        if target_flux_ae is not None:
            vae = flux_utils.load_ae(target_flux_ae, weight_dtype, device, disable_mmap=disable_mmap)
            logger.info("flux VAE loaded")
        elif vae is None:
            raise ValueError("Flux VAE weights are required for --sdxl_ae mode")
    elif vae_path is not None:
        vae = model_util.load_vae(vae_path, weight_dtype)
        logger.info("additional VAE loaded")

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def load_tokenizers(args: argparse.Namespace):
    logger.info("prepare tokenizers")

    original_paths = [TOKENIZER1_PATH, TOKENIZER2_PATH]
    tokeniers = []
    for i, original_path in enumerate(original_paths):
        tokenizer: CLIPTokenizer = None
        if args.tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(args.tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

        if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        if i == 1:
            tokenizer.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer

        tokeniers.append(tokenizer)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        logger.info(f"update token length: {args.max_token_length}")

    return tokeniers


def match_mixed_precision(args, weight_dtype):
    if args.full_fp16:
        assert (
            weight_dtype == torch.float16
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        return weight_dtype
    elif args.full_bf16:
        assert (
            weight_dtype == torch.bfloat16
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        return weight_dtype
    else:
        return None


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


def save_sd_model_on_train_end(
    args: argparse.Namespace,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(None, args, True, False, False, is_stable_diffusion_ckpt=True)
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    train_util.save_sd_model_on_train_end_common(
        args, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, diffusers_saver
    )


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(None, args, True, False, False, is_stable_diffusion_ckpt=True)
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        save_stable_diffusion_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
    )


def add_sdxl_training_arguments(parser: argparse.ArgumentParser, support_text_encoder_caching: bool = True):
    if support_text_encoder_caching:
        parser.add_argument(
            "--cache_text_encoder_outputs",
            action="store_true",
            help="cache text encoder outputs / text encoderの出力をキャッシュする",
        )
        parser.add_argument(
            "--cache_text_encoder_outputs_to_disk",
            action="store_true",
            help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
        )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment / safetensorsのmmapロードを無効にする。WSL環境等でモデル読み込みを高速化できる",
    )
    parser.add_argument(
        "--sdxl_ae",
        action="store_true",
        help="train SDXL with flux 16ch VAE branch / flux 16ch VAEを使ったSDXL分岐で学習する",
    )
    parser.add_argument(
        "--sdxl_ae_path",
        type=str,
        default=None,
        help="path to flux autoencoder weights (*.safetensors) used in --sdxl_ae mode / --sdxl_aeモードで使うflux VAEのパス",
    )


def verify_sdxl_training_args(args: argparse.Namespace, support_text_encoder_caching: bool = True):
    assert not args.v2, "v2 cannot be enabled in SDXL training / SDXL学習ではv2を有効にすることはできません"

    if args.clip_skip is not None:
        logger.warning("clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません")

    # if args.multires_noise_iterations:
    #     logger.info(
    #         f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET}, but noise_offset is disabled due to multires_noise_iterations / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されていますが、multires_noise_iterationsが有効になっているためnoise_offsetは無効になります"
    #     )
    # else:
    #     if args.noise_offset is None:
    #         args.noise_offset = DEFAULT_NOISE_OFFSET
    #     elif args.noise_offset != DEFAULT_NOISE_OFFSET:
    #         logger.info(
    #             f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET} / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されています"
    #         )
    #     logger.info(f"noise_offset is set to {args.noise_offset} / noise_offsetが{args.noise_offset}に設定されました")

    # assert (
    #     not hasattr(args, "weighted_captions") or not args.weighted_captions
    # ), "weighted_captions cannot be enabled in SDXL training currently / SDXL学習では今のところweighted_captionsを有効にすることはできません"

    if support_text_encoder_caching:
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
            logger.warning(
                "cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled / "
                + "cache_text_encoder_outputs_to_diskが有効になっているためcache_text_encoder_outputsが有効になりました"
            )

    if getattr(args, "sdxl_ae", False) or getattr(args, "sdxl_ae_path", None) is not None:
        if not getattr(args, "sdxl_ae", False):
            # auto enable when path is given
            args.sdxl_ae = True
        if args.save_model_as is not None and args.save_model_as.lower() == "diffusers":
            raise ValueError("--sdxl_ae mode cannot export Diffusers format; use ckpt/safetensors")
        if args.sdxl_ae_path is None and args.vae is None:
            logger.warning("--sdxl_ae enabled but no flux VAE path supplied; ensure checkpoint already contains flux_vae.* weights")


def sample_images(*args, **kwargs):
    # args layout matches train_util.sample_images_common(accelerator, args, ...)
    if len(args) > 0 and getattr(args[1], "sdxl_ae", False):
        return _sample_images_flux(*args, **kwargs)
    from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline

    return train_util.sample_images_common(SdxlStableDiffusionLongPromptWeightingPipeline, *args, **kwargs)


def _should_sample(args: argparse.Namespace, epoch: int, steps: int) -> bool:
    if steps == 0:
        return bool(args.sample_at_first)

    if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
        return False
    if args.sample_every_n_epochs is not None:
        if epoch is None or epoch % args.sample_every_n_epochs != 0:
            return False
    else:
        if steps % args.sample_every_n_steps != 0 or epoch is not None:
            return False
    return True


def _load_sample_prompts(sample_prompts_path: str):
    if sample_prompts_path.endswith(".txt"):
        with open(sample_prompts_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif sample_prompts_path.endswith(".toml"):
        with open(sample_prompts_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif sample_prompts_path.endswith(".json"):
        with open(sample_prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    else:
        raise ValueError(f"unsupported sample prompts file: {sample_prompts_path}")

    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = train_util.line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def _encode_text_sdxl(tokenizers, text_encoders, device, dtype, prompt: str, prompt2: str):
    tokenizer1, tokenizer2 = tokenizers
    text_encoder1, text_encoder2 = text_encoders

    batch_encoding = tokenizer1(
        prompt,
        truncation=True,
        return_length=True,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    tokens1 = batch_encoding["input_ids"].to(device)

    with torch.no_grad():
        enc_out1 = text_encoder1(tokens1, output_hidden_states=True, return_dict=True)
        text_embedding1 = enc_out1["hidden_states"][11]

    tokens2 = tokenizer2(
        prompt2,
        truncation=True,
        return_length=True,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"].to(device)

    with torch.no_grad():
        enc_out2 = text_encoder2(tokens2, output_hidden_states=True, return_dict=True)
        text_embedding2_penu = enc_out2["hidden_states"][-2]
        text_embedding2_pool = enc_out2["text_embeds"]

    text_embedding = torch.cat([text_embedding1, text_embedding2_penu], dim=2).to(device=device, dtype=dtype)
    text_embedding2_pool = text_embedding2_pool.to(device=device, dtype=dtype)
    return text_embedding, text_embedding2_pool


def _decode_flux_images(vae, latents, vae_dtype, device):
    latents = latents.to(device=device, dtype=vae_dtype)
    image = vae.decode(latents)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    return [Image.fromarray(im) for im in image]


def _sample_images_flux(accelerator, args, epoch, steps, device, vae, tokenizers, text_encoders, unet, *_, **__):
    if not _should_sample(args, epoch, steps):
        return None

    if not os.path.isfile(args.sample_prompts):
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return None

    prompts = _load_sample_prompts(args.sample_prompts)
    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")

    # unwrap models and move to device
    unet = accelerator.unwrap_model(unet)
    text_encoders = [accelerator.unwrap_model(te) for te in text_encoders]

    # remember devices to restore later
    org_vae_device = vae.device
    org_te_devices = [te.device for te in text_encoders]

    vae_dtype = torch.float32 if getattr(args, "no_half_vae", False) else unet.dtype
    vae.to(device, dtype=vae_dtype)

    orig_unet_mode = unet.training
    orig_te_modes = [te.training for te in text_encoders]

    for te in text_encoders:
        te.to(device, dtype=unet.dtype)
        te.eval()
    unet.to(device)
    unet.eval()

    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # save random state
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    for prompt_dict in prompts:
        assert isinstance(prompt_dict, dict)
        prompt = prompt_dict.get("prompt", "")
        prompt2 = prompt_dict.get("prompt2", prompt)
        negative_prompt = prompt_dict.get("negative_prompt", "")
        sample_steps = prompt_dict.get("sample_steps", 30)
        sampler_name = prompt_dict.get("sample_sampler", args.sample_sampler)
        scale = prompt_dict.get("scale", prompt_dict.get("guidance_scale", 7.5))
        seed = prompt_dict.get("seed")

        height = prompt_dict.get("height", 1024)
        width = prompt_dict.get("width", 1024)
        height = max(64, height - height % 8)
        width = max(64, width - width % 8)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        scheduler = train_util.get_my_scheduler(
            sample_sampler=sampler_name,
            v_parameterization=args.v_parameterization,
        )
        scheduler.set_timesteps(sample_steps, device)

        # encode text
        c_ctx, c_ctx_pool = _encode_text_sdxl(tokenizers, text_encoders, device, unet.dtype, prompt, prompt2)
        uc_ctx, uc_ctx_pool = _encode_text_sdxl(tokenizers, text_encoders, device, unet.dtype, negative_prompt, negative_prompt)

        size_embeddings = get_size_embeddings(
            torch.tensor([[height, width]], device=device),
            torch.tensor([[0, 0]], device=device),
            torch.tensor([[height, width]], device=device),
            device,
        ).to(dtype=unet.dtype)

        c_vector = torch.cat([c_ctx_pool, size_embeddings], dim=1)
        uc_vector = torch.cat([uc_ctx_pool, size_embeddings], dim=1)

        text_embeddings = torch.cat([uc_ctx, c_ctx])
        vector_embeddings = torch.cat([uc_vector, c_vector])

        latent_channels = sdxl_ae_util.FLUX_VAE_LATENT_CHANNELS
        latent_scale = sdxl_ae_util.FLUX_VAE_LATENT_MULT
        latents = torch.randn(
            (1, latent_channels, height // 8, width // 8),
            device=device,
            dtype=unet.dtype,
        )
        latents = (latents * scheduler.init_noise_sigma).to(unet.dtype)

        timesteps = scheduler.timesteps.to(device)
        with torch.no_grad():
            for t in timesteps:
            latent_model_input = latents.repeat((2, 1, 1, 1))
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(unet.dtype)

            t_in = t.to(unet.dtype)
            te = text_embeddings.to(unet.dtype)
            ve = vector_embeddings.to(unet.dtype)

            noise_pred = unet(latent_model_input, t_in, te, ve)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = latents / latent_scale

        images = _decode_flux_images(vae, latents, vae_dtype, device)

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        i = prompt_dict.get("enum", 0)
        img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
        images[0].save(os.path.join(save_dir, img_filename))

        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            wandb_tracker = accelerator.get_tracker("wandb")
            import wandb

            wandb_tracker.log({f"sample_{i}": wandb.Image(images[0], caption=prompt)}, commit=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # restore RNG and devices
    torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    vae.to(org_vae_device)
    for te, org_dev, mode in zip(text_encoders, org_te_devices, orig_te_modes):
        te.to(org_dev)
        te.train(mode)
    unet.train(orig_unet_mode)

    clean_memory_on_device(device)

    return None
