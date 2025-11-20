import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml
from tqdm import tqdm
import torch
import torch.nn as nn
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from library import deepspeed_utils, strategy_base, strategy_susanoo
import library.train_util as train_util
from library.utils import setup_logging, add_logging_arguments
from library.config_util import ConfigSanitizer, BlueprintGenerator
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions
from safetensors.torch import save_file

# Import Susanoo libraries
from library import susanoo_utils, susanoo_train_utils

setup_logging()
import logging
logger = logging.getLogger(__name__)

def train(args):
    # train_util.verify_training_args(args) # Skip this as it checks for v2/sdxl specific args
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    cache_latents = args.cache_latents
    
    if args.seed is not None:
        set_seed(args.seed)

    # Prepare Latent Caching Strategy
    if args.cache_latents:
        latents_caching_strategy = strategy_susanoo.SusanooLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # Tokenizer Strategy
    susanoo_tokenize_strategy = strategy_susanoo.SusanooTokenizeStrategy(args.text_encoder_path, args.max_token_length or 512, args.system_prompt)
    strategy_base.TokenizeStrategy.set_strategy(susanoo_tokenize_strategy)

    # Tokenizer (for other uses if needed, though strategy handles it)
    tokenizer = susanoo_tokenize_strategy.tokenizer
    
    # Dataset
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            user_config = config_util.load_user_config(args.dataset_config)
        else:
            user_config = {
                "datasets": [
                    {
                        "subsets": [
                            {
                                "image_dir": args.train_data_dir,
                                "metadata_file": args.in_json,
                            }
                        ]
                    }
                ]
            }
        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.set_current_strategies()
    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return

    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # Cache Latents if requested
    if args.cache_latents:
        logger.info("Loading Flux VAE for latent caching...")
        vae = susanoo_utils.load_vae(args.vae, vae_dtype, accelerator.device)
        
        train_dataset_group.new_cache_latents(vae, accelerator)
        
        # Unload VAE to free memory
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        del vae

    # Text Encoding Strategy
    text_encoding_strategy = strategy_susanoo.SusanooTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # Load Models
    # 1. Text Encoder (Qwen)
    text_encoder = None
    if args.cache_text_encoder_outputs:
        # Load for caching
        text_encoder = susanoo_utils.load_text_encoder(args.text_encoder_path, weight_dtype, accelerator.device)
        
        # Setup Caching Strategy
        text_encoder_caching_strategy = strategy_susanoo.SusanooTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)
        
        # Run Caching
        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([text_encoder], accelerator)
            
        # Cache sample prompts
        if args.sample_prompts is not None:
            logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")
            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")
                            input_ids, attention_mask = susanoo_tokenize_strategy.tokenize(p)
                            input_ids = input_ids.to(accelerator.device)
                            out = text_encoder(input_ids).last_hidden_state.to("cpu").to(weight_dtype)
                            sample_prompts_te_outputs[p] = out

        # Unload
        text_encoder.to("cpu")
        clean_memory_on_device(accelerator.device)
        del text_encoder
        text_encoder = None
    else:
        # Load for training
        text_encoder = susanoo_utils.load_text_encoder(args.text_encoder_path, weight_dtype, accelerator.device)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        sample_prompts_te_outputs = None

    # 2. VAE (Flux)
    vae = None
    if not args.cache_latents:
        vae = susanoo_utils.load_vae(args.vae, vae_dtype, accelerator.device)
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.eval()

    # 3. UNet (LSUNet)
    if args.lsunet_path:
        unet = susanoo_utils.load_lsunet(args.lsunet_path, weight_dtype, accelerator.device)
    else:
        unet = susanoo_utils.create_lsunet(weight_dtype, accelerator.device)
    
    if args.gradient_checkpointing:
        # unet.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)
        unet.enable_gradient_checkpointing()

    # 4. Text Projection (Optional/Fallback)
    # Check dimensions
    if text_encoder is not None:
        text_enc_dim = text_encoder.config.hidden_size
    else:
        # Assume Qwen 0.5B hidden size if cached and encoder is gone
        text_enc_dim = 1024 # Qwen1.5-0.5B hidden size

    # LSUNet context dim is hardcoded to 1024 in susanoo_models.py, but let's assume we might need projection
    # If user wants to force projection or if dims mismatch
    unet_context_dim = 1024 # Default for LSUNet
    
    text_projection = None
    if text_enc_dim != unet_context_dim or args.text_projection:
        text_projection = susanoo_utils.create_text_projection(text_enc_dim, unet_context_dim, weight_dtype, accelerator.device)

    # Optimizer
    trainable_params = list(unet.parameters())
    if text_projection is not None:
        trainable_params += list(text_projection.parameters())
        
    if args.blockwise_fused_optimizers:
        # Group parameters for blockwise optimization
        grouped_params = []
        param_group = {}
        
        # Helper to add params to group
        def add_to_group(name, param):
            # Determine block type and index
            if name.startswith("input_blocks"):
                # input_blocks.0.0...
                parts = name.split(".")
                block_idx = int(parts[1])
                key = f"input_blocks_{block_idx}"
            elif name.startswith("middle_block"):
                key = "middle_block"
            elif name.startswith("output_blocks"):
                parts = name.split(".")
                block_idx = int(parts[1])
                key = f"output_blocks_{block_idx}"
            else:
                key = "other"
            
            if key not in param_group:
                param_group[key] = []
            param_group[key].append(param)

        for name, param in unet.named_parameters():
            if param.requires_grad:
                add_to_group(name, param)
        
        if text_projection is not None:
            for name, param in text_projection.named_parameters():
                if param.requires_grad:
                    add_to_group(f"text_projection.{name}", param)

        # Create optimizer groups
        for key in sorted(param_group.keys()):
            grouped_params.append({"params": param_group[key], "lr": args.learning_rate})
            num_params = sum(p.numel() for p in param_group[key])
            accelerator.print(f"Block {key}: {num_params} parameters")
            
        optimizers = []
        for group in grouped_params:
            _, _, optimizer = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(optimizer)
        optimizer = optimizers[0] # Placeholder for compatibility
        
        logger.info(f"Using {len(optimizers)} optimizers for blockwise fused optimizers")
        
        # We need custom train/eval functions to handle multiple optimizers
        def optimizer_train_fn():
            for opt in optimizers:
                if hasattr(opt, "train"):
                    opt.train()
                    
        def optimizer_eval_fn():
            for opt in optimizers:
                if hasattr(opt, "eval"):
                    opt.eval()
    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=[{"params": trainable_params, "lr": args.learning_rate}])
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

    # Dataloader
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group, batch_size=1, shuffle=True, collate_fn=collator, num_workers=n_workers, persistent_workers=args.persistent_data_loader_workers
    )

    # Scheduler (Flow Matching)
    # Initialize scheduler with discrete_flow_shift
    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)

    # Training Loop
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("susanoo_train", config=vars(args))

    if text_projection is not None:
        unet, text_projection, optimizer, train_dataloader = accelerator.prepare(unet, text_projection, optimizer, train_dataloader)
    else:
        unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Resume from checkpoint
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    loss_recorder = train_util.LossRecorder()
    
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1
        unet.train()
        if text_projection is not None:
            text_projection.train()
        
        optimizer_train_fn()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(unet):
                # 1. Encode Images to Latents
                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        imgs = batch["images"].to(vae_dtype).to(accelerator.device)
                        # Flux VAE Encode
                        # encode returns sampled latents, we need to scale/shift
                        # latents = (latents - shift_factor) * scale_factor
                        # ae_params: scale_factor=0.3611, shift_factor=0.1159
                        
                        # Note: flux_models.AutoEncoder.encode returns posterior.sample()
                        # And it ALREADY applies scaling/shifting: z = scale_factor * (z - shift_factor)
                        latents = vae.encode(imgs)
                        
                        # NaN check (copied from lumina_train.py)
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                            
                        latents = latents.to(weight_dtype)

                # 2. Encode Text
                with torch.no_grad():
                    if args.cache_text_encoder_outputs:
                        encoder_hidden_states = batch["text_encoder_outputs"][0].to(accelerator.device).to(weight_dtype)
                        # pooled_output = batch["text_encoder_outputs"][1].to(accelerator.device).to(weight_dtype) # Not used
                        attention_mask = batch["text_encoder_outputs"][2].to(accelerator.device).to(weight_dtype)
                    else:
                        captions = batch["captions"]
                        # Tokenization is handled by strategy (including system prompt)
                        input_ids, attention_mask = susanoo_tokenize_strategy.tokenize(captions)
                        input_ids = input_ids.to(accelerator.device)
                        attention_mask = attention_mask.to(accelerator.device).float()
                        
                        encoder_hidden_states = text_encoder(input_ids).last_hidden_state.to(weight_dtype)
                        
                        # No pooled output

                # Apply Projection if needed
                if text_projection is not None:
                    encoder_hidden_states = text_projection(encoder_hidden_states)
                    pass

                # 5. Sample Noise and Timesteps (Flow Matching)
                noise = torch.randn_like(latents)
                
                noisy_latents, timesteps, sigmas = susanoo_train_utils.get_noisy_model_input_and_timesteps(args, noise, latents, accelerator.device)
                
                if args.gradient_checkpointing:
                    noisy_latents.requires_grad_(True)

                # Velocity Target
                # v = dx_t/dt = -x_0 + x_1 = noise - latents
                # if args.model_prediction_type == "sigma_scaled":
                #     target = latents
                # else:
                #     target = noise - latents
                target = noise - latents

                # 6. Predict
                # LSUNet expects timesteps in 0-1000 range for embedding lookup (SDXL style)
                # t=0 -> 0 (Data), t=1 -> 1000 (Noise)
                with accelerator.autocast():
                    model_pred = unet(noisy_latents, timesteps * 1000, encoder_hidden_states, context_mask=attention_mask)

                # Apply Model Prediction Type
                model_pred, weighting = susanoo_train_utils.apply_model_prediction_type(args, model_pred, noisy_latents, sigmas)
                
                # 7. Loss
                target = target.float()
                model_pred = model_pred.float()
                
                if weighting is not None:
                    loss = torch.mean(weighting.float() * (model_pred - target) ** 2)
                else:
                    loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

                accelerator.backward(loss)
                if args.blockwise_fused_optimizers:
                    for opt in optimizers:
                        opt.step()
                        opt.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                
                optimizer_eval_fn()
                
                # Sample Images
                susanoo_train_utils.sample_images(accelerator, args, epoch + 1, global_step, unet, vae, text_encoder, text_projection)
                
                optimizer_train_fn()

                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        metadata = {
                            "ss_base_model_version": "susanoo_v1",
                            "ss_session_id": args.session_id,
                        }
                        susanoo_train_utils.save_susanoo_model(args, epoch + 1, global_step, accelerator, unet, text_projection, save_dtype=save_dtype, metadata=metadata)
            
            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                # train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=True) # TODO: Add LR scheduler support
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None and (epoch + 1) % args.save_every_n_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                metadata = {
                    "ss_base_model_version": "susanoo_v1",
                    "ss_session_id": args.session_id,
                }
                susanoo_train_utils.save_susanoo_model(args, epoch + 1, global_step, accelerator, unet, text_projection, save_dtype=save_dtype, metadata=metadata)
        
        optimizer_eval_fn()
        susanoo_train_utils.sample_images(accelerator, args, epoch + 1, global_step, unet, vae, text_encoder, text_projection)
        optimizer_train_fn()

    accelerator.end_training()
    
    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    if accelerator.is_main_process:
        metadata = {
            "ss_base_model_version": "susanoo_v1",
            "ss_session_id": args.session_id,
        }
        susanoo_train_utils.save_susanoo_model_on_train_end(args, save_dtype, epoch, global_step, unet, text_projection)
        logger.info("model saved.")

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    
    susanoo_train_utils.add_susanoo_train_arguments(parser)
    
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train(args)
