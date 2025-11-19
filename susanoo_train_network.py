import argparse
from typing import List, Optional, Union, Any

import torch
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device
from diffusers import FlowMatchEulerDiscreteScheduler

init_ipex()

from library import susanoo_models, susanoo_utils, susanoo_train_utils, strategy_base, strategy_susanoo, train_util
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

class SusanooNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.text_projection = None

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        # super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        # train_network.NetworkTrainer doesn't have assert_extra_args, but subclasses do.
        # We should implement our own checks if needed.
        
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("Enabling cache_text_encoder_outputs due to disk caching")
            args.cache_text_encoder_outputs = True

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        if hasattr(text_encoder, "model") and hasattr(text_encoder.model, "embed_tokens"):
            text_encoder.model.embed_tokens.requires_grad_(True)
        elif hasattr(text_encoder, "embed_tokens"):
            text_encoder.embed_tokens.requires_grad_(True)
        elif hasattr(text_encoder, "get_input_embeddings"):
             text_encoder.get_input_embeddings().requires_grad_(True)
        else:
            logger.warning(f"Could not find embeddings for text encoder {index}, skipping gradient checkpointing workaround.")

    def load_target_model(self, args, weight_dtype, accelerator):
        # 1. LSUNet
        if args.lsunet_path:
            unet = susanoo_utils.load_lsunet(args.lsunet_path, weight_dtype, accelerator.device)
        else:
            unet = susanoo_utils.create_lsunet(weight_dtype, accelerator.device)
        
        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            
        # 2. VAE
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
        vae = susanoo_utils.load_vae(args.vae, vae_dtype, accelerator.device)
        
        # 3. Text Encoder
        text_encoder = susanoo_utils.load_text_encoder(args.text_encoder_path, weight_dtype, accelerator.device)
        
        # 4. Text Projection (Optional)
        text_enc_dim = text_encoder.config.hidden_size
        unet_context_dim = 1024 # Default for LSUNet
        
        text_projection = None
        if text_enc_dim != unet_context_dim or args.text_projection:
            text_projection = susanoo_utils.create_text_projection(text_enc_dim, unet_context_dim, weight_dtype, accelerator.device)
            if args.text_projection_path:
                 # Load weights if provided
                 tp_sd = susanoo_utils.load_checkpoint(args.text_projection_path, accelerator.device, weight_dtype)
                 if "linear.weight" in tp_sd:
                     text_projection.load_state_dict(tp_sd)

        # Return models
        self.text_projection = text_projection
        
        return "susanoo_v1", [text_encoder], vae, unet

    def encode_images_to_latents(self, args, vae, images):
        # Flux VAE returns latents directly
        return vae.encode(images)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Union[Any, None]:
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        return scheduler

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Get noisy latents and timesteps using Susanoo utils (Flow Matching logic)
        noisy_latents, timesteps, sigmas = susanoo_train_utils.get_noisy_model_input_and_timesteps(
            args, noise, latents, accelerator.device
        )

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_latents.requires_grad_(True)
            for t in text_encoder_conds:
                if t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Predict the noise residual
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            # LSUNet expects timesteps in 0-1000 range
            model_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        # Flow Matching Target
        if args.model_prediction_type == "sigma_scaled":
            # Target is x_0 (latents)
            target = latents
        else:
            # Target is v (noise - latents)
            target = noise - latents

        # Apply Model Prediction Type
        # If sigma_scaled, model_pred (v) is converted to x_0
        # If raw, model_pred (v) stays v
        model_pred, weighting = susanoo_train_utils.apply_model_prediction_type(args, model_pred, noisy_latents, sigmas)


        # We return None for weighting here because train_network.py handles loss calculation differently.
        # However, train_network.py calculates MSE(pred, target).
        # If we want weighted loss, we might need to adjust.
        # For now, we assume standard MSE on v-prediction is sufficient or weighting is handled by min_snr_gamma etc in base class.
        # But base class min_snr is for diffusion.
        # If we want to support "sigma_sqrt" etc, we should probably return it?
        # train_network.py doesn't accept weighting from here easily without modifying train loop.
        # But let's stick to basic Flow Matching loss for now.
        
        return model_pred, target, timesteps, None

    def get_tokenize_strategy(self, args):
        return strategy_susanoo.SusanooTokenizeStrategy(args.text_encoder_path, args.max_token_length or 77, args.system_prompt)

    def get_tokenizers(self, tokenize_strategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_susanoo.SusanooLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

    def get_text_encoding_strategy(self, args):
        return strategy_susanoo.SusanooTextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_susanoo.SusanooTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator: Accelerator,
        unet,
        vae,
        text_encoders,
        dataset,
        weight_dtype,
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                if self.text_projection:
                    self.text_projection.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoders to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # Cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")
                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                
                tokenize_strategy = self.get_tokenize_strategy(args)
                text_encoding_strategy = self.get_text_encoding_strategy(args)
                
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens = tokenize_strategy.tokenize(p)
                                out = text_encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens)
                                # out is [last_hidden_state, input_ids, attention_mask]
                                # We only need last_hidden_state for caching usually, but let's keep structure
                                sample_prompts_te_outputs[p] = out
                
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # move back to cpu
            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
                if self.text_projection:
                    self.text_projection.to(org_unet_device)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs" in batch:
            # Cached
            encoder_hidden_states = batch["text_encoder_outputs"][0].to(accelerator.device).to(weight_dtype)
            attention_mask = batch["text_encoder_outputs"][2].to(accelerator.device).to(weight_dtype)
        else:
            # Not cached
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            
            with torch.enable_grad():
                encoder_outputs = text_encoders[0](input_ids, attention_mask=attention_mask)
                encoder_hidden_states = encoder_outputs.last_hidden_state.to(weight_dtype)
        
        return encoder_hidden_states, attention_mask

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        indices=None,
    ):
        if len(text_conds) == 3:
            encoder_hidden_states, input_ids, attention_mask = text_conds
        else:
            encoder_hidden_states, attention_mask = text_conds
        
        # Apply projection if needed
        if self.text_projection is not None:
            # Ensure projection is on correct device/dtype
            self.text_projection.to(accelerator.device, dtype=weight_dtype)
            encoder_hidden_states = self.text_projection(encoder_hidden_states)

        # LSUNet expects timesteps in 0-1000 range
        model_pred = unet(noisy_latents, timesteps * 1000, encoder_hidden_states, context_mask=attention_mask)
        return model_pred

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
    ):
        susanoo_train_utils.sample_images(
            accelerator, args, epoch, global_step, unet, vae, text_encoder, self.text_projection
        )

def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    susanoo_train_utils.add_susanoo_train_arguments(parser)
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    trainer = SusanooNetworkTrainer()
    trainer.train(args)
