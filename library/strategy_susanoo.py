import os
from typing import Any, List, Optional, Union
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from library import train_util
from library.strategy_base import TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy, LatentsCachingStrategy
import logging

logger = logging.getLogger(__name__)

class SusanooTokenizeStrategy(TokenizeStrategy):
    def __init__(self, tokenizer_path: str, max_length: int = 512, system_prompt: Optional[str] = None) -> None:
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.system_prompt = system_prompt
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        
        new_text = []
        for t in text:
            # Check if tokenizer has a chat template (e.g. Qwen-Chat)
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                try:
                    messages = []
                    if self.system_prompt:
                        messages.append({"role": "system", "content": self.system_prompt})
                    messages.append({"role": "user", "content": t})
                    
                    full_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=False, 
                        tokenize=False
                    )
                    new_text.append(full_prompt)
                    continue
                except Exception as e:
                    logger.warning(f"Failed to apply chat template: {e}")
                    # Fallback to manual formatting below
            
            # Fallback or no chat template
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} <Prompt Start> {t}"
                new_text.append(full_prompt)
            else:
                new_text.append(t)
        
        text = new_text
        
        tokens = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return [tokens.input_ids, tokens.attention_mask]

class SusanooTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        
        text_encoder = models[0]
        input_ids = tokens[0].to(text_encoder.device)
        attention_mask = tokens[1].to(text_encoder.device)
        
        with torch.no_grad():
            encoder_outputs = text_encoder(input_ids, attention_mask=attention_mask)
            last_hidden_state = encoder_outputs.last_hidden_state
            
        return [last_hidden_state, input_ids, attention_mask]

class SusanooTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    SUSANOO_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_susanoo_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + SusanooTextEncoderOutputsCachingStrategy.SUSANOO_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "last_hidden_state" not in npz:
                return False
            if "input_ids" not in npz:
                return False
            if "attention_mask" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        last_hidden_state = data["last_hidden_state"]
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        return [last_hidden_state, input_ids, attention_mask]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        captions = [info.caption for info in infos]
        tokens = tokenize_strategy.tokenize(captions)
        
        with torch.no_grad():
            last_hidden_state, input_ids, attention_mask = text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, tokens
            )

        if last_hidden_state.dtype == torch.bfloat16:
            last_hidden_state = last_hidden_state.float()
        
        last_hidden_state = last_hidden_state.cpu().numpy()
        input_ids = input_ids.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()

        for i, info in enumerate(infos):
            last_hidden_state_i = last_hidden_state[i]
            input_ids_i = input_ids[i]
            attention_mask_i = attention_mask[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    last_hidden_state=last_hidden_state_i,
                    input_ids=input_ids_i,
                    attention_mask=attention_mask_i,
                )
            else:
                info.text_encoder_outputs = [last_hidden_state_i, input_ids_i, attention_mask_i]

class SusanooLatentsCachingStrategy(LatentsCachingStrategy):
    SUSANOO_LATENTS_NPZ_SUFFIX = "_susanoo_latents.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return SusanooLatentsCachingStrategy.SUSANOO_LATENTS_NPZ_SUFFIX

    def get_image_size_from_disk_cache_path(self, absolute_path: str, npz_path: str) -> tuple[Optional[int], Optional[int]]:
        w, h = os.path.splitext(npz_path)[0].split("_")[-3].split("x")
        return int(w), int(h)

    def get_latents_npz_path(self, absolute_path: str, image_size: tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + SusanooLatentsCachingStrategy.SUSANOO_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        # Assuming downsampling factor of 8 for Flux VAE
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: tuple[int, int]
    ) -> tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
