import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from library.strategy_base import TokenizeStrategy, TextEncodingStrategy, TextEncoderOutputsCachingStrategy


from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


TOKENIZER1_PATH = "openai/clip-vit-large-patch14"
TOKENIZER2_PATH = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"


class SdxlTokenizeStrategy(TokenizeStrategy):
    def __init__(self, max_length: Optional[int], tokenizer_cache_dir: Optional[str] = None) -> None:
        self.tokenizer1 = self._load_tokenizer(CLIPTokenizer, TOKENIZER1_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
        self.tokenizer2 = self._load_tokenizer(CLIPTokenizer, TOKENIZER2_PATH, tokenizer_cache_dir=tokenizer_cache_dir)
        self.tokenizer2.pad_token_id = 0  # use 0 as pad token for tokenizer2

        if max_length is None:
            self.max_length = self.tokenizer1.model_max_length
        else:
            self.max_length = max_length + 2

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        return (
            torch.stack([self._get_input_ids(self.tokenizer1, t, self.max_length) for t in text], dim=0),
            torch.stack([self._get_input_ids(self.tokenizer2, t, self.max_length) for t in text], dim=0),
        )

    def tokenize_with_weights(self, text: str | List[str]) -> Tuple[List[torch.Tensor]]:
        text = [text] if isinstance(text, str) else text
        tokens1_list, tokens2_list = [], []
        weights1_list, weights2_list = [], []
        for t in text:
            tokens1, weights1 = self._get_input_ids(self.tokenizer1, t, self.max_length, weighted=True)
            tokens2, weights2 = self._get_input_ids(self.tokenizer2, t, self.max_length, weighted=True)
            tokens1_list.append(tokens1)
            tokens2_list.append(tokens2)
            weights1_list.append(weights1)
            weights2_list.append(weights2)
        return [torch.stack(tokens1_list, dim=0), torch.stack(tokens2_list, dim=0)], [
            torch.stack(weights1_list, dim=0),
            torch.stack(weights2_list, dim=0),
        ]


class LlmTokenizeStrategy(TokenizeStrategy):
    def __init__(self, tokenizer, max_length: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        if max_length is None:
            self.max_length = getattr(tokenizer, 'model_max_length', 256)
        else:
            self.max_length = max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        tokens = []
        for t in text:
            input_ids = self._get_input_ids(self.tokenizer, t, self.max_length)
            tokens.append(input_ids)
        return [torch.stack(tokens, dim=0)]

    def tokenize_with_weights(self, text: str | List[str]) -> Tuple[List[torch.Tensor]]:
        # For LLM, we only have one tokenizer, so return single list
        text = [text] if isinstance(text, str) else text
        tokens_list = []
        weights_list = []
        for t in text:
            # Use simple tokenization for now - TODO: implement proper weighted parsing for LLM
            input_ids = self.tokenizer(t, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt").input_ids
            tokens_list.append(input_ids.squeeze(0))
            # Create uniform weights for now
            weights = torch.ones_like(input_ids.squeeze(0), dtype=torch.float32)
            weights_list.append(weights)
        return [torch.stack(tokens_list, dim=0)], [torch.stack(weights_list, dim=0)]


class SdxlTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def _pool_workaround(
        self, text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
    ):
        r"""
        workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
        instead of the hidden states for the EOS token
        If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

        Original code from CLIP's pooling function:

        \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
        \# take features from the eot embedding (eot_token is the highest number in each sequence)
        \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        """

        # input_ids: b*n,77
        # find index for EOS token

        # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
        # eos_token_index = torch.where(input_ids == eos_token_id)[1]
        # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

        # Create a mask where the EOS tokens are
        eos_token_mask = (input_ids == eos_token_id).int()

        # Use argmax to find the last index of the EOS token for each element in the batch
        eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
        eos_token_index = eos_token_index.to(device=last_hidden_state.device)

        # get hidden states for EOS token
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index
        ]

        # apply projection: projection may be of different dtype than last_hidden_state
        pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
        pooled_output = pooled_output.to(last_hidden_state.dtype)

        return pooled_output

    def _get_hidden_states_sdxl(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        text_encoder1: Union[CLIPTextModel, torch.nn.Module],
        text_encoder2: Union[CLIPTextModelWithProjection, torch.nn.Module],
        unwrapped_text_encoder2: Optional[CLIPTextModelWithProjection] = None,
    ):
        # input_ids: b,n,77 -> b*n, 77
        b_size = input_ids1.size()[0]
        if input_ids1.size()[1] == 1:
            max_token_length = None
        else:
            max_token_length = input_ids1.size()[1] * input_ids1.size()[2]
        input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
        input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77
        input_ids1 = input_ids1.to(text_encoder1.device)
        input_ids2 = input_ids2.to(text_encoder2.device)

        # text_encoder1
        enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
        hidden_states1 = enc_out["hidden_states"][11]

        # text_encoder2
        enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
        hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

        # pool2 = enc_out["text_embeds"]
        unwrapped_text_encoder2 = unwrapped_text_encoder2 or text_encoder2
        pool2 = self._pool_workaround(unwrapped_text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

        # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
        n_size = 1 if max_token_length is None else max_token_length // 75
        hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
        hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

        if max_token_length is not None:
            # bs*3, 77, 768 or 1024
            # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer1.model_max_length):
                states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
            states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
            hidden_states1 = torch.cat(states_list, dim=1)

            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer2.model_max_length):
                chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
                # this causes an error:
                # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # if i > 1:
                #     for j in range(len(chunk)):  # batch_size
                #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            hidden_states2 = torch.cat(states_list, dim=1)

            # pool はnの最初のものを使う
            pool2 = pool2[::n_size]

        return hidden_states1, hidden_states2, pool2

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            tokenize_strategy: TokenizeStrategy
            models: List of models, [text_encoder1, text_encoder2, unwrapped text_encoder2 (optional)].
                If text_encoder2 is wrapped by accelerate, unwrapped_text_encoder2 is required
            tokens: List of tokens, for text_encoder1 and text_encoder2
        """
        if len(models) == 2:
            text_encoder1, text_encoder2 = models
            unwrapped_text_encoder2 = None
        else:
            text_encoder1, text_encoder2, unwrapped_text_encoder2 = models
        tokens1, tokens2 = tokens
        sdxl_tokenize_strategy = tokenize_strategy  # type: SdxlTokenizeStrategy
        tokenizer1, tokenizer2 = sdxl_tokenize_strategy.tokenizer1, sdxl_tokenize_strategy.tokenizer2

        hidden_states1, hidden_states2, pool2 = self._get_hidden_states_sdxl(
            tokens1, tokens2, tokenizer1, tokenizer2, text_encoder1, text_encoder2, unwrapped_text_encoder2
        )
        return [hidden_states1, hidden_states2, pool2]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        hidden_states1, hidden_states2, pool2 = self.encode_tokens(tokenize_strategy, models, tokens_list)

        weights_list = [weights.to(hidden_states1.device) for weights in weights_list]

        # apply weights
        if weights_list[0].shape[1] == 1:  # no max_token_length
            # weights: ((b, 1, 77), (b, 1, 77)), hidden_states: (b, 77, 768), (b, 77, 768)
            hidden_states1 = hidden_states1 * weights_list[0].squeeze(1).unsqueeze(2)
            hidden_states2 = hidden_states2 * weights_list[1].squeeze(1).unsqueeze(2)
        else:
            # weights: ((b, n, 77), (b, n, 77)), hidden_states: (b, n*75+2, 768), (b, n*75+2, 768)
            for weight, hidden_states in zip(weights_list, [hidden_states1, hidden_states2]):
                for i in range(weight.shape[1]):
                    hidden_states[:, i * 75 + 1 : i * 75 + 76] = hidden_states[:, i * 75 + 1 : i * 75 + 76] * weight[
                        :, i, 1:-1
                    ].unsqueeze(-1)

        return [hidden_states1, hidden_states2, pool2]


class SdxlTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    SDXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_te_outputs.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        is_weighted: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial, is_weighted)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + SdxlTextEncoderOutputsCachingStrategy.SDXL_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "hidden_state1" not in npz or "hidden_state2" not in npz or "pool2" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        hidden_state1 = data["hidden_state1"]
        hidden_state2 = data["hidden_state2"]
        pool2 = data["pool2"]
        return [hidden_state1, hidden_state2, pool2]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        sdxl_text_encoding_strategy = text_encoding_strategy  # type: SdxlTextEncodingStrategy
        captions = [info.caption for info in infos]

        if self.is_weighted:
            tokens_list, weights_list = tokenize_strategy.tokenize_with_weights(captions)
            with torch.no_grad():
                hidden_state1, hidden_state2, pool2 = sdxl_text_encoding_strategy.encode_tokens_with_weights(
                    tokenize_strategy, models, tokens_list, weights_list
                )
        else:
            tokens1, tokens2 = tokenize_strategy.tokenize(captions)
            with torch.no_grad():
                hidden_state1, hidden_state2, pool2 = sdxl_text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, [tokens1, tokens2]
                )

        if hidden_state1.dtype == torch.bfloat16:
            hidden_state1 = hidden_state1.float()
        if hidden_state2.dtype == torch.bfloat16:
            hidden_state2 = hidden_state2.float()
        if pool2.dtype == torch.bfloat16:
            pool2 = pool2.float()

        hidden_state1 = hidden_state1.cpu().numpy()
        hidden_state2 = hidden_state2.cpu().numpy()
        pool2 = pool2.cpu().numpy()

        for i, info in enumerate(infos):
            hidden_state1_i = hidden_state1[i]
            hidden_state2_i = hidden_state2[i]
            pool2_i = pool2[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_state1=hidden_state1_i,
                    hidden_state2=hidden_state2_i,
                    pool2=pool2_i,
                )
            else:
                info.text_encoder_outputs = [hidden_state1_i, hidden_state2_i, pool2_i]


class LlmTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    LLM_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_llm_te_outputs.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        is_weighted: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial, is_weighted)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + LlmTextEncoderOutputsCachingStrategy.LLM_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "hidden_states" not in npz or "pooled_output" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        hidden_states = data["hidden_states"]
        pooled_output = data["pooled_output"]
        return [hidden_states, pooled_output]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        llm_text_encoding_strategy = text_encoding_strategy  # type: LlmTextEncodingStrategy
        captions = [info.caption for info in infos]

        if self.is_weighted:
            # For LLM, weights are handled in the encoding strategy
            tokens_list = tokenize_strategy.tokenize(captions)
            weights_list = None  # LLM handles weights internally if needed
            with torch.no_grad():
                hidden_states, pooled_output = llm_text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens_list
                )
        else:
            tokens = tokenize_strategy.tokenize(captions)
            with torch.no_grad():
                hidden_states, pooled_output = llm_text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens
                )

        if hidden_states.dtype == torch.bfloat16:
            hidden_states = hidden_states.float()
        if pooled_output.dtype == torch.bfloat16:
            pooled_output = pooled_output.float()

        hidden_states = hidden_states.cpu().numpy()
        pooled_output = pooled_output.cpu().numpy()

        for i, info in enumerate(infos):
            hidden_states_i = hidden_states[i]
            pooled_output_i = pooled_output[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_states=hidden_states_i,
                    pooled_output=pooled_output_i,
                )
            else:
                info.text_encoder_outputs = [hidden_states_i, pooled_output_i]


class LlmTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, projection=None, system_prompt=None, text_encoder=None):
        """
        Args:
            projection: Optional projection layer. If None and text_encoder is provided,
                       automatically creates separate projections for each SDXL encoder
            system_prompt: Optional system prompt to prepend to user prompts
            text_encoder: LLM text encoder model to auto-detect hidden size for projection
        """
        self.system_prompt = system_prompt or ""

        # Handle different projection formats
        if projection is not None:
            # Manual projection provided - use legacy single projection mode
            self.projection = projection
            self.projection1 = None
            self.projection2 = None
        elif text_encoder is not None:
            # Auto-create separate projections for each encoder
            self.projection = None  # Legacy compatibility
            self._create_auto_projection(text_encoder)
        else:
            # No projection provided
            self.projection = None
            self.projection1 = None
            self.projection2 = None

    def _create_auto_projection(self, text_encoder):
        """Automatically create projection from LLM hidden size to SDXL cross_attention_dim"""
        llm_hidden_size = text_encoder.config.hidden_size
        target_hidden_size = 2048  # SDXL cross_attention_dim

        # Create separate projections for each SDXL encoder
        sdxl_encoder1_dim = 768   # CLIP ViT-L/14
        sdxl_encoder2_dim = 1280  # CLIP ViT-G/14

        # Create projection layers for each encoder
        self.projection1 = nn.Linear(llm_hidden_size, sdxl_encoder1_dim, bias=False)
        self.projection2 = nn.Linear(llm_hidden_size, sdxl_encoder2_dim, bias=False)

        # Initialize weights
        with torch.no_grad():
            # For projection1 (LLM -> 768): use first part of identity or orthogonal
            if llm_hidden_size >= sdxl_encoder1_dim:
                self.projection1.weight.data = torch.eye(sdxl_encoder1_dim, llm_hidden_size)
            else:
                torch.nn.init.orthogonal_(self.projection1.weight)

            # For projection2 (LLM -> 1280): use orthogonal initialization
            torch.nn.init.orthogonal_(self.projection2.weight)

        # Move to same device and dtype as text_encoder
        self.projection1 = self.projection1.to(text_encoder.device, dtype=text_encoder.dtype)
        self.projection2 = self.projection2.to(text_encoder.device, dtype=text_encoder.dtype)

        logger.info(f"Auto-created separate projections: {llm_hidden_size} -> {sdxl_encoder1_dim} + {sdxl_encoder2_dim}")

    def _combine_prompt_with_system(self, user_prompt: str, tokenizer) -> str:
        """Combine system prompt with user prompt based on tokenizer capabilities"""
        if not self.system_prompt:
            return user_prompt
            
        # Check if tokenizer has chat template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                full_prompt = tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=False,
                    tokenize=False
                )
                return full_prompt
            except Exception as e:
                print(f"Chat template application failed, falling back to text concatenation: {e}")
        
        # Fallback to text concatenation
        return f'{self.system_prompt} <Prompt Start> {user_prompt}'

    def _get_hidden_states_llm(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        text_encoder,
        attention_mask: torch.Tensor = None,
    ):
        # Handle both 2D (batch_size, seq_len) and 3D (batch_size, n, max_length) inputs
        b_size = input_ids.size()[0]
        if input_ids.dim() == 3:
            # 3D input: b,n,max_length -> b*n, max_length
            if input_ids.size()[1] == 1:
                max_token_length = None
            else:
                max_token_length = input_ids.size()[1] * input_ids.size()[2]
            input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*n, max_length
            if attention_mask is not None:
                attention_mask = attention_mask.reshape((-1, tokenizer.model_max_length))
        else:
            # 2D input: batch_size, seq_len
            max_token_length = None
            # Ensure input_ids is properly shaped for the model
            if input_ids.size(1) > tokenizer.model_max_length:
                input_ids = input_ids[:, :tokenizer.model_max_length]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :tokenizer.model_max_length]

        input_ids = input_ids.to(text_encoder.device)
        if attention_mask is None:
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
        attention_mask = attention_mask.to(text_encoder.device)

        # LLM encoding
        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-2]  # penultimate layer

            # Apply projection if available (outside no_grad for gradient flow)
            if self.projection1 is not None and self.projection2 is not None:
                # Use separate projections for each encoder
                encoder_hidden_states1 = self.projection1(hidden_states)
                encoder_hidden_states2 = self.projection2(hidden_states)
                # Concatenate for pooled output
                pooled_hidden = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=-1)
                pooled_output = (pooled_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
            elif self.projection is not None:
                # Legacy single projection mode
                hidden_states = self.projection(hidden_states)
                # Simple pooling: mean pooling
                pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
            else:
                # No projection - use raw LLM output (assume it's already 2048-dim)
                pooled_output = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

        # b*n, max_length, hidden_size -> b, n*max_length, hidden_size
        n_size = 1 if max_token_length is None else max_token_length // tokenizer.model_max_length
        hidden_states = hidden_states.reshape((b_size, -1, hidden_states.shape[-1]))

        if max_token_length is not None:
            # Handle token concatenation similar to SDXL
            states_list = [hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer.model_max_length):
                states_list.append(hidden_states[:, i : i + tokenizer.model_max_length - 2])  # content
            states_list.append(hidden_states[:, -1].unsqueeze(1))  # <EOS>
            hidden_states = torch.cat(states_list, dim=1)

            # pool uses first of n
            pooled_output = pooled_output[::n_size]

        return hidden_states, pooled_output

    def _get_hidden_states_llm_separate(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        text_encoder,
        attention_mask: torch.Tensor = None,
    ):
        """Get hidden states using separate projections for each SDXL encoder"""
        # Handle both 2D (batch_size, seq_len) and 3D (batch_size, n, max_length) inputs
        b_size = input_ids.size()[0]
        if input_ids.dim() == 3:
            # 3D input: b,n,max_length -> b*n, max_length
            if input_ids.size()[1] == 1:
                max_token_length = None
            else:
                max_token_length = input_ids.size()[1] * input_ids.size()[2]
            input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*n, max_length
            if attention_mask is not None:
                attention_mask = attention_mask.reshape((-1, tokenizer.model_max_length))
        else:
            # 2D input: batch_size, seq_len
            max_token_length = None
            # Ensure input_ids is properly shaped for the model
            if input_ids.size(1) > tokenizer.model_max_length:
                input_ids = input_ids[:, :tokenizer.model_max_length]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :tokenizer.model_max_length]

        input_ids = input_ids.to(text_encoder.device)
        if attention_mask is None:
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
        attention_mask = attention_mask.to(text_encoder.device)

        # LLM encoding
        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            llm_hidden_states = outputs.hidden_states[-2]  # penultimate layer

        # Apply separate projections for each encoder (outside no_grad for gradient flow)
        encoder_hidden_states1 = self.projection1(llm_hidden_states)  # -> 768 dim
        encoder_hidden_states2 = self.projection2(llm_hidden_states)  # -> 1280 dim

        # Create pooled output from encoder2 (following SDXL convention)
        pool2 = (encoder_hidden_states2 * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

        # b*n, max_length, hidden_size -> b, n*max_length, hidden_size
        n_size = 1 if max_token_length is None else max_token_length // tokenizer.model_max_length
        encoder_hidden_states1 = encoder_hidden_states1.reshape((b_size, -1, encoder_hidden_states1.shape[-1]))
        encoder_hidden_states2 = encoder_hidden_states2.reshape((b_size, -1, encoder_hidden_states2.shape[-1]))

        if max_token_length is not None:
            # Handle token concatenation similar to SDXL
            states_list1 = [encoder_hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
            states_list2 = [encoder_hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, max_token_length, tokenizer.model_max_length):
                states_list1.append(encoder_hidden_states1[:, i : i + tokenizer.model_max_length - 2])  # content
                states_list2.append(encoder_hidden_states2[:, i : i + tokenizer.model_max_length - 2])  # content
            states_list1.append(encoder_hidden_states1[:, -1].unsqueeze(1))  # <EOS>
            states_list2.append(encoder_hidden_states2[:, -1].unsqueeze(1))  # <EOS>
            encoder_hidden_states1 = torch.cat(states_list1, dim=1)
            encoder_hidden_states2 = torch.cat(states_list2, dim=1)

            # pool uses first of n
            pool2 = pool2[::n_size]

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def encode_tokens(
        self, tokenize_strategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Encode tokens for LLM
        Returns SDXL-compatible format: [encoder_hidden_states1, encoder_hidden_states2, pool2]
        """
        text_encoder, tokenizer = models[0], models[1]
        tokens_input = tokens[0]  # Single token input for LLM

        if self.projection1 is not None and self.projection2 is not None:
            # Use separate projections - get encoder outputs directly
            encoder_hidden_states1, encoder_hidden_states2, pool2 = self._get_hidden_states_llm_separate(
                tokens_input, tokenizer, text_encoder
            )
        else:
            # Legacy mode - single projection or no projection
            hidden_states, pooled_output = self._get_hidden_states_llm(
                tokens_input, tokenizer, text_encoder
            )

            # SDXL expects two encoder outputs: 768-dim and 1280-dim
            # Split the LLM output (2048-dim) into two parts for SDXL compatibility
            sdxl_encoder1_dim = 768   # CLIP ViT-L/14
            sdxl_encoder2_dim = 1280  # CLIP ViT-G/14

            encoder_hidden_states1 = hidden_states[:, :, :sdxl_encoder1_dim]  # First 768 dims
            encoder_hidden_states2 = hidden_states[:, :, sdxl_encoder1_dim:sdxl_encoder1_dim+sdxl_encoder2_dim]  # Next 1280 dims

            # For pooled output, use the second encoder's pooled representation (last 1280 dims)
            pool2 = pooled_output[:, sdxl_encoder1_dim:sdxl_encoder1_dim+sdxl_encoder2_dim]  # Last 1280 dims of pooled output

        return [encoder_hidden_states1, encoder_hidden_states2, pool2]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy,
        models: List[Any],
        tokens_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
        prompts: List[str] = None,
    ) -> List[torch.Tensor]:
        """
        Encode tokens with weights for LLM
        Returns SDXL-compatible format: [encoder_hidden_states1, encoder_hidden_states2, pool2]
        """
        text_encoder, tokenizer = models[0], models[1]
        tokens_input = tokens_list[0]  # Single token input for LLM
        weights_input = weights_list[0] if weights_list and len(weights_list) > 0 else None

        # Apply system prompt if provided
        if prompts and self.system_prompt:
            # Re-tokenize with combined prompts
            combined_prompts = [self._combine_prompt_with_system(prompt, tokenizer) for prompt in prompts]
            encodings = tokenizer(
                combined_prompts,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                pad_to_multiple_of=8,
            )
            tokens_input = encodings.input_ids.to(text_encoder.device)
            attention_mask = encodings.attention_mask.to(text_encoder.device)
        else:
            # Use provided tokens, assume attention_mask is all 1s
            attention_mask = torch.ones_like(tokens_input)

        # Get the hidden states
        hidden_states, pooled_output = self._get_hidden_states_llm(
            tokens_input, tokenizer, text_encoder, attention_mask
        )

        # Apply weights if provided and not all 1.0
        if weights_input is not None and not torch.allclose(weights_input, torch.ones_like(weights_input)):
            # Apply weights to hidden states (expand weights to match hidden dimension)
            if weights_input.dim() == 2 and weights_input.size(0) == hidden_states.size(0):
                # weights_input shape: [batch, seq_len], hidden_states shape: [batch, seq_len, hidden_dim]
                weights_expanded = weights_input.unsqueeze(-1).expand_as(hidden_states)
                hidden_states = hidden_states * weights_expanded

        # SDXL expects two encoder outputs: 768-dim and 1280-dim
        # Split the LLM output (2048-dim) into two parts for SDXL compatibility
        sdxl_encoder1_dim = 768   # CLIP ViT-L/14
        sdxl_encoder2_dim = 1280  # CLIP ViT-G/14

        encoder_hidden_states1 = hidden_states[:, :, :sdxl_encoder1_dim]  # First 768 dims
        encoder_hidden_states2 = hidden_states[:, :, sdxl_encoder1_dim:sdxl_encoder1_dim+sdxl_encoder2_dim]  # Next 1280 dims

        # For pooled output, use the second encoder's pooled representation (last 1280 dims)
        pool2 = pooled_output[:, sdxl_encoder1_dim:sdxl_encoder1_dim+sdxl_encoder2_dim]  # Last 1280 dims of pooled output

        return [encoder_hidden_states1, encoder_hidden_states2, pool2]
