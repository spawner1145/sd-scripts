import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union


class LSNetToClipAdapter(nn.Module):
    """
    Adapter to transform LSNet image embeddings into SDXL CLIP conditioning format.

    Supports two modes:
    1. Project to pooled embedding dimension and fuse with text_pool (recommended)
    2. Generate extra tokens and concatenate to text_embeddings
    """

    def __init__(
        self,
        lsnet_feature_dim: int = 384,
        clip_hidden_dim: int = 2048,  # CLIP text hidden: 768 + 1280 = 2048
        clip_pooled_dim: int = 1280,  # CLIP pooled dim
        num_extra_tokens: int = 0,  # If > 0, generate tokens for concatenation
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lsnet_feature_dim = lsnet_feature_dim
        self.clip_hidden_dim = clip_hidden_dim
        self.clip_pooled_dim = clip_pooled_dim
        self.num_extra_tokens = num_extra_tokens

        # Projection to pooled dimension (for fusion with text_pool)
        self.proj_to_pooled = nn.Linear(lsnet_feature_dim, clip_pooled_dim)
        if use_layer_norm:
            self.pooled_ln = nn.LayerNorm(clip_pooled_dim)
        else:
            self.pooled_ln = nn.Identity()
        self.pooled_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional: projection to generate extra tokens
        if num_extra_tokens > 0:
            self.proj_to_tokens = nn.Linear(lsnet_feature_dim, num_extra_tokens * clip_hidden_dim)
            if use_layer_norm:
                self.tokens_ln = nn.LayerNorm(clip_hidden_dim)
            else:
                self.tokens_ln = nn.Identity()
            self.tokens_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward_pooled(
        self,
        lsnet_emb: torch.Tensor,
        text_pool: torch.Tensor,
        alpha: float = 0.5,
        mode: str = "add"  # "add", "replace", "concat"
    ) -> torch.Tensor:
        """
        Fuse LSNet embedding with CLIP pooled embedding.

        Args:
            lsnet_emb: (batch, lsnet_feature_dim)
            text_pool: (batch, clip_pooled_dim)
            alpha: fusion weight
            mode: "add" (text_pool + alpha * proj), "replace" (alpha * proj), "concat" (torch.cat), "keep" (return text_pool unchanged)

        Returns:
            new_text_pool: (batch, clip_pooled_dim) or (batch, 2*clip_pooled_dim) if concat
        """
        # Project and normalize
        proj_emb = self.proj_to_pooled(lsnet_emb)
        proj_emb = self.pooled_ln(proj_emb)
        proj_emb = self.pooled_dropout(proj_emb)

        if mode == "add":
            return text_pool + alpha * proj_emb
        elif mode == "replace":
            return alpha * proj_emb
        elif mode == "concat":
            return torch.cat([text_pool, proj_emb], dim=-1)
        elif mode == "keep":
            return text_pool  # Keep original pool unchanged
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward_tokens(
        self,
        lsnet_emb: torch.Tensor,
        text_embeddings: torch.Tensor,
        insert_position: int = 1  # Changed default to 1 (after BOS)
    ) -> torch.Tensor:
        """
        Generate extra tokens from LSNet embedding and concatenate to text_embeddings.
        Follows SDXL's long token concatenation logic.

        Args:
            lsnet_emb: (batch, lsnet_feature_dim)
            text_embeddings: (batch, seq_len, clip_hidden_dim)
            insert_position: where to insert tokens (1 for after BOS, 0 for before BOS, -1 for append)

        Returns:
            new_text_embeddings: (batch, seq_len + num_extra_tokens, clip_hidden_dim)
        """
        if self.num_extra_tokens == 0:
            return text_embeddings

        # Generate tokens: (batch, num_extra_tokens * clip_hidden_dim) -> (batch, num_extra_tokens, clip_hidden_dim)
        tokens_flat = self.proj_to_tokens(lsnet_emb)
        tokens = tokens_flat.view(lsnet_emb.shape[0], self.num_extra_tokens, self.clip_hidden_dim)
        tokens = self.tokens_ln(tokens)
        tokens = self.tokens_dropout(tokens)

        # Use the new helper function for proper concatenation
        extended_embeddings, _ = extend_text_embeddings_sdxl_style(
            text_embeddings, tokens, torch.zeros(text_embeddings.shape[0], 1280), insert_position
        )
        return extended_embeddings

    def forward(
        self,
        lsnet_emb: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_pool: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        pooled_mode: str = "add",
        token_insert_position: int = 1  # Changed default to 1
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full forward pass, optionally fusing both pooled and tokens.
        Follows SDXL's long token concatenation logic for pool handling.

        Returns:
            (new_text_embeddings, new_text_pool)
        """
        new_text_pool = text_pool  # Keep original pool unchanged (SDXL style)
        # Optional: light fusion if requested
        if text_pool is not None and pooled_mode != "keep":
            new_text_pool = self.forward_pooled(lsnet_emb, text_pool, alpha, pooled_mode)

        new_text_embeddings = None
        if text_embeddings is not None and self.num_extra_tokens > 0:
            new_text_embeddings = self.forward_tokens(lsnet_emb, text_embeddings, token_insert_position)
        else:
            new_text_embeddings = text_embeddings

        return new_text_embeddings, new_text_pool


def load_lsnet_embedding_from_npz(npz_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Load LSNet embedding from npz file (for cached usage).

    Assumes npz contains 'lsnet_emb' key with shape (lsnet_feature_dim,)
    """
    data = np.load(npz_path)
    emb = torch.from_numpy(data["lsnet_emb"]).float()
    return emb.to(device)


def save_lsnet_embedding_to_npz(emb: torch.Tensor, npz_path: str):
    """
    Save LSNet embedding to npz file.

    Args:
        emb: (lsnet_feature_dim,) or (batch, lsnet_feature_dim) - will save first if batched
        npz_path: output path
    """
    if emb.dim() > 1:
        emb = emb[0]  # Take first in batch
    np.savez(npz_path, lsnet_emb=emb.cpu().numpy())


def extend_text_embeddings_sdxl_style(
    text_embeddings: torch.Tensor,  # (batch, seq_len, hidden_dim) - typically 77 tokens
    extra_tokens: torch.Tensor,     # (batch, num_extra, hidden_dim) - LSNet tokens
    original_pool: torch.Tensor,    # (batch, pool_dim) - original pooled embedding
    insert_position: int = 1        # 1 = after BOS (recommended)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extend text embeddings with extra tokens, directly following SDXL's get_hidden_states_sdxl logic.

    Based on: library/train_util.py get_hidden_states_sdxl function
    Reference code:
    - states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
    - for i in range(1, max_token_length, 77):
    -     states_list.append(hidden_states1[:, i : i + 77 - 2])  # <BOS>后到<EOS>前
    - states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
    - pool = pool[::n_size]  # 只用第一个块的pool
    """
    batch_size = text_embeddings.shape[0]

    if insert_position == 1:
        # 直接照抄SDXL逻辑：BOS + extra_tokens + 其余内容
        # 相当于在"BOS后到EOS前"的逻辑中插入extra_tokens
        bos_token = text_embeddings[:, 0:1]  # BOS
        rest_after_bos = text_embeddings[:, 1:]  # BOS后的所有内容（包括EOS等）
        extended = torch.cat([bos_token, extra_tokens, rest_after_bos], dim=1)

    elif insert_position == 0:
        # Insert before BOS - not following SDXL logic
        extended = torch.cat([extra_tokens, text_embeddings], dim=1)

    else:
        # Insert at specific position
        before = text_embeddings[:, :insert_position]
        after = text_embeddings[:, insert_position:]
        extended = torch.cat([before, extra_tokens, after], dim=1)

    # 直接照抄SDXL pool处理：pool = pool[::n_size] - 只用第一个块的pool
    # 在我们的情况下，original_pool就是第一个（也是唯一）块的pool，直接保持不变
    return extended, original_pool


import torch
from backend_lsnet.adapter import LSNetToClipAdapter, extend_text_embeddings_sdxl_style

# Test the new logic
adapter = LSNetToClipAdapter(num_extra_tokens=40)
lsnet_emb = torch.randn(1, 384)
text_emb = torch.randn(1, 77, 2048)
pool = torch.randn(1, 1280)

# Test extend function
lsnet_tokens = torch.randn(1, 40, 2048)
extended_emb, new_pool = extend_text_embeddings_sdxl_style(text_emb, lsnet_tokens, pool, insert_position=1)
print(f'Original shape: {text_emb.shape}')
print(f'Extended shape: {extended_emb.shape}')
print(f'Pool unchanged: {torch.allclose(pool, new_pool)}')

# Test adapter
new_emb, new_pool = adapter(lsnet_emb, text_emb, pool, pooled_mode='add', token_insert_position=1)
print(f'Adapter result shape: {new_emb.shape}')
print(f'Pool fused: {not torch.allclose(pool, new_pool)}')