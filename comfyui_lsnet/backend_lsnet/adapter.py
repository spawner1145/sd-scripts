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
            mode: "add" (text_pool + alpha * proj), "replace" (alpha * proj), "concat" (torch.cat)

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
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward_tokens(
        self,
        lsnet_emb: torch.Tensor,
        text_embeddings: torch.Tensor,
        insert_position: int = 0  # 0: prepend, -1: append
    ) -> torch.Tensor:
        """
        Generate extra tokens from LSNet embedding and concatenate to text_embeddings.

        Args:
            lsnet_emb: (batch, lsnet_feature_dim)
            text_embeddings: (batch, seq_len, clip_hidden_dim)
            insert_position: where to insert tokens (0 for prepend, -1 for append)

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

        # Insert tokens
        if insert_position == 0:
            return torch.cat([tokens, text_embeddings], dim=1)
        elif insert_position == -1:
            return torch.cat([text_embeddings, tokens], dim=1)
        else:
            # Insert at specific position
            before = text_embeddings[:, :insert_position]
            after = text_embeddings[:, insert_position:]
            return torch.cat([before, tokens, after], dim=1)

    def forward(
        self,
        lsnet_emb: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_pool: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        pooled_mode: str = "add",
        token_insert_position: int = 0
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full forward pass, optionally fusing both pooled and tokens.

        Returns:
            (new_text_embeddings, new_text_pool)
        """
        new_text_pool = None
        if text_pool is not None:
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


# Example usage
if __name__ == "__main__":
    # Example dimensions
    adapter = LSNetToClipAdapter(
        lsnet_feature_dim=384,
        clip_hidden_dim=2048,
        clip_pooled_dim=1280,
        num_extra_tokens=4,  # Generate 4 extra tokens
    )

    # Dummy inputs
    lsnet_emb = torch.randn(2, 384)  # batch=2
    text_embeddings = torch.randn(2, 77, 2048)
    text_pool = torch.randn(2, 1280)

    # Forward
    new_text_emb, new_text_pool = adapter(
        lsnet_emb, text_embeddings, text_pool,
        alpha=0.5, pooled_mode="add", token_insert_position=0
    )

    print(f"Original text_emb shape: {text_embeddings.shape}")
    print(f"New text_emb shape: {new_text_emb.shape}")
    print(f"Original text_pool shape: {text_pool.shape}")
    print(f"New text_pool shape: {new_text_pool.shape}")