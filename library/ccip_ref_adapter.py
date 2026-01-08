from __future__ import annotations

import os
import re
from typing import Optional

import torch
from torch import nn
from ccip_lib import ccip_batch_extract_features


_REF_OPEN_RE = re.compile(r"<img(\d+)>")
_REF_ANY_RE = re.compile(r"<img(\d+)>")


def strip_ref_tags_for_tokenization(text: str) -> str:
    """Preprocess ref tags in text before tokenization.
    
    Instead of removing them completely (which breaks sentence structure),
    we replace '<img1>' with 'img1' to keep the semantic placeholder in natural language form.
    """

    if not text:
        return text

    # Replace <imgK> with " imgK " to preserve position and ID distinction naturally
    # e.g. "photo of <img1>" -> "photo of img1"
    def _repl(m):
        return f" img{m.group(1)} "

    text = _REF_ANY_RE.sub(_repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_ref_order_from_caption(text: str, *, max_refs: int = 3) -> list[int]:
    """Parse <imgk> ordering from caption.

    Returns a list like [1,2,3]. If no tags found, returns [].
    """

    if not text:
        return []

    order: list[int] = []
    for m in _REF_OPEN_RE.finditer(text):
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        if idx < 1 or idx > max_refs:
            continue
        order.append(idx)
    return order


class CCIPToGemmaAdapter(nn.Module):
    """Map a CCIP feature (768) to a fixed number of Gemma2-compatible tokens.

    This follows a common multimodal pattern: produce K "image tokens" and append them to text tokens.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 2304, tokens_per_ref: int = 8):
        super().__init__()
        if tokens_per_ref <= 0:
            raise ValueError("tokens_per_ref must be >= 1")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tokens_per_ref = tokens_per_ref

        self.base_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.token_embed = nn.Parameter(torch.randn(tokens_per_ref, out_dim) * 0.02)
        
        # Improved Post-Processing: Standard FFN (Feed-Forward Network) structure
        # Linear -> GELU -> Linear is more expressive than the previous shallow mapping.
        self.post = nn.Sequential(
            nn.LayerNorm(out_dim),       # Pre-Norm for stability
            nn.Linear(out_dim, out_dim * 4), # Expansion (standard Transformer FFN ratio)
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim), # Projection back
        )

        # Zero-init the last projection to start with identity-like behavior for the main network
        nn.init.zeros_(self.post[-1].weight)
        nn.init.zeros_(self.post[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 768) -> (N, K, 2304)"""
        if x.ndim != 2:
            raise ValueError(f"expected (N,{self.in_dim}) but got {tuple(x.shape)}")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"expected feature dim {self.in_dim} but got {x.shape[1]}")
        base = self.base_proj(x)  # (N, D)
        tokens = base[:, None, :] + self.token_embed[None, :, :]
        return self.post(tokens)


def inject_ccip_refs_into_gemma_hidden_states(
    *,
    gemma_hidden_states: torch.Tensor,  # (B, S, 2304)
    attention_mask: torch.Tensor,  # (B, S)
    captions: list[str],
    ref_image_paths: list[list[Optional[str]]],  # (B, R)
    ccip_model_dir: str,
    ccip_image_size: int,
    adapter: CCIPToGemmaAdapter,
    dtype: torch.dtype,
    max_refs: int = 3,
    position: str = "begin",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ref images with CCIP and append adapter tokens after text tokens.

    Important: Lumina's NextDiT uses `cap_len = cap_mask.sum()` and then slices prefix `cap_feats[:, :cap_len]`.
    So we must keep the mask as a *prefix* of 1s (contiguous), and place injected tokens right after text.
    """

    if not ccip_model_dir:
        return gemma_hidden_states, attention_mask

    if gemma_hidden_states.ndim != 3:
        raise ValueError(f"gemma_hidden_states must be (B,S,D), got {gemma_hidden_states.shape}")
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be (B,S), got {attention_mask.shape}")

    bsz, seq_len, _ = gemma_hidden_states.shape
    k = int(adapter.tokens_per_ref)

    if position not in {"end", "begin"}:
        raise ValueError("position must be 'end' or 'begin'")

    out_hidden = gemma_hidden_states
    out_mask = attention_mask

    # Build flat batch for CCIP inference; store per-sample requested refs in order.
    flat_paths: list[str] = []
    flat_targets: list[tuple[int, int]] = []  # (sample_index, ref_order_index)
    
    # We don't need per_sample_ref_counts for logic, but might be useful for debug
    # per_sample_ref_counts: list[int] = [0 for _ in range(bsz)]

    for i in range(bsz):
        caption = captions[i] if i < len(captions) else ""
        order = parse_ref_order_from_caption(caption, max_refs=max_refs)
        if not order:
            # fallback: use sequential order based on available ref slots
            order = list(range(1, min(max_refs, len(ref_image_paths[i]) if i < len(ref_image_paths) else 0) + 1))

        # Current effective cap length (prefix ones)
        cap_len = int(out_mask[i].sum().item())
        if cap_len >= seq_len:
            continue

        # Collect valid refs in requested order
        refs = ref_image_paths[i] if i < len(ref_image_paths) else []
        ordered_paths: list[str] = []
        for ref_idx in order:
            if ref_idx - 1 < 0 or ref_idx - 1 >= len(refs):
                continue
            p = refs[ref_idx - 1]
            if p is None or not os.path.isfile(p):
                continue
            ordered_paths.append(p)

        if not ordered_paths:
            continue

        # Fit into remaining sequence length.
        max_fit = (seq_len - cap_len) // k
        if max_fit <= 0:
            continue
        if len(ordered_paths) > max_fit:
            ordered_paths = ordered_paths[:max_fit]

        # per_sample_ref_counts[i] = len(ordered_paths)
        for j, p in enumerate(ordered_paths):
            flat_paths.append(p)
            flat_targets.append((i, j))

    if not flat_paths:
        return out_hidden, out_mask

    # CCIP returns (N, 768) float32 numpy
    feats_np = ccip_batch_extract_features(flat_paths, size=ccip_image_size, model=ccip_model_dir)
    feats = torch.from_numpy(feats_np).to(device=out_hidden.device, dtype=torch.float32)

    # Adapter forward with grad enabled (we want to train it), but keep CCIP+adapter in full precision.
    # Under mixed precision (bf16/fp16), adapter weights can be cast by surrounding .to(dtype=...) calls.
    # That leads to matmul dtype mismatch (Float vs BFloat16). We explicitly keep adapter compute in fp32.
    try:
        p0 = next(adapter.parameters())
        if p0.dtype != torch.float32 and not getattr(adapter, "_ccip_forced_fp32", False):
            adapter.to(dtype=torch.float32)
            setattr(adapter, "_ccip_forced_fp32", True)
    except StopIteration:
        pass

    device_type = "cuda" if out_hidden.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        tokens_fp32 = adapter(feats)  # (N, K, D) float32

    tokens = tokens_fp32.to(dtype=dtype)  # (N, K, D)

    # Group tokens per sample
    per_sample_tokens: list[list[torch.Tensor]] = [[] for _ in range(bsz)]
    for n, (i, j) in enumerate(flat_targets):
        if 0 <= i < bsz:
            per_sample_tokens[i].append(tokens[n])  # (K, D)

    # Clone once before in-place writes to avoid mutating cached tensors
    out_hidden = out_hidden.clone()
    out_mask = out_mask.clone()

    for i in range(bsz):
        if not per_sample_tokens[i]:
            continue
        img_block = torch.cat(per_sample_tokens[i], dim=0)  # (R*K, D)
        total_img = img_block.shape[0]
        if total_img <= 0:
            continue

        cap_len = int(out_mask[i].sum().item())
        if cap_len > seq_len:
            cap_len = seq_len

        if position == "end":
            # Append after text (cap_len)
            if cap_len + total_img > seq_len:
                total_fit = seq_len - cap_len
                if total_fit <= 0:
                    continue
                img_block = img_block[:total_fit]
                total_img = img_block.shape[0]

            out_hidden[i, cap_len : cap_len + total_img, :] = img_block
            out_mask[i, cap_len : cap_len + total_img] = 1
            continue

        # position == "begin": prepend image tokens and shift text right.
        # Keep as many leading text tokens as can fit; drop tail tokens if needed.
        if total_img >= seq_len:
            # No room for text
            out_hidden[i, :, :] = img_block[:seq_len]
            out_mask[i, :] = 1
            continue

        keep_text = min(cap_len, seq_len - total_img)
        text_block = out_hidden[i, :keep_text, :].clone()

        out_hidden[i, :total_img, :] = img_block
        out_hidden[i, total_img : total_img + keep_text, :] = text_block
        # Zero out the remaining mask (padding)
        out_mask[i, : total_img + keep_text] = 1
        out_mask[i, total_img + keep_text :] = 0

    return out_hidden, out_mask
