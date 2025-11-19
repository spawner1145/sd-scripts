import torch
import os
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
from library import susanoo_models, lumina_util
from library.susanoo_models import LSUNet, TextProjection
from library.safetensors_utils import load_safetensors
import logging

logger = logging.getLogger(__name__)

def load_checkpoint(path, device, dtype=None):
    if path.endswith(".safetensors"):
        return load_safetensors(path, device=device, dtype=dtype)
    else:
        return torch.load(path, map_location="cpu")

def load_tokenizer(path):
    if path:
        logger.info(f"Loading Tokenizer from {path}...")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    else:
        logger.warning("No text_encoder_path provided, using default Qwen/Qwen1.5-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    return tokenizer

def load_text_encoder(path, dtype, device):
    logger.info(f"Loading Qwen Text Encoder from {path}...")
    if path:
        text_encoder = AutoModel.from_pretrained(path, trust_remote_code=True)
    else:
        text_encoder = AutoModel.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    
    text_encoder.to(device, dtype=dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    return text_encoder

def load_vae(ckpt_path, dtype, device, disable_mmap=False):
    return lumina_util.load_ae(ckpt_path, dtype, device, disable_mmap)

def load_lsunet(path, dtype, device, disable_mmap=False):
    logger.info("Loading LSUNet...")
    with torch.device("meta"):
        unet = LSUNet()
        
    if path:
        logger.info(f"Loading LSUNet weights from {path}")
        if path.endswith(".safetensors"):
            sd = load_safetensors(path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
        else:
            sd = torch.load(path, map_location="cpu")
            
        unet.load_state_dict(sd, strict=False, assign=True)
    else:
        # Should use create_lsunet for scratch
        unet = unet.to_empty(device=device).to(dtype)
    
    unet.to(device, dtype=dtype)
    unet.enable_gradient_checkpointing()
    return unet

def create_lsunet(dtype, device):
    logger.info("Creating LSUNet from scratch...")
    unet = LSUNet()
    unet.to(device, dtype=dtype)
    unet.enable_gradient_checkpointing()
    return unet

def create_text_projection(in_dim, out_dim, dtype, device):
    logger.info(f"Creating Text Projection Layer: {in_dim} -> {out_dim}")
    text_projection = TextProjection(in_dim, out_dim)
    text_projection.to(device, dtype=dtype)
    text_projection.train()
    return text_projection

def load_text_projection(path, dtype, device):
    logger.info(f"Loading Text Projection from {path}...")
    sd = load_checkpoint(path, device, dtype)
    
    # Infer dims from state dict
    if "linear.weight" in sd:
        weight = sd["linear.weight"]
        out_dim, in_dim = weight.shape
        text_projection = TextProjection(in_dim, out_dim)
        text_projection.load_state_dict(sd)
    else:
        # Fallback or error
        logger.error("Could not find linear.weight in text projection checkpoint")
        raise ValueError("Invalid text projection checkpoint")
        
    text_projection.to(device, dtype=dtype)
    return text_projection
