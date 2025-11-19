import sys
import os
import torch
from unittest.mock import patch, MagicMock
import argparse

# Ensure dummy data exists
if not os.path.exists("dummy_data/img/10_dog/test.png"):
    import create_dummy_data

import susanoo_train_network
from library import susanoo_models, flux_models
from transformers import AutoConfig, AutoModel

def mock_load_text_encoder(path, dtype, device):
    print("Mocking Text Encoder loading...")
    # Create a minimal Qwen config manually to avoid internet access
    # Qwen2Config
    from transformers import Qwen2Config, Qwen2Model
    config = Qwen2Config(
        vocab_size=1000, # Small vocab
        hidden_size=64, # Small hidden size
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = Qwen2Model(config)
    model.to(dtype=dtype, device=device)
    return model

def mock_load_vae(path, dtype, device, disable_mmap=False):
    print("Mocking VAE loading...")
    # Use real VAE structure but random weights
    # Flux VAE params
    params = flux_models.configs["schnell"].ae_params
    # We can try to reduce channels to make it smaller?
    # params.z_channels is fixed by architecture probably.
    with torch.device("meta"):
         ae = flux_models.AutoEncoder(params).to(dtype)
    ae.to_empty(device=device)
    return ae

def mock_load_lsunet(path, dtype, device, disable_mmap=False):
    print("Mocking LSUNet loading...")
    with torch.device("meta"):
        unet = susanoo_models.LSUNet()
    unet.to_empty(device=device)
    return unet

def mock_create_lsunet(dtype, device):
    return mock_load_lsunet(None, dtype, device)

# Patching
with patch('library.susanoo_utils.load_text_encoder', side_effect=mock_load_text_encoder), \
     patch('library.susanoo_utils.load_vae', side_effect=mock_load_vae), \
     patch('library.susanoo_utils.load_lsunet', side_effect=mock_load_lsunet), \
     patch('library.susanoo_utils.create_lsunet', side_effect=mock_create_lsunet):

    print("Starting training test...")
    parser = susanoo_train_network.setup_parser()
    
    # Use a list of arguments
    args_list = [
        "--network_module", "networks.lora_susanoo",
        "--train_data_dir", "dummy_data/img",
        "--output_dir", "dummy_output",
        "--resolution", "512,512",
        "--train_batch_size", "1",
        "--max_train_steps", "1", # Just 1 step
        "--learning_rate", "1e-4",
        "--lsunet_path", "dummy_path",
        "--text_encoder_path", "dummy_path",
        "--vae", "dummy_path",
        "--mixed_precision", "no",
        "--save_precision", "float",
        "--dataset_repeats", "1",
        "--no_half_vae", # Force float32 VAE
        "--optimizer_type", "AdamW", # Use standard optimizer
    ]
    
    args = parser.parse_args(args_list)
    
    trainer = susanoo_train_network.SusanooNetworkTrainer()
    trainer.train(args)
