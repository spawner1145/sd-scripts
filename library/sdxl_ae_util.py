import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from library import flux_models, flux_utils, sdxl_original_unet

# 16-channel flux autoencoder settings
FLUX_VAE_LATENT_CHANNELS = flux_models.configs[flux_utils.MODEL_NAME_DEV].ae_params.z_channels
FLUX_VAE_SCALE_FACTOR = flux_models.configs[flux_utils.MODEL_NAME_DEV].ae_params.scale_factor
FLUX_VAE_SHIFT_FACTOR = flux_models.configs[flux_utils.MODEL_NAME_DEV].ae_params.shift_factor
# encode() of the flux AE already applies scale/shift, so no extra latent multiplier is needed in the training loop
FLUX_VAE_LATENT_MULT = 1.0


def enable_flux_vae_unet_channels(target_channels: Optional[int] = None) -> None:
    """Patch SDXL original UNet channel settings to match the flux VAE latent size."""
    channels = target_channels or FLUX_VAE_LATENT_CHANNELS
    if sdxl_original_unet.IN_CHANNELS != channels:
        sdxl_original_unet.IN_CHANNELS = channels
    if sdxl_original_unet.OUT_CHANNELS != channels:
        sdxl_original_unet.OUT_CHANNELS = channels


def _expand_conv_channels(weight: torch.Tensor, target_channels: int, dim: int) -> torch.Tensor:
    current = weight.shape[dim]
    if current == target_channels:
        return weight
    if target_channels % current != 0:
        raise ValueError(f"Cannot expand channels from {current} to {target_channels}")

    repeat = target_channels // current
    scale = repeat ** -0.5  # keep fan-in variance roughly stable

    out_shape = list(weight.shape)
    out_shape[dim] = target_channels
    expanded = weight.new_empty(out_shape)

    # copy existing channels and scale
    expanded.narrow(dim, 0, current).copy_(weight * scale)

    # initialize the extra channels with small Kaiming-normal noise to break symmetry
    if target_channels > current:
        tail = expanded.narrow(dim, current, target_channels - current)
        nn.init.kaiming_normal_(tail)
        tail.mul_(0.01 * scale)

    return expanded


def _expand_bias_channels(bias: torch.Tensor, target_channels: int) -> torch.Tensor:
    current = bias.shape[0]
    if current == target_channels:
        return bias
    if target_channels % current != 0:
        raise ValueError(f"Cannot expand bias from {current} to {target_channels}")

    out = bias.new_zeros(target_channels)
    out[:current] = bias
    return out


def upgrade_unet_state_dict_for_flux(
    state_dict: Dict[str, torch.Tensor], target_channels: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Expand the first/last conv layers of an SDXL UNet state dict to the flux VAE latent width."""
    if state_dict is None:
        return state_dict

    channels = target_channels or FLUX_VAE_LATENT_CHANNELS
    sd = copy.deepcopy(state_dict)

    # first conv: [model_channels, in_channels, 3, 3]
    key = "input_blocks.0.0.weight"
    if key in sd:
        sd[key] = _expand_conv_channels(sd[key], channels, dim=1)

    # last conv: [out_channels, model_channels, 3, 3] and bias
    key = "out.2.weight"
    if key in sd:
        sd[key] = _expand_conv_channels(sd[key], channels, dim=0)
    key = "out.2.bias"
    if key in sd:
        sd[key] = _expand_bias_channels(sd[key], channels)

    return sd
