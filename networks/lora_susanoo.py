import torch
from networks import lora
from typing import List, Optional, Union, Dict, Type
from transformers import CLIPTextModel
from diffusers import AutoencoderKL
import logging

logger = logging.getLogger(__name__)

import torch
from networks import lora
from networks.lora import LoRAModule
from typing import List, Optional, Union, Dict, Type
from transformers import CLIPTextModel
from diffusers import AutoencoderKL
import logging
import os

logger = logging.getLogger(__name__)

class SusanooLoRANetwork(lora.LoRANetwork):
    # Define target modules for Susanoo
    UNET_TARGET_REPLACE_MODULE = [
        "LSTransformer2DModel", 
        "ResnetBlock2D", 
        "Downsample2D", 
        "Upsample2D",
        # "LSUNetBlock", # Covered by LSTransformer2DModel
        # "LSConv", 
        # "CrossAttention", 
        # "FeedForward"
    ]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Qwen3Attention", "Qwen3MLP"]
    
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        block_dims: Optional[List[int]] = None,
        block_alphas: Optional[List[float]] = None,
        conv_block_dims: Optional[List[int]] = None,
        conv_block_alphas: Optional[List[float]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        module_class: Type[object] = LoRAModule,
        varbose: Optional[bool] = False,
        is_sdxl: Optional[bool] = False,
    ) -> None:
        # We cannot call super().__init__ because it uses LoRANetwork.UNET_TARGET_REPLACE_MODULE
        # So we copy the init logic here.
        
        torch.nn.Module.__init__(self)
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info(f"create LoRA network from weights")
        elif block_dims is not None:
            logger.info(f"create LoRA network from block_dims")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            logger.info(f"block_dims: {block_dims}")
            logger.info(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                logger.info(f"conv_block_dims: {conv_block_dims}")
                logger.info(f"conv_block_alphas: {conv_block_alphas}")
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            if self.conv_lora_dim is not None:
                logger.info(
                    f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}"
                )

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],  # None, 1, 2
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
        ):
            prefix = (
                self.LORA_PREFIX_UNET
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                )
            )
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                # U-Netでblock_dims指定あり
                                # block_idx = get_block_index(lora_name, is_sdxl) # TODO: Implement block index for Susanoo
                                # For now, just use default
                                dim = self.lora_dim
                                alpha = self.alpha
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(lora_name)
                                continue

                            lora_mod = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                            )
                            loras.append(lora_mod)
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        # create LoRA for text encoder
        self.text_encoder_loras = []
        skipped_te = []
        for i, text_encoder in enumerate(text_encoders):
            if len(text_encoders) > 1:
                index = i + 1
                logger.info(f"create LoRA for Text Encoder {index}:")
            else:
                index = None
                logger.info(f"create LoRA for Text Encoder:")

            text_encoder_loras, skipped = create_modules(False, index, text_encoder, self.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped
        logger.info(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend LSU-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = self.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += self.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            # Remove duplicates
            target_modules = list(set(target_modules))

        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
        logger.info(f"create LoRA for LSU-Net: {len(self.unet_loras)} modules.")

        skipped = skipped_te + skipped_un
        if varbose and len(skipped) > 0:
            logger.warning(
                f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                logger.info(f"  {name}")

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: List[float] = None
        self.block_lr = False

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    # We can inherit other methods from LoRANetwork

def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: AutoencoderKL,
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    network = SusanooLoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=kwargs.get("rank_dropout", None),
        module_dropout=kwargs.get("module_dropout", None),
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        modules_dim=kwargs.get("modules_dim", None),
        modules_alpha=kwargs.get("modules_alpha", None),
        varbose=kwargs.get("varbose", False),
    )

    return network

def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if file is None:
            raise ValueError("file or weights_sd must be specified")
        weights_sd, metadata = lora.load_state_dict_with_metadata(file)
    else:
        metadata = {}
    
    # Get dim/alpha from weights if possible
    # This is a simplified version, assuming standard LoRA
    # For full support we should parse metadata like lora.py does
    
    # For now, let's just create a network with default dims and load weights
    # The LoRANetwork class handles loading weights into modules with different dims if needed?
    # No, we need to know the dim to create the modules.
    
    # We can use lora.create_network_from_weights logic but substitute the class
    # But lora.create_network_from_weights is complex.
    
    # Let's try to use lora.create_network_from_weights but since we can't inject the class easily...
    # Actually, lora.create_network_from_weights calls create_network.
    # If I copy create_network_from_weights and call my create_network, it should work.
    
    # Simplified implementation for now:
    # 1. Parse metadata to get dim/alpha
    # 2. Call create_network
    # 3. Load weights
    
    # But wait, lora.py has a lot of logic for parsing metadata.
    # I'll just use lora.create_network_from_weights but I need to make sure it uses SusanooLoRANetwork.
    # It doesn't. It uses LoRANetwork.
    
    # So I have to reimplement it or copy it.
    # I'll copy the essential parts.
    
    if metadata is None:
        metadata = {}
        
    # Parse dim/alpha from metadata if available
    network_dim = None
    network_alpha = None
    
    if "network_dim" in metadata:
        network_dim = int(metadata["network_dim"])
    if "network_alpha" in metadata:
        network_alpha = float(metadata["network_alpha"])
        
    if network_dim is None:
        # Try to guess from weights (first weight)
        for key, value in weights_sd.items():
            if "lora_down.weight" in key:
                network_dim = value.shape[0]
                break
                
    if network_dim is None:
        network_dim = 4
        
    if network_alpha is None:
        network_alpha = 1.0 # Default

    # Parse modules_dim and modules_alpha from weights_sd
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            
    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]
        
    network = create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoder,
        unet,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        **kwargs
    )
    
    # info = network.load_state_dict(weights_sd, strict=False)
    # logger.info(f"Loaded Susanoo LoRA weights: {info}")
    
    return network, weights_sd
