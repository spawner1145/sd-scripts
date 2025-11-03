# cache LSNet image embeddings to disk in advance for SDXL LoRA training
# python tools/cache_lsnet_outputs.py --sdxl --cache_lsnet_outputs_to_disk --train_data_dir "../test_cache_data" --output_dir "../test_output" --pretrained_model_name_or_path "../noobaiXLNAIXL_epsilonPred11Version.safetensors" --lsnet_checkpoint "../lsnet448/best_checkpoint.pth" --resolution 1024 --max_data_loader_n_workers 0 --num_lsnet_tokens 4

import argparse
import math
from multiprocessing import Value
import os
import sys

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'comfyui_lsnet'))

from accelerate.utils import set_seed
import torch
from tqdm import tqdm

from library import config_util
from library import train_util
from library import sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.utils import setup_logging, add_logging_arguments
setup_logging()
import logging
logger = logging.getLogger(__name__)

# Import LSNet components
from lsnet_model.lsnet_artist import lsnet_xl_artist_448
from inference_artist import (
    load_checkpoint_state,
    normalize_state_dict_keys,
    resolve_num_classes,
    resolve_feature_dim,
    load_class_mapping,
    get_args_parser,
    load_model,
    process_single_image
)
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
import numpy as np
from PIL import Image

# Import adapter
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'comfyui_lsnet', 'backend_lsnet'))
from adapter import LSNetToClipAdapter


def load_lsnet_model_from_folder(model_folder_path, device='cuda'):
    """
    Load LSNet model from a model folder (ComfyUI style).
    This avoids weight mismatch warnings by properly resolving model parameters.
    """
    import json
    
    # Paths
    checkpoint_path = os.path.join(model_folder_path, "best_checkpoint.pth")
    csv_path = os.path.join(model_folder_path, "class_mapping.csv")
    config_path = os.path.join(model_folder_path, "config.json")
    
    # Check files exist
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")
    
    # Load class mapping and resolve parameters
    class_mapping = load_class_mapping(csv_path)
    state_dict = load_checkpoint_state(checkpoint_path)
    state_dict = normalize_state_dict_keys(state_dict)
    num_classes = resolve_num_classes(None, class_mapping, state_dict)
    feature_dim = resolve_feature_dim(None, state_dict)
    
    # Load model type from config
    model_type = 'lsnet_xl_artist'  # default
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'model' in config and config['model'] in ['lsnet_t_artist', 'lsnet_s_artist', 'lsnet_b_artist', 'lsnet_l_artist', 'lsnet_xl_artist', 'lsnet_xl_artist_448']:
                    model_type = config['model']
                    logger.info(f"Model type loaded from config: {model_type}")
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    
    # Create and load model
    model = create_model(
        model_type,
        pretrained=False,
        num_classes=num_classes,
        feature_dim=feature_dim,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Get input size from model config - import here to avoid circular imports
    try:
        from lsnet_model.lsnet_artist import default_cfgs_artist
        input_size = 224  # default
        if model_type in default_cfgs_artist:
            model_cfg = default_cfgs_artist[model_type]
            configured_input_size = model_cfg.get('input_size', (3, 224, 224))[1]
            input_size = configured_input_size
            logger.info(f"Auto-setting input_size to {input_size} for model {model_type}")
    except ImportError:
        logger.warning("Could not import default_cfgs_artist, using default input_size=224")
        input_size = 224
    
    # Create transform
    config = resolve_data_config({'input_size': (3, input_size, input_size)}, model=model)
    transform = create_transform(**config)
    
    return model, transform, input_size


def cache_to_disk(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    train_util.prepare_dataset_args(args, True)

    # check cache arg
    assert (
        args.cache_lsnet_outputs_to_disk
    ), "cache_lsnet_outputs_to_disk must be True / cache_lsnet_outputs_to_diskはTrueである必要があります"

    # Only SDXL supported for now
    assert (
        args.sdxl
    ), "cache_lsnet_outputs_to_disk is only available for SDXL / cache_lsnet_outputs_to_diskはSDXLのみ利用可能です"

    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    # tokenizerを準備する：datasetを動かすために必要
    if args.sdxl:
        tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)
        tokenizers = [tokenizer1, tokenizer2]
    else:
        tokenizer = train_util.load_tokenizer(args)
        tokenizers = [tokenizer]

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizers)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizers)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    # acceleratorを準備する
    logger.info("prepare accelerator")
    args.deepspeed = False
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, _ = train_util.prepare_dtype(args)

    # モデルを読み込む
    logger.info("load model")
    if args.sdxl:
        (_, text_encoder1, text_encoder2, _, _, _, _) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
        text_encoders = [text_encoder1, text_encoder2]
    else:
        text_encoder1, _, _, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        text_encoders = [text_encoder1]

    for text_encoder in text_encoders:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    # Load LSNet model (if not skipping)
    if not args.skip_lsnet:
        if args.lsnet_checkpoint is None and args.lsnet_model_folder is None:
            raise ValueError("--lsnet_checkpoint or --lsnet_model_folder is required when not using --skip_lsnet")
        if args.lsnet_checkpoint is not None and args.lsnet_model_folder is not None:
            raise ValueError("Cannot specify both --lsnet_checkpoint and --lsnet_model_folder")
        
        logger.info("load LSNet model")
        
        if args.lsnet_model_folder is not None:
            # Use ComfyUI-style folder loading (no warnings)
            lsnet_model, lsnet_transform, input_size = load_lsnet_model_from_folder(
                args.lsnet_model_folder, 
                device=str(accelerator.device)
            )
            lsnet_model.to(accelerator.device, dtype=weight_dtype)
        else:
            # Use legacy single checkpoint loading (may have warnings)
            logger.warning("Using single checkpoint loading - weight mismatch warnings may appear")
            
            # Create args object for LSNet
            class LSNetArgs:
                def __init__(self):
                    self.model = 'lsnet_xl_artist_448'
                    self.checkpoint = args.lsnet_checkpoint
                    self.num_classes = None
                    self.feature_dim = None
                    self.input_size = 448
                    self.device = str(accelerator.device)
                    self.mode = 'cluster'  # We only need features, not classification
                    self.allow_head_reinit = True
            
            lsnet_args = LSNetArgs()
            
            lsnet_state_dict = load_checkpoint_state(args.lsnet_checkpoint)
            lsnet_state_dict = normalize_state_dict_keys(lsnet_state_dict)
            lsnet_model = load_model(lsnet_args, lsnet_state_dict)
            lsnet_model.to(accelerator.device, dtype=weight_dtype)
            
            # Prepare LSNet transform
            config = resolve_data_config({'input_size': (3, lsnet_args.input_size, lsnet_args.input_size)}, model=lsnet_model)
            lsnet_transform = create_transform(**config)
            input_size = lsnet_args.input_size
        
        lsnet_model.eval()

        # Initialize adapter for merging LSNet emb with text embeddings
        adapter = LSNetToClipAdapter(
            lsnet_feature_dim=lsnet_model.feature_dim,
            clip_hidden_dim=2048,  # SDXL text hidden: 768 + 1280
            clip_pooled_dim=1280,  # SDXL pooled dim
            num_extra_tokens=args.num_lsnet_tokens,
        ).to(accelerator.device, dtype=weight_dtype)
    else:
        logger.info("Skipping LSNet model loading (--skip_lsnet)")
        lsnet_model = None
        lsnet_transform = None
        adapter = None

    # dataloaderを準備する
    train_dataset_group.set_caching_mode("text")

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # acceleratorを使ってモデルを準備する：マルチGPUで使えるようになるはず
    train_dataloader = accelerator.prepare(train_dataloader)

    # データ取得のためのループ
    for batch in tqdm(train_dataloader):
        absolute_paths = batch["absolute_paths"]
        input_ids1_list = batch["input_ids1_list"]
        input_ids2_list = batch["input_ids2_list"]

        image_infos = []
        for absolute_path, input_ids1, input_ids2 in zip(absolute_paths, input_ids1_list, input_ids2_list):
            image_info = train_util.ImageInfo(absolute_path, 1, "dummy", False, absolute_path)
            image_info.text_encoder_outputs_npz = os.path.splitext(absolute_path)[0] + train_util.TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX
            image_info

            if args.skip_existing:
                if os.path.exists(image_info.text_encoder_outputs_npz):
                    logger.warning(f"Skipping {image_info.text_encoder_outputs_npz} because it already exists.")
                    continue
                
            image_info.input_ids1 = input_ids1
            image_info.input_ids2 = input_ids2
            image_infos.append(image_info)

        if len(image_infos) > 0:
            b_input_ids1 = torch.stack([image_info.input_ids1 for image_info in image_infos])
            b_input_ids2 = torch.stack([image_info.input_ids2 for image_info in image_infos])
            
            # First cache text encoder outputs
            train_util.cache_batch_text_encoder_outputs(
                image_infos, tokenizers, text_encoders, args.max_token_length, True, b_input_ids1, b_input_ids2, weight_dtype
            )

            # Then merge LSNet embeddings with text embeddings (if not skipping)
            if not args.skip_lsnet:
                for image_info in image_infos:
                    # Load and preprocess image
                    image = Image.open(image_info.absolute_path).convert('RGB')
                    input_tensor = lsnet_transform(image).unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

                    with torch.no_grad():
                        lsnet_emb = lsnet_model.forward_features(input_tensor)  # (1, feature_dim)

                    # Load existing npz
                    existing_npz = np.load(image_info.text_encoder_outputs_npz)
                    text_encoder_outputs1 = torch.from_numpy(existing_npz['hidden_state1']).unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
                    text_encoder_outputs2 = torch.from_numpy(existing_npz['hidden_state2']).unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
                    text_encoder_pool2 = torch.from_numpy(existing_npz['pool2']).to(accelerator.device, dtype=weight_dtype)

                    # Concatenate text embeddings
                    text_embeddings = torch.cat([text_encoder_outputs1, text_encoder_outputs2], dim=2)  # (1, 77, 2048)

                    # Use adapter to prepend LSNet tokens AND update pool
                    new_text_embeddings, new_text_pool2 = adapter(
                        lsnet_emb, text_embeddings, text_encoder_pool2.unsqueeze(0),  # Include pooled fusion
                        alpha=1.0, pooled_mode="add", token_insert_position=0
                    )

                    # Split back to encoder1 and encoder2 outputs
                    new_text_encoder_outputs1 = new_text_embeddings[:, :, :768]  # First 768 dims
                    new_text_encoder_outputs2 = new_text_embeddings[:, :, 768:2048]  # Next 1280 dims

                    # Save merged npz
                    np.savez(
                        image_info.text_encoder_outputs_npz,
                        hidden_state1=new_text_encoder_outputs1.detach().cpu().numpy().squeeze(0),
                        hidden_state2=new_text_encoder_outputs2.detach().cpu().numpy().squeeze(0),
                        pool2=new_text_pool2.detach().cpu().numpy().squeeze(0),  # Use updated pool
                        lsnet_emb=lsnet_emb.cpu().numpy().squeeze(0)  # Also save raw lsnet_emb for reference
                    )
            else:
                logger.info("Skipped LSNet processing, only text encoder outputs cached")

    accelerator.wait_for_everyone()
    accelerator.print(f"Finished caching LSNet embeddings for {len(train_dataset_group)} batches.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, True)
    train_util.add_dataset_arguments(parser, True, True, True)
    config_util.add_config_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model / SDXLモデルを使用する")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip images if npz already exists (both normal and flipped exists if flip_aug is enabled) / npzが既に存在する画像をスキップする（flip_aug有効時は通常、反転の両方が存在する画像をスキップ）",
    )
    parser.add_argument(
        "--cache_lsnet_outputs_to_disk",
        action="store_true",
        help="Cache LSNet outputs to disk / LSNet出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--lsnet_checkpoint",
        type=str,
        default=None,
        help="Path to LSNet checkpoint (single file) / LSNetチェックポイントのパス（単一ファイル）",
    )
    parser.add_argument(
        "--lsnet_model_folder",
        type=str,
        default=None,
        help="Path to LSNet model folder (ComfyUI style with config.json, class_mapping.csv) / LSNetモデルフォルダのパス（ComfyUI形式）",
    )
    parser.add_argument(
        "--num_lsnet_tokens",
        type=int,
        default=4,
        help="Number of LSNet tokens to prepend to text embeddings / テキスト埋め込みの前に追加するLSNetトークンの数",
    )
    parser.add_argument(
        "--skip_lsnet",
        action="store_true",
        help="Skip LSNet processing, only cache text encoder outputs / LSNet処理をスキップし、テキストエンコーダーの出力のみをキャッシュ",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    cache_to_disk(args)