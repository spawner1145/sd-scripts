# Lumina-NextDiT ID Adapter Training

本文档介绍了如何在 `sd-scripts` 框架下训练 Lumina-NextDiT 的 ID Adapter。  
该功能允许通过参考图（Reference Image）提取人物特征，并将其转化为 Image Tokens 注入到 Gemma2 文本编码器中，实现高保真的人物一致性生成。

## 1. 核心特性

- **多模态注入**：将单张参考图的特征映射为 K 个（默认 32）Image Tokens，扩展信息带宽。
- **精确控制**：通过 Caption 中的 `<img1>`, `<img2>` 等标签，精确控制每张参考图对应的文本上下文位置。
- **无缝集成**：Adapter 可与 LoRA / LyCORIS 一起训练，也可以单独微调 Adapter 而冻结 DiT（通过 `--train_dit=False`）。
- **自然语言融合**：标签 `<imgk>` 在送入 Text Encoder 前会被转换为普通文本 `imgk`（如 `<img1>` -> `img1`），保持句子结构完整，便于模型理解指代关系。

## 2. 数据准备

### 2.1 文件结构

数据集格式与常规 `sd-scripts` 训练一致，但在图片同级目录下需要放置参考图。

```text
/dataset/
  ├── char_001.jpg            # 训练主图
  ├── char_001.txt            # Caption 文件
  ├── char_001_ref1.jpg       # 参考图 1 (对应 <img1>)
  ├── char_001_ref2.jpg       # 参考图 2 (对应 <img2>, 可选)
  └── ...
```

**命名规则**：
- 参考图文件名必须以 **主图文件名** 开头。
- 后缀必须是 `_ref1`, `_ref2`, `_ref3` ... (支持常见图片格式)。

### 2.2 Caption 写法

在 Caption 文本中，使用 `<imgk>` 标签来指代对应的参考图。

**示例**：
```text
A photo of <img1>, smiling, wearing a hat. <img2> is standing in the background.
```
- `<img1>` 将会读取 `char_001_ref1.jpg` 的特征并注入。
- `<img2>` 将会读取 `char_001_ref2.jpg` 的特征并注入。
- 最终送入 Text Encoder 的文本实际上是：`"A photo of img1, smiling, wearing a hat. img2 is standing in the background."`（标签转换），Adapter 生成的 Image Tokens 会被追加到序列末尾，模型会自动学习 `img1` 文本与对应视觉 Tokens 的关联。

## 3. 准备特征提取模型 (CCIP)

你需要准备 CCIP (或兼容的 InsightFace) 模型目录，该目录应包含特征提取所需的 ONNX 模型（如 `model_feat.onnx`）。

假设存放路径为：`D:/models/ccip_model/`

## 4. 训练命令

使用 `lumina_train_network.py` 启动训练。

### 关键参数说明

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--ccip_model_dir` | CCIP 模型文件夹路径 | 可选 (启用 Adapter 时必须) |
| `--ccip_image_size` | 提取特征时的图片缩放尺寸 | 384 |
| `--ccip_feat_dim` | 输入特征向量的维度 | 768 |
| `--adapter_tokens_per_ref` | **每张参考图**生成的 Token 数量 | 32 |
| `--adapter_inject_position` | Image Tokens 注入位置 (`begin` 或 `end`) | `begin` |
| `--adapter_output_path` | Adapter 权重单独保存**目录** (可选，每个 checkpoint 生成一个文件) | 强烈建议设置，以便分离保存 |
| `--train_dit` | 是否训练主网络 (True/False)。设为 False 可只训 Adapter | True |
| `--train_adapter` | 是否训练 Adapter (True/False)。设为 False 可冻结 Adapter | True |
| `--adapter_lr` | Adapter 专用学习率。不填则默认跟随全局 LR | None |

### 示例脚本

```bash
accelerate launch --num_cpu_threads_per_process 2 lumina_train_network.py ^
    --pretrained_model_name_or_path "D:/models/Lumina-Next-SFT-Diffusers" ^
    --dataset_config "D:/train/dataset_config.toml" ^
    --output_dir "D:/train/output" ^
    --output_name "my_lumina_adapter" ^
    --learning_rate 1e-4 ^
    --network_module "networks.lora_lumina" ^
    --network_dim 32 --network_alpha 16 ^
    --ccip_model_dir "D:/models/ccip_model" ^
    --ccip_image_size 384 ^
    --ccip_feat_dim 768 ^
    --adapter_tokens_per_ref 32 ^
    --adapter_inject_position "begin" ^
    --adapter_output_path "D:/train/output/adapter_weights" ^
    --train_dit "True" ^
    --mixed_precision "bf16" ^
    --save_precision "bf16" ^
    --gradient_checkpointing ^
    --max_train_epochs 10 ^
```

## 5. 进阶用法

### 5.1 训练模式组合

我们支持三种灵活的训练模式，适应不同阶段的需求：

#### A. 联合训练 (Joint Training) - 默认推荐
同时微调 DiT (LoRA) 和 Adapter。这能让模型学会如何最好地利用 Adapter 提供的 Token 信息。
*   参数：`--train_dit True --train_adapter True` (默认)
*   适用：最初阶段的训练。

#### B. 仅训练 Adapter (Adapter Only Stage)
冻结 DiT，只训练 Adapter 将图片特征对齐到文本空间。
*   参数：`--train_dit False --train_adapter True`
*   适用：当你有一个强大的 Base 模型，只想为其添加 ID 能力而不破坏原有画风时。

#### C. 仅训练 DiT (DiT Only Stage)
使用预训练好的 Adapter（作为类似 IP-Adapter 的条件输入），冻结 Adapter，只训练 DiT (LoRA) 来适应特定风格。
*   参数：`--train_dit True --train_adapter False --adapter_model_path "你的预训练adapter.safetensors"`
*   适用：Adapter 已经训练得很好了，现在想把它应用到某个特定画风的 LoRA 训练中。

### 5.2 关于 Token 长度的警告 (Context Window)

Lumina 使用 Gemma-2B 作为文本编码器，其输入长度是有限的（默认为 256）。
Adapter 生成的 Image Tokens 会占用这个宝贵的长度空间。

**估算公式**：
`总消耗 ≈ System Prompt (约20) +User Prompt (Caption) + 参考图数量 × 每图 Token 数`

如果你的配置是：
- Caption 长度：70 tokens
- 参考图：3 张
- 每图 Token 数：32 (`--adapter_tokens_per_ref 32`)
- **总计**：20 + 70 + (3 * 32) = 186 < 256 (安全)

**风险**：
如果你设 `tokens_per_ref=64`，总计将达到 282 > 256。
*   如果是 `position="end"`：会**丢弃注入的图片特征**，导致训练无效。
*   如果是 `position="begin"` (默认)：会丢弃 Caption 末尾的文本 Token，保留图片特征。虽然比 end 安全，但仍建议提升 Context 长度以保留完整文本语义。

**建议**：
如果使用 Adapter，**强烈建议**在训练命令中显式增加 `--gemma2_max_token_length`：
```bash
--gemma2_max_token_length 512
```
(注意：增加 Token 长度会线性增加文本编码器的显存占用，请根据显存酌情调整)

### 5.3 调整 Token 数量
增加 `--adapter_tokens_per_ref` (如 64) 可以增加 ID 的细节还原能力，但可能增加训练难度；减少 (如 8) 则更轻量，更不易崩坏。目前默认为 32。

### 5.4 自定义特征来源
如果你使用的不是标准 CCIP 模型，而是自己提取的特征：
1. 请确保特征维度与 `--ccip_feat_dim` 一致。
2. 你可能需要修改 `library/ccip_ref_adapter.py` 或相关逻辑来对接你的特征后端。

## 6. 保存与加载

### 6.1 保存

- **LoRA / network 权重**：仍按原逻辑保存到 `--output_dir` 下的 `.safetensors`。
- **Adapter 权重**：如果设置了 `--adapter_output_path`，会在该**目录**下按 checkpoint 名称保存为独立文件（因此通常会有多个文件）。
- **严格分离**：Adapter 权重不会写入 LoRA / network 的 `.safetensors` 文件中。

当你处于“仅训练 Adapter”的模式（例如冻结 DiT/UNet 和文本编码器的 LoRA，只训练 Adapter）时，脚本会只保存 Adapter 文件，不会额外保存 LoRA / network checkpoint。

### 6.2 加载

- 使用 `--adapter_model_path` 加载之前训练好的 Adapter 权重继续微调或进行推理。
