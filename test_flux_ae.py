import torch
from library import flux_utils, sdxl_ae_util, sdxl_model_util, sdxl_original_unet

ae_path = "ae.safetensors"  # placed in repo root

def main():
    # 1) load flux AE
    ae = flux_utils.load_ae(ae_path, torch.float32, "cpu", disable_mmap=True)
    print("[AE] loaded dtype:", next(ae.parameters()).dtype)

    # 2) encode/decode sanity
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        z = ae.encode(x)
        y = ae.decode(z)
    print("[AE] z shape:", z.shape, z.dtype, "y shape:", y.shape, y.dtype)

    # 3) UNet 16ch adapt: expand mapping layers and load
    sdxl_ae_util.enable_flux_vae_unet_channels()
    with torch.device("meta"):
        unet = sdxl_original_unet.SdxlUNet2DConditionModel()
    unet = unet.to_empty(device="cpu")  # materialize params before loading

    sd = {
        "input_blocks.0.0.weight": torch.zeros(320, 4, 3, 3),
        "out.2.weight": torch.zeros(16, 320, 3, 3),
        "out.2.bias": torch.zeros(16),
    }
    sd = sdxl_ae_util.upgrade_unet_state_dict_for_flux(sd)
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print("[UNet] load missing:", missing, "unexpected:", unexpected)

    # 4) forward smoke test
    unet = unet.to(device="cpu", dtype=torch.float32)
    latents = torch.zeros(1, 16, 64, 64)
    timesteps = torch.zeros(1)
    text_emb = torch.zeros(1, 77, sdxl_original_unet.CONTEXT_DIM)
    vector_emb = torch.zeros(1, sdxl_original_unet.ADM_IN_CHANNELS)
    with torch.no_grad():
        out = unet(latents, timesteps, text_emb, vector_emb)
    print("[UNet] forward out shape:", out.shape)

if __name__ == "__main__":
    main()
