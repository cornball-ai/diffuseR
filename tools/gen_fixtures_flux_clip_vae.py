# Generate CLIP (quick_gelu + pooled output) and 16-channel VAE decoder
# parity fixtures for the FLUX R port.
#
# Uses HF transformers CLIPTextModel and diffusers AutoencoderKL
# (Apache-2.0) with tiny random-init configs. Run via
# tools/gen_fixtures.sh; never executed at package test/run time.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from transformers import CLIPTextConfig, CLIPTextModel  # noqa: E402

from diffusers import AutoencoderKL  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(46)

# --- tiny CLIP text model (quick_gelu, legacy argmax pooling) ------------------
# eos_token_id=2 selects the legacy argmax(input_ids) pooling path, which
# is what FLUX's CLIP-L config uses.
clip_cfg = CLIPTextConfig(
    vocab_size=1000,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_position_embeddings=77,
    hidden_act="quick_gelu",
    eos_token_id=2,
)
clip = CLIPTextModel(clip_cfg)
clip.eval()

# EOS (as the max id, 999) at different positions; padding after
input_ids = torch.tensor([
    [49, 23, 61, 7, 999, 0, 0, 0],
    [33, 999, 5, 5, 5, 5, 5, 5],
])
with torch.no_grad():
    out = clip(input_ids, output_hidden_states=False)

save_file(
    {k: v.contiguous() for k, v in clip.state_dict().items()},
    os.path.join(OUT_DIR, "clip_tiny.safetensors"),
)

io = {
    "clip_input_ids": input_ids,
    "clip_last_hidden": out.last_hidden_state,
    "clip_pooled": out.pooler_output,
}

# --- tiny 16-channel VAE (FLUX/SD3 shape: no quant convs, mid attention) --------
vae = AutoencoderKL(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(8, 16, 32, 32),
    layers_per_block=2,
    latent_channels=16,
    norm_num_groups=8,
    use_quant_conv=False,
    use_post_quant_conv=False,
    mid_block_add_attention=True,
    sample_size=32,
)
vae.eval()

latent = torch.randn(1, 16, 4, 4)
with torch.no_grad():
    image = vae.decode(latent, return_dict=False)[0]

save_file(
    {k: v.contiguous() for k, v in vae.state_dict().items()},
    os.path.join(OUT_DIR, "vae16_tiny.safetensors"),
)

io["vae_latent"] = latent
io["vae_image"] = image
io = {k: v.contiguous() for k, v in io.items()}
save_file(io, os.path.join(OUT_DIR, "clip_vae_io.safetensors"))
print(f"wrote clip_tiny, vae16_tiny, and {len(io)} io tensors to {OUT_DIR}")
