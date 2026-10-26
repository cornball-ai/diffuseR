# Generate FLUX.2 VAE encode parity fixtures for the R port.
#
# Loads the tiny random-init AutoencoderKLFlux2 already checked in by
# gen_fixtures_flux2_vae.py (same config) and encodes a random image the
# way Flux2KleinPipeline does for a reference image: posterior mean
# (sample_mode="argmax"), patchify, BatchNorm normalize, pack. Run via
# tools/gen_fixtures.sh.

import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers import AutoencoderKLFlux2  # noqa: E402
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")

vae = AutoencoderKLFlux2(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(8, 16, 32, 32),
    layers_per_block=2,
    latent_channels=32,
    norm_num_groups=8,
    use_quant_conv=True,
    use_post_quant_conv=True,
    sample_size=64,
)
vae.load_state_dict(load_file(os.path.join(OUT_DIR, "vae_flux2_tiny.safetensors")))
vae.eval()

torch.manual_seed(52)
image = torch.rand(1, 3, 32, 48) * 2 - 1
with torch.no_grad():
    mean = vae.encode(image).latent_dist.mode()
    lat = Flux2KleinPipeline._patchify_latents(mean)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    tokens = Flux2KleinPipeline._pack_latents((lat - bn_mean) / bn_std)
    ids = Flux2KleinPipeline._prepare_image_ids([lat])[0].float()

io = {"image": image, "mean": mean, "tokens": tokens, "image_ids": ids}
io = {k: v.contiguous() for k, v in io.items()}
save_file(io, os.path.join(OUT_DIR, "vae_flux2_enc_io.safetensors"),
          metadata={"purpose": "diffuseR FLUX.2 test fixture"})
print({k: tuple(v.shape) for k, v in io.items()})
