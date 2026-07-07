# Generate video VAE parity fixtures for the LTX-2.3 R port.
# Tiny encoder/decoder with the real 2.3 block structure (mixed
# down/upsample types, factors (2,2,1,2), no residual, zeros padding).

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.autoencoders.autoencoder_kl_ltx2 import (  # noqa: E402
    LTX2VideoCausalConv3d,
    LTX2VideoDecoder3d,
    LTX2VideoDownsampler3d,
    LTX2VideoEncoder3d,
    LTX2VideoResnetBlock3d,
    LTX2VideoUpsampler3d,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(23)
fx = {}


def add_module(prefix, module):
    for name, p in module.state_dict().items():
        fx[f"{prefix}.{name}"] = p


# --- Causal conv: causal and non-causal paths ----------------------------------
cc = LTX2VideoCausalConv3d(4, 6, kernel_size=3).eval()
add_module("cc", cc)
x = torch.randn(2, 4, 5, 6, 6)
fx["cc_x"] = x
with torch.no_grad():
    fx["cc_out_causal"] = cc(x, causal=True)
    fx["cc_out_noncausal"] = cc(x, causal=False)

# --- Resnet block with channel change (LayerNorm + conv shortcut) --------------
rb = LTX2VideoResnetBlock3d(in_channels=4, out_channels=8).eval()
add_module("rb", rb)
with torch.no_grad():
    fx["rb_out"] = rb(x, causal=True)

# --- Downsampler (temporal) and upsampler (spatiotemporal, residual) ------------
ds = LTX2VideoDownsampler3d(in_channels=8, out_channels=16, stride=(2, 1, 1)).eval()
add_module("ds", ds)
xd = torch.randn(1, 8, 5, 4, 4)
fx["ds_x"] = xd
with torch.no_grad():
    fx["ds_out"] = ds(xd, causal=True)

us = LTX2VideoUpsampler3d(in_channels=16, stride=(2, 2, 2), residual=True, upscale_factor=2).eval()
add_module("us", us)
xu = torch.randn(1, 16, 3, 4, 4)
fx["us_x"] = xu
with torch.no_grad():
    fx["us_out"] = us(xu, causal=False)

# --- Tiny encoder (2.3 structure) ------------------------------------------------
enc = LTX2VideoEncoder3d(
    in_channels=3,
    out_channels=4,
    block_out_channels=(8, 16, 32, 32),
    spatio_temporal_scaling=(True, True, True, True),
    layers_per_block=(1, 1, 1, 1, 1),
    downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    patch_size=4,
    patch_size_t=1,
    is_causal=True,
    spatial_padding_mode="zeros",
).eval()
add_module("enc", enc)
xe = torch.randn(1, 3, 9, 32, 32)
fx["enc_x"] = xe
with torch.no_grad():
    fx["enc_out"] = enc(xe)

# --- Tiny decoder (2.3 structure: 4 up blocks, mixed types) ----------------------
dec = LTX2VideoDecoder3d(
    in_channels=4,
    out_channels=3,
    block_out_channels=(16, 32, 32, 64),
    spatio_temporal_scaling=(True, True, True, True),
    layers_per_block=(1, 1, 1, 1, 1),
    upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    patch_size=4,
    patch_size_t=1,
    is_causal=False,
    inject_noise=(False,) * 5,
    timestep_conditioning=False,
    upsample_residual=(False,) * 4,
    upsample_factor=(2, 2, 1, 2),
    spatial_padding_mode="zeros",
).eval()
add_module("dec", dec)
xz = torch.randn(1, 4, 3, 2, 2)
fx["dec_x"] = xz
with torch.no_grad():
    fx["dec_out"] = dec(xz)
print("decoder out shape:", tuple(fx["dec_out"].shape))

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "vae_ltx23.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/vae_ltx23.safetensors")
