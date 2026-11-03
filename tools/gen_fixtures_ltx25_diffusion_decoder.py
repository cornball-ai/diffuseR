# Generate LTX-2.5 diffusion video decoder parity fixtures for the R port.
#
# Uses diffusers' LTX2VideoDiffusionDecoderModel (Apache-2.0) with a tiny
# random config whose grids span several query tiles per axis (with a short
# last temporal tile), so the R tiled neighborhood attention meets interior
# and border windows. Attention runs on diffusers' default flex processor
# (eager on CPU). Weights are saved in the checkpoint layout the R loader
# reads (decoder.* with fused qkv, attn.proj, q_norm/k_norm, t_embedder.mlp),
# plus an encoder key and type_emb the loader must skip. Run via
# tools/gen_fixtures.sh.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers import LTX2VideoDiffusionDecoderModel  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(25)

config = dict(
    out_channels=3,
    latent_channels=8,
    patch_size=4,
    scaling_factor=1.0,
    decoder_head_dim=16,
    decoder_stage_channels=(128, 64, 32, 32, 16),
    decoder_stage_depths=(1, 1, 1, 1, 2),
    decoder_stage_kernels=((3, 3, 3), (3, 3, 3), (3, 5, 5), (3, 3, 3)),
    decoder_upsample_strides=((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
    decoder_upsample_channel_reductions=(2, 2, 1, 2),
    decoder_stage5_kernel=(5, 7, 5),
    decoder_t_emb_dim=32,
)
model = LTX2VideoDiffusionDecoderModel(**config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name or "scale_shift_table" in name:
            p.add_(0.2 * torch.randn_like(p))
    model.latents_mean.copy_(0.1 * torch.randn(8))
    model.latents_std.copy_(1 + 0.2 * torch.rand(8))

dec = model.decoder
z = torch.randn(1, 8, 3, 3, 3)
with torch.no_grad():
    context = dec.forward_stage_4(dec.forward_stages_1_to_3(z))
    shape = (1, 3, context.shape[1], context.shape[2] * 4, context.shape[3] * 4)
    # fp16-representable noise, so the fixture can store it at half size
    noise = torch.randn(shape).half().float()
    x1 = dec.denoise(context, noise, 1)
    x2 = dec.denoise(context, noise, 2)
print("context", tuple(context.shape), "pixels", tuple(x1.shape))

# Tiled decode with tiles small enough to split every axis (interior and
# merged-remnant tiles, the origin and trailing temporal tiles). A two-step
# decode draws one canvas of noise; inject it so R can reuse it.
import diffusers.models.autoencoders.ltx2_diffusion_decoder as dd_mod  # noqa: E402

tiled_noise = torch.randn(1, 3, 17, 96, 96).half().float()
dd_mod.randn_tensor = lambda shape, **kwargs: tiled_noise
model.enable_tiling(
    tile_sample_min_height=48, tile_sample_min_width=48, tile_sample_min_num_frames=8,
    tile_sample_stride_height=32, tile_sample_stride_width=32, tile_sample_stride_num_frames=4,
)
with torch.no_grad():
    xt = model.tiled_decode(z, num_inference_steps=2)
print("tiled", tuple(xt.shape))

# Checkpoint layout: undo diffusers' renames and fuse q/k/v.
renames = [
    ("t_embedder.timestep_embedder.linear_1.", "t_embedder.mlp.0."),
    ("t_embedder.timestep_embedder.linear_2.", "t_embedder.mlp.2."),
    (".attn.to_out.0.", ".attn.proj."),
    (".attn.norm_q.", ".attn.q_norm."),
    (".attn.norm_k.", ".attn.k_norm."),
]
sd = model.state_dict()
ckpt = {
    "per_channel_statistics.mean-of-means": sd.pop("latents_mean"),
    "per_channel_statistics.std-of-means": sd.pop("latents_std"),
    "encoder.conv_in.conv.weight": torch.zeros(4, 3),
    "decoder.type_emb": torch.randn(8),
}
for key in list(sd):
    if ".attn.to_q." in key:
        leaf = key.rsplit(".", 1)[1]
        prefix = key[: -len(f"to_q.{leaf}")]
        qkv = torch.cat([sd.pop(prefix + f"to_{x}.{leaf}") for x in "qkv"])
        ckpt[prefix + f"qkv.{leaf}"] = qkv
for key, value in sd.items():
    for a, b in renames:
        key = key.replace(a, b)
    ckpt[key] = value
ckpt = {k: v.contiguous() for k, v in ckpt.items()}
save_file(ckpt, os.path.join(OUT_DIR, "ltx25_diffusion_decoder_tiny.safetensors"),
          metadata={"model_version": "2.5.0"})
save_file(
    # pixels as fp16 to keep the fixture small (compare at ~2e-3)
    {"z": z, "context": context.contiguous(), "noise": noise.half(),
     "x1": x1.half(), "x2": x2.half(), "tiled_noise": tiled_noise.half(),
     "tiled": xt.half()},
    os.path.join(OUT_DIR, "ltx25_diffusion_decoder_io.safetensors"),
)
