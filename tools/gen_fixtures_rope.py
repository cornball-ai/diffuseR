# Generate RoPE parity fixtures for the LTX-2.3 R port.
#
# Runs the diffusers reference implementation (Apache-2.0) on small fixed
# inputs and saves {inputs, expected outputs} as safetensors fixtures that
# the R tinytest suite compares against. Run via tools/gen_fixtures.sh;
# never executed at package test/run time.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.transformers.transformer_ltx2 import (  # noqa: E402
    LTX2AudioVideoRotaryPosEmbed,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
fx = {}

# --- apply_split_rotary_emb: 4D per-head input -------------------------------
B, H, T, D = 2, 4, 6, 16
x4 = torch.randn(B, H, T, D)
cos4 = torch.randn(B, H, T, D // 2)
sin4 = torch.randn(B, H, T, D // 2)
fx["split4_x"] = x4
fx["split4_cos"] = cos4
fx["split4_sin"] = sin4
fx["split4_out"] = apply_split_rotary_emb(x4, (cos4, sin4))

# --- apply_split_rotary_emb: 3D input reshaped per-head ----------------------
x3 = torch.randn(B, T, H * D)
fx["split3_x"] = x3
fx["split3_out"] = apply_split_rotary_emb(x3, (cos4, sin4))

# --- apply_interleaved_rotary_emb ---------------------------------------------
S, C = 6, 16
xi = torch.randn(B, S, C)
cosi = torch.randn(B, S, C)
sini = torch.randn(B, S, C)
fx["inter_x"] = xi
fx["inter_cos"] = cosi
fx["inter_sin"] = sini
fx["inter_out"] = apply_interleaved_rotary_emb(xi, (cosi, sini))

# --- embedder: video coords + split freqs -------------------------------------
rope_v = LTX2AudioVideoRotaryPosEmbed(
    dim=64,
    base_num_frames=20,
    base_height=2048,
    base_width=2048,
    scale_factors=(8, 32, 32),
    theta=10000.0,
    modality="video",
    double_precision=True,
    rope_type="split",
    num_attention_heads=4,
)
vc = rope_v.prepare_video_coords(batch_size=2, num_frames=3, height=4, width=6, device="cpu", fps=24.0)
fx["video_coords"] = vc
vcos, vsin = rope_v(vc)
fx["video_cos"] = vcos
fx["video_sin"] = vsin

# fp16 dtype preservation through apply (values compared in the R test)
xv = torch.randn(2, 4, 3 * 4 * 6, 16, dtype=torch.float16)
fx["video_x_f16"] = xv
fx["video_out_f16"] = apply_split_rotary_emb(xv, (vcos, vsin))

# --- embedder: audio coords + split freqs --------------------------------------
rope_a = LTX2AudioVideoRotaryPosEmbed(
    dim=32,
    base_num_frames=20,
    sampling_rate=16000,
    hop_length=160,
    scale_factors=(8, 32, 32),
    modality="audio",
    double_precision=True,
    rope_type="split",
    num_attention_heads=4,
)
ac = rope_a.prepare_audio_coords(batch_size=2, num_frames=5, device="cpu", shift=0)
fx["audio_coords"] = ac
acos, asin = rope_a(ac)
fx["audio_cos"] = acos
fx["audio_sin"] = asin

# --- embedder: interleaved variant ---------------------------------------------
rope_i = LTX2AudioVideoRotaryPosEmbed(
    dim=64,
    modality="video",
    double_precision=True,
    rope_type="interleaved",
    num_attention_heads=4,
)
icos, isin = rope_i(vc)
fx["video_inter_cos"] = icos
fx["video_inter_sin"] = isin

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "rope_ltx23.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/rope_ltx23.safetensors")
