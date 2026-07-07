# Generate FLUX transformer-block parity fixtures for the R port.
#
# Runs the diffusers reference modules (Apache-2.0) with tiny random-init
# configs and saves {state dicts, inputs, outputs} as safetensors fixtures
# for the R tinytest suite. Run via tools/gen_fixtures.sh; never executed
# at package test/run time.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.normalization import (  # noqa: E402
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from diffusers.models.transformers.transformer_flux import (  # noqa: E402
    FluxAttention,
    FluxPosEmbed,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(43)
fx = {}

# Tiny config: dim 16 = 2 heads x head_dim 8, rope axes (2, 2, 4).
DIM, HEADS, HEAD_DIM = 16, 2, 8
B, S_TXT = 2, 7
GRID_H, GRID_W = 8, 12
S_IMG = GRID_H * GRID_W  # 96
S = S_TXT + S_IMG  # 103

# Position ids and rotary freqs shared by all attention fixtures
img_ids = torch.zeros(GRID_H, GRID_W, 3)
img_ids[..., 1] = torch.arange(GRID_H)[:, None]
img_ids[..., 2] = torch.arange(GRID_W)[None, :]
img_ids = img_ids.reshape(S_IMG, 3)
ids = torch.cat([torch.zeros(S_TXT, 3), img_ids], dim=0)
rope_cos, rope_sin = FluxPosEmbed(theta=10000, axes_dim=[2, 2, 4])(ids)
fx["rope_cos"] = rope_cos
fx["rope_sin"] = rope_sin

img_x = torch.randn(B, S_IMG, DIM)
txt_x = torch.randn(B, S_TXT, DIM)
joint_x = torch.randn(B, S, DIM)
temb = torch.randn(B, DIM)
fx["img_x"] = img_x
fx["txt_x"] = txt_x
fx["joint_x"] = joint_x
fx["temb"] = temb


def add_state(prefix, module):
    for k, v in module.state_dict().items():
        fx[f"{prefix}.{k}"] = v


# --- AdaLayerNormZero -----------------------------------------------------------
adazero = AdaLayerNormZero(DIM)
add_state("adazero", adazero)
with torch.no_grad():
    x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = adazero(joint_x, emb=temb)
fx["adazero_x_norm"] = x_norm
fx["adazero_gate_msa"] = gate_msa
fx["adazero_shift_mlp"] = shift_mlp
fx["adazero_scale_mlp"] = scale_mlp
fx["adazero_gate_mlp"] = gate_mlp

# --- AdaLayerNormZeroSingle -------------------------------------------------------
adasingle = AdaLayerNormZeroSingle(DIM)
add_state("adasingle", adasingle)
with torch.no_grad():
    xs_norm, gate = adasingle(joint_x, emb=temb)
fx["adasingle_x_norm"] = xs_norm
fx["adasingle_gate"] = gate

# --- AdaLayerNormContinuous (norm_out config) --------------------------------------
adacont = AdaLayerNormContinuous(DIM, DIM, elementwise_affine=False, eps=1e-6)
add_state("adacont", adacont)
with torch.no_grad():
    fx["adacont_out"] = adacont(joint_x, temb)

# --- FluxAttention, double-stream variant (added_kv, joint attention) ---------------
attn_d = FluxAttention(
    query_dim=DIM,
    added_kv_proj_dim=DIM,
    dim_head=HEAD_DIM,
    heads=HEADS,
    out_dim=DIM,
    context_pre_only=False,
    bias=True,
    eps=1e-6,
)
add_state("attnd", attn_d)
with torch.no_grad():
    attn_out, ctx_out = attn_d(
        hidden_states=img_x,
        encoder_hidden_states=txt_x,
        image_rotary_emb=(rope_cos, rope_sin),
    )
fx["attnd_out"] = attn_out
fx["attnd_ctx_out"] = ctx_out

# --- FluxAttention, pre_only variant (single blocks) ---------------------------------
attn_s = FluxAttention(
    query_dim=DIM,
    dim_head=HEAD_DIM,
    heads=HEADS,
    out_dim=DIM,
    bias=True,
    eps=1e-6,
    pre_only=True,
)
add_state("attns", attn_s)
with torch.no_grad():
    fx["attns_out"] = attn_s(hidden_states=joint_x, image_rotary_emb=(rope_cos, rope_sin))

# --- FluxTransformerBlock (double) ----------------------------------------------------
dbl = FluxTransformerBlock(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM)
add_state("dbl", dbl)
with torch.no_grad():
    enc_out, hid_out = dbl(
        hidden_states=img_x,
        encoder_hidden_states=txt_x,
        temb=temb,
        image_rotary_emb=(rope_cos, rope_sin),
    )
fx["dbl_enc_out"] = enc_out
fx["dbl_hid_out"] = hid_out

# --- FluxSingleTransformerBlock --------------------------------------------------------
sgl = FluxSingleTransformerBlock(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM)
add_state("sgl", sgl)
with torch.no_grad():
    s_enc_out, s_hid_out = sgl(
        hidden_states=img_x,
        encoder_hidden_states=txt_x,
        temb=temb,
        image_rotary_emb=(rope_cos, rope_sin),
    )
fx["sgl_enc_out"] = s_enc_out
fx["sgl_hid_out"] = s_hid_out

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "dit_flux.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/dit_flux.safetensors")
