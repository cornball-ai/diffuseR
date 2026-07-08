# Generate Z-Image phase-1 parity fixtures for the R port: 3-axis RoPE
# (theta 256, f32-angle cast), coordinate-grid position ids with the
# SEQ_MULTI_OF padding scheme, patchify/unpatchify, the Z-Image timestep
# sinusoid, and the static shift-3.0 sigma schedule.
#
# Runs the diffusers reference (Apache-2.0) on small fixed inputs. Run
# via tools/gen_fixtures.sh; never executed at package test/run time.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.transformers.transformer_z_image import (  # noqa: E402
    RopeEmbedder,
    TimestepEmbedder,
    ZImageTransformer2DModel,
)
from diffusers.pipelines.z_image.pipeline_z_image import (  # noqa: E402
    get_default_z_image_sigmas,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(53)
fx = {}

# --- position ids via patchify_and_embed (tiny model, real geometry) -----------
# cap len 37 pads to 64; image 6x10 = 60 tokens pads to 64. Asymmetric H/W
# so an axis swap fails.
tiny = ZImageTransformer2DModel(
    in_channels=16,
    dim=32,
    n_layers=1,
    n_refiner_layers=1,
    n_heads=2,
    n_kv_heads=2,
    cap_feat_dim=24,
    axes_dims=[4, 6, 6],
    axes_lens=[64, 32, 32],
)

CAP_LEN = 37
IMG = torch.randn(16, 1, 12, 20)
CAP = torch.randn(CAP_LEN, 24)

(
    all_img_out,
    all_cap_out,
    all_img_size,
    all_img_pos_ids,
    all_cap_pos_ids,
    all_img_pad_mask,
    all_cap_pad_mask,
) = tiny.patchify_and_embed([IMG], [CAP], patch_size=2, f_patch_size=1)

fx["img"] = IMG
fx["cap"] = CAP
fx["img_tokens_padded"] = all_img_out[0]              # [64, 64] pads = last patch
fx["cap_feats_padded"] = all_cap_out[0]               # [64, 24] pads = last row
fx["img_pos_ids"] = all_img_pos_ids[0].float()        # [64, 3], pads (0,0,0)
# cap pos ids carry a dead (0,0,0) tail that _prepare_sequence truncates;
# keep only the effective first cap_padded rows (the 1..64 ramp)
fx["cap_pos_ids"] = all_cap_pos_ids[0][:64].float()   # [64, 3]
fx["img_pad_mask"] = all_img_pad_mask[0].float()
fx["cap_pad_mask"] = all_cap_pad_mask[0].float()

# --- unpatchify: slices [:ori_len] off the unified sequence ---------------------
uni_tokens = torch.randn(80, 2 * 2 * 1 * 16)  # 64 img tokens + cap tail
unpat = tiny.unpatchify([uni_tokens.clone()], [(1, 12, 20)], patch_size=2, f_patch_size=1)[0]
fx["unpat_tokens"] = uni_tokens
fx["unpat_out"] = unpat                                # [16, 1, 12, 20]

# --- RopeEmbedder at the real Turbo config --------------------------------------
# Positions include the axis maxima to expose the f32-angle-cast rounding.
rope = RopeEmbedder(theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512])
ids = torch.tensor(
    [
        [1, 0, 0],
        [2, 0, 0],
        [64, 0, 0],
        [65, 0, 0],
        [65, 3, 7],
        [65, 5, 9],
        [130, 45, 60],
        [1535, 511, 511],
        [0, 0, 0],
    ],
    dtype=torch.int32,
)
freqs_cis = rope(ids)  # [9, 64] complex64
fx["rope_ids"] = ids.float()
fx["rope_cos"] = freqs_cis.real.repeat_interleave(2, dim=-1)  # [9, 128]
fx["rope_sin"] = freqs_cis.imag.repeat_interleave(2, dim=-1)


# --- RoPE application (processor convention: x [B, S, H, D]) ---------------------
def apply_rotary_emb(x_in, freqs_cis):
    x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(2)
    x_out = torch.view_as_real(x * freqs_cis).flatten(3)
    return x_out.type_as(x_in)


rope_x = torch.randn(1, 9, 3, 128)
fx["rope_x"] = rope_x
fx["rope_out"] = apply_rotary_emb(rope_x, freqs_cis.unsqueeze(0))

# --- timestep sinusoid ------------------------------------------------------------
# Model input is t * t_scale with pipeline t = (1000 - t_sched)/1000 in [0, 1].
t_in = torch.tensor([0.0, 125.0, 437.5, 875.0, 1000.0], dtype=torch.float32)
fx["t_emb_in"] = t_in
fx["t_emb_out"] = TimestepEmbedder.timestep_embedding(t_in, 256)  # [5, 256]

# --- scheduler: static shift 3.0 on linspace(1, 1/N, N) --------------------------
for n in (4, 8):
    sched = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False
    )
    sched.set_timesteps(sigmas=get_default_z_image_sigmas(n))
    fx[f"sched_sigmas_{n}"] = sched.sigmas.float()        # [n + 1]
    fx[f"sched_timesteps_{n}"] = sched.timesteps.float()  # [n]

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "rope_zimage.safetensors"),
          metadata={"purpose": "diffuseR Z-Image test fixture"})
print(f"wrote {len(fx)} tensors to {OUT_DIR}/rope_zimage.safetensors")
