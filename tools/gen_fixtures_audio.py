# Generate audio VAE + vocoder parity fixtures for the LTX-2.3 R port.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import (  # noqa: E402
    LTX2AudioDecoder,
)
from diffusers.pipelines.ltx2.vocoder import (  # noqa: E402
    DownSample1d,
    LTX2Vocoder,
    LTX2VocoderWithBWE,
    SnakeBeta,
    UpSample1d,
    kaiser_sinc_filter1d,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(31)
fx = {}


def add_module(prefix, module):
    for name, p in module.state_dict().items():
        fx[f"{prefix}.{name}"] = p


# --- Audio VAE decoder (tiny, real structure) -----------------------------------
adec = LTX2AudioDecoder(
    base_channels=8,
    output_channels=2,
    num_res_blocks=1,
    in_channels=2,
    latent_channels=4,
    ch_mult=(1, 2),
    norm_type="pixel",
    causality_axis="height",
    mid_block_add_attention=False,
    mel_bins=8,
).eval()
add_module("adec", adec)
za = torch.randn(1, 4, 5, 4)
fx["adec_x"] = za
with torch.no_grad():
    fx["adec_out"] = adec(za)
print("audio decoder out:", tuple(fx["adec_out"].shape))  # expect (1, 2, 17, 8)

# --- Kaiser sinc filter + window parity -------------------------------------------
fx["kaiser_filt_12"] = kaiser_sinc_filter1d(0.25, 0.3, 12)
fx["kaiser_filt_13"] = kaiser_sinc_filter1d(0.1, 0.12, 13)
fx["kaiser_window_12"] = torch.kaiser_window(12, beta=4.7, periodic=False)

# --- Up/DownSample1d ---------------------------------------------------------------
ds1 = DownSample1d(ratio=2, kernel_size=12).eval()
us1 = UpSample1d(ratio=2, kernel_size=12).eval()
ush = UpSample1d(ratio=4, window_type="hann").eval()
xw = torch.randn(2, 3, 40)
fx["w_x"] = xw
with torch.no_grad():
    fx["w_down"] = ds1(xw)
    fx["w_up"] = us1(xw)
    fx["w_up_hann"] = ush(xw)

# --- SnakeBeta -----------------------------------------------------------------------
sb = SnakeBeta(channels=3)
with torch.no_grad():
    sb.alpha.copy_(torch.randn(3) * 0.1)
    sb.beta.copy_(torch.randn(3) * 0.1)
add_module("sb", sb)
with torch.no_grad():
    fx["sb_out"] = sb(xw)

# --- Tiny vocoder stage ----------------------------------------------------------------
voc = LTX2Vocoder(
    in_channels=8,  # 2 channels x 4 mel bins
    hidden_channels=16,
    out_channels=2,
    upsample_kernel_sizes=[4, 4],
    upsample_factors=[2, 2],
    resnet_kernel_sizes=[3],
    resnet_dilations=[[1, 3]],
    act_fn="snakebeta",
    antialias=True,
    final_act_fn=None,
    final_bias=False,
).eval()
add_module("voc", voc)
mel_in = torch.randn(1, 2, 6, 4)  # [B, C, T, M]
fx["voc_x"] = mel_in
with torch.no_grad():
    fx["voc_out"] = voc(mel_in)
print("vocoder out:", tuple(fx["voc_out"].shape))  # expect (1, 2, 24)

# --- Tiny BWE wrapper --------------------------------------------------------------------
bwe = LTX2VocoderWithBWE(
    in_channels=8,
    hidden_channels=16,
    out_channels=2,
    upsample_kernel_sizes=[4, 4],
    upsample_factors=[2, 2],
    resnet_kernel_sizes=[3],
    resnet_dilations=[[1, 3]],
    act_fn="snakebeta",
    antialias=True,
    final_act_fn=None,
    final_bias=False,
    bwe_in_channels=16,  # 2 channels x 8 mel channels
    bwe_hidden_channels=8,
    bwe_upsample_kernel_sizes=[8, 4],
    bwe_upsample_factors=[4, 2],
    bwe_resnet_kernel_sizes=[3],
    bwe_resnet_dilations=[[1, 3]],
    bwe_act_fn="snakebeta",
    bwe_antialias=True,
    bwe_final_act_fn=None,
    bwe_final_bias=False,
    filter_length=8,
    hop_length=2,
    window_length=8,
    num_mel_channels=8,
    input_sampling_rate=100,
    output_sampling_rate=400,
).eval()
# Fill STFT/mel bases with reproducible values (checkpoints carry real ones)
with torch.no_grad():
    bwe.mel_stft.stft_fn.forward_basis.copy_(torch.randn_like(bwe.mel_stft.stft_fn.forward_basis) * 0.1)
    bwe.mel_stft.mel_basis.copy_(torch.rand_like(bwe.mel_stft.mel_basis))
add_module("bwe", bwe)
with torch.no_grad():
    fx["bwe_out"] = bwe(mel_in)
print("bwe out:", tuple(fx["bwe_out"].shape))  # expect (1, 2, 96)

fx = {k: v.contiguous() for k, v in fx.items()}
save_file(fx, os.path.join(OUT_DIR, "audio_ltx23.safetensors"))
print(f"wrote {len(fx)} tensors to {OUT_DIR}/audio_ltx23.safetensors")
