# LTX-2.3 audio VAE decoder parity fixture (torch reference), RANDOM-INIT,
# SMALL config. Instantiates diffuseR::ltx23_audio_decoder at a memory-light
# channel/frame config with default random weights, de-normalizes + unpacks
# a random packed audio latent (the pipeline's pre-decode step), decodes it
# to a mel spectrogram, and saves:
#   - ltx_audio_vae_weights.safetensors : the random state_dict (native
#       ltx23_audio_decoder key names; the anvl loader reads this)
#   - ltx_audio_vae.safetensors         : { z_packed, latents_mean,
#       latents_std, num_mel_bins, out } (the anvl test reads this; z_packed
#       is the pre-de-normalization packed [B, T, C*M] latent, out is the
#       decoder mel output post crop/pad)
#
# The audio VAE uses per-channel RMS norm (eps 1e-6, no learned params, so
# the whole state_dict is Conv2d weight/bias pairs) and ZERO-padded causal
# convolution on the time ("height") axis. Up stages upsample nearest-2x on
# both axes and their first ResNet changes channels (nin_shortcut). The
# decoder itself applies no latent normalization; the pipeline de-normalizes
# and unpacks the packed latent first, which is what the fixture replays.
#
# Small config (base 8, latent 4, ch_mult c(1,2,4) -> block channels
# c(8,16,32); mid=2 blocks, 2 blocks/stage; mel_bins 16; packed latent
# [1, T=5, C*M=20], M=5 -> decoder input [1, 4, 5, 5]): output [1, 2, 17, 16]
# (time 4T-3=17 exact; mel 4*5=20 cropped to 16).
#
# Usage: /home/troy/diffuseR-ltx-lib/ranvl tools/gen_fixture_ltx_audio_vae.R

suppressMessages(library(torch))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
weights_path <- file.path(fixture_dir, "ltx_audio_vae_weights.safetensors")
fixture_path <- file.path(fixture_dir, "ltx_audio_vae.safetensors")

set.seed(23)
torch_manual_seed(23)

base_channels <- 8L
latent_channels <- 4L
num_mel_bins <- 5L # latent mel bins M (decoder input freq)
mel_bins <- 16L # output mel bins (crop target; 4*M = 20 -> crop to 16)
frames <- 5L # packed time length T

dec <- diffuseR::ltx23_audio_decoder(
    base_channels = base_channels,
    output_channels = 2L,
    num_res_blocks = 1L,
    latent_channels = latent_channels,
    ch_mult = c(1L, 2L, 4L),
    causality_axis = "height",
    mel_bins = mel_bins
)
dec$eval()

# Save the random state_dict (native module keys, no decoder. prefix).
sd <- dec$state_dict()
sd <- lapply(sd, function(t) t$detach()$contiguous())
safetensors::safe_save_file(sd, weights_path)

# Random packed audio latent [B, T, C*M] plus per-channel packed stats.
cm <- latent_channels * num_mel_bins
z_packed <- torch_randn(1L, frames, cm)
latents_mean <- torch_randn(cm)
latents_std <- torch_rand(cm)$add(0.5) # positive, ~[0.5, 1.5]

out <- with_no_grad({
    # de-normalize on the packed [B, T, C*M] rep, then unpack to [B, C, T, M]
    z_denorm <- z_packed * latents_std + latents_mean
    z_unpacked <- z_denorm$unflatten(3L, c(-1L, num_mel_bins))$transpose(2L, 3L)
    dec(z_unpacked$contiguous())
})

safetensors::safe_save_file(list(
    z_packed = z_packed$contiguous(),
    latents_mean = latents_mean$contiguous(),
    latents_std = latents_std$contiguous(),
    num_mel_bins = torch_tensor(as.integer(num_mel_bins))$contiguous(),
    out = out$contiguous()
), fixture_path)

cat(sprintf("weights: %s (%.2f MB, %d tensors)\n",
            weights_path, file.size(weights_path) / 1e6, length(sd)))
cat(sprintf("fixture: %s (%.2f MB)\n",
            fixture_path, file.size(fixture_path) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
