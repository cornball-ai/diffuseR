# LTX-2.3 3D video VAE decoder parity fixture (torch reference),
# RANDOM-INIT, SMALL config. Instantiates diffuseR::ltx23_video_decoder3d
# at a memory-light channel/frame config with default random weights,
# assigns random per-channel latent statistics, de-normalizes a random
# latent (z * std + mean), decodes it non-causal, and saves:
#   - ltx_vae_weights.safetensors : the random state_dict (native
#       ltx23_video_decoder3d key names; the anvl loader reads this)
#   - ltx_vae.safetensors         : { z_raw, latents_mean, latents_std,
#       out } (the anvl test reads this; z_raw is pre-de-normalization,
#       out is the decoder output post-un-patchification)
#
# The LTX VAE uses per-channel RMS norm (no learned params, so the whole
# state_dict is Conv3d weight/bias pairs) and edge-replication temporal
# padding. The decoder runs with causal = FALSE (symmetric temporal pad).
# Latent de-normalization is a per-channel affine z * std + mean applied
# host-side before conv_in (scaling_factor is 1.0).
#
# Small config (config-order channels c(16,32,32,64) -> reversed
# c(64,32,32,16); latent 8, F=2, H=W=4): output [1, 3, 9, 128, 128].
#
# Usage: /home/troy/diffuseR-ltx-lib/ranvl tools/gen_fixture_ltx_vae.R

suppressMessages(library(torch))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
weights_path <- file.path(fixture_dir, "ltx_vae_weights.safetensors")
fixture_path <- file.path(fixture_dir, "ltx_vae.safetensors")

set.seed(23)
torch_manual_seed(23)

latent_channels <- 8L

dec <- diffuseR::ltx23_video_decoder3d(
    in_channels = latent_channels,
    out_channels = 3L,
    block_out_channels = c(16L, 32L, 32L, 64L),   # config order
    layers_per_block = c(1L, 1L, 1L, 1L, 2L),      # rev -> mid=2, ups=1
    spatio_temporal_scaling = c(TRUE, TRUE, TRUE, TRUE),
    upsample_type = c("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    upsample_residual = c(FALSE, FALSE, FALSE, FALSE),
    upsample_factor = c(2L, 2L, 1L, 2L),
    patch_size = 4L, patch_size_t = 1L,
    is_causal = FALSE
)
dec$eval()

# Save the random state_dict (native module keys, no decoder. prefix).
sd <- dec$state_dict()
sd <- lapply(sd, function(t) t$detach()$contiguous())
safetensors::safe_save_file(sd, weights_path)

# Random per-channel latent statistics and a random normalized latent.
latents_mean <- torch_randn(latent_channels)
latents_std <- torch_rand(latent_channels)$add(0.5)   # positive, ~[0.5, 1.5]
z_raw <- torch_randn(1L, latent_channels, 2L, 4L, 4L)

out <- with_no_grad({
    mean_v <- latents_mean$view(c(1L, -1L, 1L, 1L, 1L))
    std_v <- latents_std$view(c(1L, -1L, 1L, 1L, 1L))
    z_in <- z_raw * std_v + mean_v
    dec(z_in, causal = FALSE)
})

safetensors::safe_save_file(list(
    z_raw = z_raw$contiguous(),
    latents_mean = latents_mean$contiguous(),
    latents_std = latents_std$contiguous(),
    out = out$contiguous()
), fixture_path)

cat(sprintf("weights: %s (%.2f MB, %d tensors)\n",
            weights_path, file.size(weights_path) / 1e6, length(sd)))
cat(sprintf("fixture: %s (%.2f MB)\n",
            fixture_path, file.size(fixture_path) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
