# Z-Image / FLUX.1 16-channel VAE decoder parity fixture (torch
# reference), RANDOM-INIT. Instantiates diffuseR::vae_decoder_native at
# latent_channels = 16 with default random weights, applies the FLUX.1
# scaling/shift latent de-normalization, decodes a random 16-channel
# latent, and saves:
#   - zimage_vae_weights.safetensors : the random state_dict (native
#       vae_decoder_native key names; the anvl loader reads this)
#   - zimage_vae.safetensors         : { z_raw, out } (the anvl test reads
#       this; z_raw is pre-normalization, out is the decoder output)
#
# The FLUX.1 VAE has NO post_quant_conv and NO BatchNorm: de-normalization
# is a scalar affine z / scaling + shift (scaling 0.3611, shift 0.1159),
# applied host-side before conv_in.
#
# Usage: /home/troy/diffuseR-zimage-lib/ranvl tools/gen_fixture_zimage_vae.R

suppressMessages(library(torch))

scaling <- 0.3611
shift <- 0.1159

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
weights_path <- file.path(fixture_dir, "zimage_vae_weights.safetensors")
fixture_path <- file.path(fixture_dir, "zimage_vae.safetensors")

set.seed(11)
torch_manual_seed(11)

dec <- diffuseR::vae_decoder_native(
    latent_channels = 16L,
    block_channels = c(512L, 512L, 256L, 128L)
)
dec$eval()

# Save the random state_dict (native module keys, no decoder. prefix).
sd <- dec$state_dict()
sd <- lapply(sd, function(t) t$detach()$contiguous())
safetensors::safe_save_file(sd, weights_path)

# Random latent from the sampling loop (pre-normalization), small so the
# whole decode runs in a blink: 8x8 -> 64x64.
z_raw <- torch_randn(1L, 16L, 8L, 8L)
out <- with_no_grad({
    z_in <- z_raw$div(scaling)$add(shift)
    dec(z_in)
})

safetensors::safe_save_file(list(
    z_raw = z_raw$contiguous(),
    out = out$contiguous()
), fixture_path)

cat(sprintf("weights: %s (%.2f MB, %d tensors)\n",
            weights_path, file.size(weights_path) / 1e6, length(sd)))
cat(sprintf("fixture: %s (%.2f MB)\n",
            fixture_path, file.size(fixture_path) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
