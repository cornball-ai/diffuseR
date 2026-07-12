# Generate the SD 2.1 VAE decoder parity fixture (torch reference). Loads
# the real F16 diffusers VAE checkpoint (post_quant_conv + decoder half,
# upcast to f32) into diffuseR::flux2_vae_decoder -- the shared
# AutoencoderKL post_quant_conv + decoder body, instantiated with
# latent_channels = 4 for SD 2.1 (no BatchNorm, so the module's bn buffers
# stay at their unused defaults). Runs it on a fixed random 4-channel
# latent and saves input + output (NOT weights) to an f32 safetensors
# fixture. The anvl test reloads the same checkpoint via
# yq_sd_vae_load_weights and feeds this input.
#
# The saved z is treated as the direct decode input (already scaling-factor
# rescaled); yq_sd_vae_prepare's scalar rescale is unit-checked separately
# in the test.
#
# Usage: /home/troy/diffuseR-anvl-lib/ranvl tools/gen_fixture_sd21_vae.R

suppressMessages(library(torch))

vae <- file.path(Sys.getenv("HOME"),
                 ".local/share/R/diffuseR/sd21-diffusers/vae",
                 "diffusion_pytorch_model.safetensors")
stopifnot(file.exists(vae))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "sd21_vae.safetensors")

dec <- diffuseR::flux2_vae_decoder(latent_channels = 4L,
                                   block_channels = c(512L, 512L, 256L, 128L))

# Load post_quant_conv + decoder keys via the safetensors package handle
# (framework = "torch" upcasts the checkpoint's F16 to the f32 dests on
# copy_). bn.* is absent from the SD checkpoint and unused by forward, so
# the module's bn buffers stay at their (zeros/ones) defaults.
handle <- safetensors::safetensors$new(vae, framework = "torch")
keys <- setdiff(handle$keys(), "__metadata__")
keep <- keys[startsWith(keys, "decoder.") | startsWith(keys, "post_quant_conv.")]
dests <- c(dec$named_parameters(), dec$named_buffers())
with_no_grad({
    for (key in keep) {
        dest <- dests[[key]]
        stopifnot(!is.null(dest))
        dest$copy_(handle$get_tensor(key))
    }
})
cat("loaded", length(keep), "checkpoint keys into the decoder\n")
dec$eval()

set.seed(21); torch_manual_seed(21)
z <- torch_randn(1L, 4L, 8L, 8L)
out <- with_no_grad(dec(z))

safetensors::safe_save_file(list(
    z = z$contiguous(),
    out = out$contiguous()
), fixture)
cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
