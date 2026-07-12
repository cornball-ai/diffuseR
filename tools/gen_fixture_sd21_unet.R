# Generate the SD 2.1 UNet parity fixture (torch reference). Loads the
# real F16 diffusers checkpoint into diffuseR::unet_native (upcast to
# f32), runs it on a fixed random latent, scalar timestep, and random
# text embeds, and saves inputs + output (NOT weights) to an f32
# safetensors fixture. The anvl test reloads the same checkpoint via
# yq_sd_unet_load_weights and feeds these inputs.
#
# Usage: /home/troy/diffuseR-anvl-lib/ranvl tools/gen_fixture_sd21_unet.R

suppressMessages(library(torch))

ckpt <- file.path(Sys.getenv("HOME"),
                  ".local/share/R/diffuseR/sd21-diffusers/unet",
                  "diffusion_pytorch_model.safetensors")
stopifnot(file.exists(ckpt))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "sd21_unet.safetensors")

unet <- diffuseR::unet_native_from_safetensors(ckpt, verbose = TRUE)
unet$eval()

set.seed(21)
torch_manual_seed(21)
H <- 32L; W <- 32L; seq_len <- 16L; cross_dim <- 1024L
sample <- torch_randn(1L, 4L, H, W)
context <- torch_randn(1L, seq_len, cross_dim)
timestep_val <- 500
timestep <- torch_tensor(timestep_val, dtype = torch_float32())$reshape(1L)

# Same sinusoid the UNet computes internally (flip_sin_to_cos = TRUE,
# downscale_freq_shift = 0); saved so the anvl port can feed it directly.
t_sin <- diffuseR:::timestep_embedding(timestep, 320L, flip_sin_to_cos = TRUE,
                                       downscale_freq_shift = 0L)

out <- with_no_grad(unet(sample, timestep, context))

safetensors::safe_save_file(list(
    sample = sample$contiguous(),
    t_sin = t_sin$contiguous(),
    context = context$contiguous(),
    timestep = timestep$contiguous(),
    out = out$contiguous()
), fixture)

cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
