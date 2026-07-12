# Generate the SDXL VAE decoder parity fixture (torch reference) on REAL
# weights. Loads the diffusers SDXL AutoencoderKL decoder
# (vae_decoder_native_from_safetensors, 4-channel) plus its post_quant_conv
# (read straight from the same safetensors so it matches the anvl loader
# byte-for-byte), decodes a small random 4-channel latent, and saves the
# input z (post-scaling, i.e. the post_quant_conv input) and the decoded
# pixels. The anvl test reloads the same vae/diffusion_pytorch_model.safetensors
# via yq_sdxl_vae_load_weights and compares yq_sdxl_vae_decode(z).
#
# Usage: /home/troy/diffuseR-sdxl-lib/ranvl tools/gen_fixture_sdxl_vae.R

suppressMessages(library(torch))
suppressMessages(library(diffuseR))

vae_dir <- Sys.glob(file.path(
  Sys.getenv("HOME"),
  ".cache/huggingface/hub/datasets--cornball-ai--sdxl-R",
  "snapshots/*/diffusers/vae"))[1]
stopifnot(!is.na(vae_dir), dir.exists(vae_dir))
vae_file <- file.path(vae_dir, "diffusion_pytorch_model.safetensors")

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "sdxl_vae.safetensors")

# Decoder body (decoder.* keys) as a native module, real weights (bf16/f16
# upcast to f32).
dec <- vae_decoder_native_from_safetensors(vae_dir, latent_channels = 4L,
                                           verbose = TRUE)

# post_quant_conv (1x1 conv) read from the same file so torch and anvl use
# byte-identical weights.
handle <- safetensors::safetensors$new(vae_file, framework = "torch")
pq_w <- handle$get_tensor("post_quant_conv.weight")$to(dtype = torch_float32())
pq_b <- handle$get_tensor("post_quant_conv.bias")$to(dtype = torch_float32())

set.seed(13); torch_manual_seed(13)
z <- torch_randn(1L, 4L, 8L, 8L)
out <- with_no_grad({
  pqc <- nnf_conv2d(z, weight = pq_w, bias = pq_b)
  dec(pqc)
})

safetensors::safe_save_file(list(
  z = z$contiguous(),
  out = out$contiguous()
), fixture)
cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("out shape %s sd %.4f range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
