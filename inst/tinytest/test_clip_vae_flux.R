# Parity tests for the FLUX CLIP extensions (quick_gelu, safetensors
# loader, pooled output) and the 16-channel VAE decoder (fixtures from
# tools/gen_fixtures_flux_clip_vae.py, checked in).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

fixture <- function(name) {
  p <- system.file("tinytest", "fixtures", name, package = "diffuseR")
  if (p == "") p <- file.path("fixtures", name)
  p
}
io_path <- fixture("clip_vae_io.safetensors")
clip_path <- fixture("clip_tiny.safetensors")
vae_path <- fixture("vae16_tiny.safetensors")
if (!file.exists(io_path)) exit_file("clip/vae fixtures missing")

fx <- safetensors::safe_load_file(io_path, framework = "torch")

max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(
    a$to(dtype = torch::torch_float32()) - b$to(dtype = torch::torch_float32())
  )))
}

# --- CLIP: quick_gelu forward + argmax pooled output -------------------------------

enc <- text_encoder_native(
  vocab_size = 1000L, context_length = 77L, embed_dim = 16L,
  num_layers = 2L, num_heads = 2L, mlp_dim = 32L,
  apply_final_ln = TRUE, gelu_type = "quick"
)
enc$eval()
load_text_encoder_safetensors(enc, clip_path, verbose = FALSE)

ids <- fx$clip_input_ids$to(dtype = torch::torch_long())
hidden <- torch::with_no_grad(enc(ids))
expect_true(max_abs_diff(hidden, fx$clip_last_hidden) < 1e-5)

pooled <- clip_pooled_output(hidden, ids)
expect_equal(as.integer(pooled$shape), c(2L, 16L))
expect_true(max_abs_diff(pooled, fx$clip_pooled) < 1e-5)

# --- 16-channel VAE decoder ----------------------------------------------------------

dec <- vae_decoder_native(
  latent_channels = 16L,
  block_channels = c(32L, 32L, 16L, 8L),
  norm_groups = 8L
)
dec$eval()
load_decoder_safetensors(dec, vae_path, verbose = FALSE)

img <- torch::with_no_grad(dec(fx$vae_latent))
expect_equal(as.integer(img$shape), as.integer(fx$vae_image$shape))
expect_true(max_abs_diff(img, fx$vae_image) < 1e-5)
