# SD-from-safetensors component constructors: config-based CLIP arch
# detection (portable), and the VAE-decoder + CLIP text-encoder
# from-safetensors builders (validated against cached FLUX analogs -
# FLUX's VAE exercises the shared decoder, and FLUX's text_encoder IS
# the SDXL CLIP ViT-L).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}
library(diffuseR)

# --- config-based CLIP arch detection (portable) ----------------------------------

tmp <- tempfile()
dir.create(tmp)
writeLines(jsonlite::toJSON(list(vocab_size = 49408L,
  max_position_embeddings = 77L, hidden_size = 1280L,
  num_hidden_layers = 32L, num_attention_heads = 20L,
  intermediate_size = 5120L), auto_unbox = TRUE),
  file.path(tmp, "config.json"))
arch <- diffuseR:::.detect_text_encoder_config(tmp)
expect_equal(arch$vocab_size, 49408L)
expect_equal(arch$context_length, 77L)
expect_equal(arch$embed_dim, 1280L)      # bigG dims
expect_equal(arch$num_layers, 32L)
expect_equal(arch$num_heads, 20L)
expect_equal(arch$mlp_dim, 5120L)
# accepts a config.json path directly too
expect_equal(diffuseR:::.detect_text_encoder_config(
  file.path(tmp, "config.json"))$embed_dim, 1280L)
expect_error(diffuseR:::.detect_text_encoder_config(tempfile()),
  pattern = "config.json")
unlink(tmp, recursive = TRUE)

# --- against cached FLUX analogs (skipped where the cache is absent) ---------------

flux_vae <- Sys.glob(file.path("~/.cache/huggingface/hub",
  "models--black-forest-labs--FLUX.1-schnell/snapshots/*/vae"))
flux_te <- Sys.glob(file.path("~/.cache/huggingface/hub",
  "models--black-forest-labs--FLUX.1-schnell/snapshots/*/text_encoder"))

if (at_home() && length(flux_vae) && dir.exists(flux_vae[1])) {
  d <- vae_decoder_native_from_safetensors(flux_vae[1], latent_channels = 16L,
    verbose = FALSE)
  ci <- d$named_parameters()[["conv_in.weight"]]
  expect_equal(as.integer(ci$shape[2]), 16L)                 # 16-channel latent
  expect_true(as.numeric(ci$abs()$sum()$item()) > 0)         # actually loaded
  expect_false(d$training)
  rm(d)
  gc()
}

if (at_home() && length(flux_te) && dir.exists(flux_te[1])) {
  a <- diffuseR:::.detect_text_encoder_config(flux_te[1])
  expect_equal(a$embed_dim, 768L)                            # CLIP ViT-L
  expect_equal(a$num_layers, 12L)
  enc <- text_encoder_native_from_safetensors(flux_te[1], verbose = FALSE)
  expect_false(enc$training)
  tok <- torch::torch_tensor(
    matrix(c(49406L, 320L, 49407L, rep(49407L, 74)), nrow = 1),
    dtype = torch::torch_long())
  out <- torch::with_no_grad(enc(tok))
  expect_equal(as.integer(out$shape), c(1L, 77L, 768L))
  rm(enc)
  gc()
}
