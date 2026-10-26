# FLUX.2 klein image + prompt input: VAE encoder parity against a tiny
# random-init diffusers AutoencoderKLFlux2 (fixtures from
# tools/gen_fixtures_flux2_vae_encoder.py), reference ids, preprocessing,
# and the reference-token wiring of the denoise loop.

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
vae_path <- fixture("vae_flux2_tiny.safetensors")
io_path <- fixture("vae_flux2_enc_io.safetensors")
if (!file.exists(io_path)) exit_file("flux2 vae encoder fixtures missing")
io <- safetensors::safe_load_file(io_path, framework = "torch")

max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(a - b)))
}

# --- Encoder parity -----------------------------------------------------------
enc <- load_flux2_vae_encoder(vae_path, block_channels = c(8L, 16L, 32L, 32L),
                              norm_groups = 8L, verbose = FALSE)
mean <- torch::with_no_grad(enc(io$image))
expect_equal(as.integer(mean$shape), c(1L, 32L, 4L, 6L))
expect_true(max_abs_diff(mean, io$mean) < 1e-5)

ref <- flux2_encode_image(enc, io$image)
expect_equal(c(ref$height, ref$width), c(2L, 3L))
expect_true(max_abs_diff(ref$tokens, io$tokens) < 1e-5)

# Reference ids: T = 10 for the first image, then the row/column grid
ids <- flux2_prepare_image_ids(list(c(ref$height, ref$width)))
expect_equal(as.matrix(ids), as.matrix(io$image_ids))
ids2 <- flux2_prepare_image_ids(list(c(1L, 2L), c(1L, 1L)))
expect_equal(as.numeric(ids2[, 1]), c(10, 10, 20))

# The loader refuses a file it cannot fill
expect_error(load_flux2_vae_encoder(vae_path, verbose = FALSE), "unmapped|unfilled|size")

# --- Preprocessing --------------------------------------------------------------
img <- array(0.75, c(70L, 100L, 3L))
p <- flux2_preprocess_image(img)
expect_equal(as.integer(p$shape), c(1L, 3L, 64L, 96L))
expect_true(max(abs(as.numeric(p) - 0.5)) < 1e-6)

# Center crop: a column marker survives at the right offset
img <- array(0, c(64L, 72L, 3L))
img[, 5L, ] <- 1 # crop drops 4 columns from the left
p <- flux2_preprocess_image(img)
expect_equal(as.integer(p$shape), c(1L, 3L, 64L, 64L))
expect_equal(as.numeric(p[1, 1, 1, 1]), 1)

# Grayscale and alpha inputs become RGB
expect_equal(as.integer(flux2_preprocess_image(matrix(0.5, 64L, 64L))$shape),
             c(1L, 3L, 64L, 64L))
expect_equal(as.integer(flux2_preprocess_image(array(0.5, c(64L, 64L, 4L)))$shape),
             c(1L, 3L, 64L, 64L))

# Area cap: 300 x 200 over a 100^2-pixel cap -> 122 x 81 -> 112 x 80
p <- flux2_preprocess_image(array(0.25, c(200L, 300L, 3L)), max_area = 100^2)
expect_equal(as.integer(p$shape), c(1L, 3L, 80L, 112L))
expect_true(max(abs(as.numeric(p) + 0.5)) < 1e-5) # a constant stays constant

expect_error(flux2_preprocess_image(array(0, c(40L, 100L, 3L))), "too small")
expect_error(flux2_preprocess_image(array(0, c(64L, 600L, 3L))), "aspect ratio")
expect_error(flux2_preprocess_image("no-such-file.png"), "not found")

# Lanczos weights: rows sum to 1; same size is the identity
w <- diffuseR:::.lanczos_weights(10L, 4L)
expect_equal(rowSums(w), rep(1, 4))
expect_equal(diffuseR:::.lanczos_weights(5L, 5L), diag(5), tolerance = 1e-12)

# --- Denoise wiring: reference tokens ride after the latents ---------------------
seen <- new.env()
fake <- function(hidden_states, encoder_hidden_states, timestep,
                 image_rotary_emb, chunk_size = NULL) {
  seen$len <- hidden_states$shape[2]
  hidden_states$mul(0)
}
sched <- flowmatch_scheduler_create(use_dynamic_shifting = TRUE,
                                    time_shift_type = "exponential")
sched <- flowmatch_set_timesteps(sched, 2L, mu = 1, sigmas = c(1, 0.5))
lat <- torch::torch_randn(1L, 6L, 128L)
out <- diffuseR:::.flux2_denoise(fake, lat, sched, NULL, NULL,
                                 torch::torch_float32(), verbose = FALSE,
                                 ref_tokens = torch::torch_randn(1L, 4L, 128L))
expect_equal(seen$len, 10L)
expect_equal(as.integer(out$shape), c(1L, 6L, 128L))
expect_true(max_abs_diff(out, lat) < 1e-6) # zero velocity leaves latents unchanged

# --- End to end with tiny components ------------------------------------------------
torch::torch_manual_seed(8)
transformer <- flux2_transformer(in_channels = 128L, num_layers = 1L,
                                 num_single_layers = 1L, attention_head_dim = 8L,
                                 num_attention_heads = 2L, joint_attention_dim = 24L,
                                 mlp_ratio = 3.0, axes_dims_rope = c(2L, 2L, 2L, 2L),
                                 rope_theta = 2000)
transformer$eval()
decoder <- flux2_vae_decoder(latent_channels = 32L,
                             block_channels = c(32L, 32L, 16L, 8L), norm_groups = 8L)
decoder$eval()
pipeline <- structure(list(transformer = transformer, decoder = decoder,
                           device = "cpu", text_device = "cpu", phase_offload = FALSE,
                           fp8_resident = FALSE, format = "full", attn_chunk = NULL,
                           vae_bn_eps = 1e-4),
                      class = "flux2_pipeline")
emb <- torch::torch_randn(1L, 7L, 24L)

# Without an encoder the edit says how to load one
expect_error(img2img_flux2("x", array(0.5, c(64L, 96L, 3L)), pipeline = pipeline,
                           num_inference_steps = 1L, prompt_embeds = emb,
                           save_file = FALSE, verbose = FALSE),
             "encoder = TRUE")

pipeline$encoder <- enc
res <- img2img_flux2("tiny edit", array(0.5, c(64L, 96L, 3L)), pipeline = pipeline,
                     num_inference_steps = 2L, seed = 42L, prompt_embeds = emb,
                     save_file = FALSE, verbose = FALSE)
expect_equal(dim(res$image), c(64L, 96L, 3L)) # output size follows the reference
expect_true(all(is.finite(res$image)))
expect_true(all(res$image >= 0) && all(res$image <= 1))
