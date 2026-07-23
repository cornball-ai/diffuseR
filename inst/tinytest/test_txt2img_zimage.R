# End-to-end smoke test for the Z-Image Turbo pipeline wiring on the CPU
# with tiny random-init components: latents -> denoise (2 steps, static
# shift, reversed timestep + negated output) -> scale/shift -> decode.
# Numeric quality comes from the per-component parity tests; this checks
# the phase plumbing (onload/offload routing, decode).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

torch::torch_manual_seed(9)

transformer <- zimage_transformer(
  in_channels = 16L, dim = 48L, n_layers = 1L, n_refiner_layers = 1L,
  n_heads = 2L, cap_feat_dim = 24L, axes_dims = c(8L, 8L, 8L)
)
transformer$eval()

decoder <- vae_decoder_native(
  latent_channels = 16L,
  block_channels = c(32L, 32L, 16L, 8L),
  norm_groups = 8L
)
decoder$eval()

pipeline <- structure(
  list(
    transformer = transformer,
    decoder = decoder,
    device = "cpu",
    text_device = "cpu",
    phase_offload = FALSE,
    fp8_resident = FALSE,
    format = "full",
    attn_chunk = NULL,
    config = list(in_channels = 16L),
    sched_shift = 3.0,
    te_penult_layer = 35L,
    vae_scaling_factor = 0.3611,
    vae_shift_factor = 0.1159
  ),
  class = "zimage_pipeline"
)

res <- txt2img_zimage(
  "tiny smoke test",
  pipeline = pipeline,
  width = 64L, height = 64L,
  num_inference_steps = 2L,
  seed = 42L,
  prompt_embeds = torch::torch_randn(7L, 24L),
  save_file = FALSE,
  verbose = FALSE
)

expect_equal(dim(res$image), c(64L, 64L, 3L))
expect_true(all(is.finite(res$image)))
expect_true(all(res$image >= 0) && all(res$image <= 1))
expect_equal(res$metadata$steps, 2L)
expect_equal(res$metadata$model, "zimage-turbo")
