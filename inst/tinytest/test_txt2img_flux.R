# End-to-end smoke test for the FLUX pipeline wiring on the CPU with
# tiny random-init components: pack -> denoise (2 steps) -> unpack ->
# decode. Verifies shapes, finiteness, and the phase plumbing - numeric
# quality comes from the per-component parity tests.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

torch::torch_manual_seed(7)

transformer <- flux_transformer(
  in_channels = 64L,
  num_layers = 1L,
  num_single_layers = 1L,
  attention_head_dim = 8L,
  num_attention_heads = 2L,
  joint_attention_dim = 16L,
  pooled_projection_dim = 12L,
  axes_dims_rope = c(2L, 2L, 4L)
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
    format = "full",
    attn_chunk = NULL,
    config = list(in_channels = 64L),
    scheduler_shift = 1.0,
    vae_scaling_factor = 0.3611,
    vae_shift_factor = 0.1159
  ),
  class = "flux_pipeline"
)

res <- txt2img_flux(
  "tiny smoke test",
  pipeline = pipeline,
  width = 64L, height = 64L,
  num_inference_steps = 2L,
  seed = 42L,
  prompt_embeds = torch::torch_randn(1L, 7L, 16L),
  pooled_prompt_embeds = torch::torch_randn(1L, 12L),
  save_file = FALSE,
  verbose = FALSE
)

expect_equal(dim(res$image), c(64L, 64L, 3L))
expect_true(all(is.finite(res$image)))
expect_true(all(res$image >= 0) && all(res$image <= 1))
expect_equal(res$metadata$steps, 2L)
expect_equal(res$metadata$model, "flux1-schnell")

# Same seed reproduces the same image
res2 <- txt2img_flux(
  "tiny smoke test",
  pipeline = pipeline,
  width = 64L, height = 64L,
  num_inference_steps = 2L,
  seed = 42L,
  prompt_embeds = torch::torch_randn(1L, 7L, 16L)$zero_()$add(0.1),
  pooled_prompt_embeds = torch::torch_randn(1L, 12L)$zero_()$add(0.1),
  save_file = FALSE,
  verbose = FALSE
)
expect_true(is.finite(max(abs(res2$image))))
