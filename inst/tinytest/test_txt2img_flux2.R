# End-to-end smoke test for the FLUX.2 klein pipeline wiring on the CPU
# with tiny random-init components: pack -> denoise (2 steps, dynamic
# shifting) -> unpack-with-ids -> BN denorm -> unpatchify -> decode.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

torch::torch_manual_seed(8)

transformer <- flux2_transformer(
  in_channels = 128L,
  num_layers = 1L,
  num_single_layers = 1L,
  attention_head_dim = 8L,
  num_attention_heads = 2L,
  joint_attention_dim = 24L,
  mlp_ratio = 3.0,
  axes_dims_rope = c(2L, 2L, 2L, 2L),
  rope_theta = 2000
)
transformer$eval()

decoder <- flux2_vae_decoder(
  latent_channels = 32L,
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
    config = list(in_channels = 128L),
    vae_bn_eps = 1e-4
  ),
  class = "flux2_pipeline"
)

res <- txt2img_flux2(
  "tiny smoke test",
  pipeline = pipeline,
  width = 64L, height = 64L,
  num_inference_steps = 2L,
  seed = 42L,
  prompt_embeds = torch::torch_randn(1L, 7L, 24L),
  save_file = FALSE,
  verbose = FALSE
)

expect_equal(dim(res$image), c(64L, 64L, 3L))
expect_true(all(is.finite(res$image)))
expect_true(all(res$image >= 0) && all(res$image <= 1))
expect_equal(res$metadata$steps, 2L)
expect_equal(res$metadata$model, "flux2-klein-4b")
