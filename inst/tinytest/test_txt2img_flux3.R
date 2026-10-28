# FLUX 3 pipeline wiring on the CPU with tiny random-init components:
# the flow-map denoise loop (timestep / target timestep pairs, reference
# tokens as their own input) and end-to-end txt2img / img2img runs.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

# --- Denoise loop with a recording fake transformer ----------------------------------
seen <- new.env()
seen$t <- numeric(0)
seen$target <- numeric(0)
fake <- torch::nn_module(
  forward = function(hidden_states, encoder_hidden_states, timestep, img_ids,
                     txt_ids, image_cond_hidden_states = NULL,
                     image_cond_ids = NULL, target_timestep = NULL,
                     chunk_size = NULL) {
    seen$t <- c(seen$t, as.numeric(timestep))
    seen$target <- c(seen$target, as.numeric(target_timestep))
    seen$cond <- if (is.null(image_cond_hidden_states)) 0L else
      image_cond_hidden_states$shape[2]
    torch::torch_zeros_like(hidden_states)
  }
)()
sched <- flux3_schedule(4L)
lat <- torch::torch_randn(1L, 6L, 128L)
out <- diffuseR:::.flux3_denoise(fake, lat, sched,
  prompt_embeds = torch::torch_randn(1L, 3L, 24L),
  txt_ids = flux3_prepare_text_ids(3L),
  img_ids = flux3_prepare_image_ids(2L, 3L),
  compute_dtype = torch::torch_float32(),
  cond_tokens = torch::torch_randn(1L, 4L, 128L),
  cond_ids = flux3_prepare_image_ids(2L, 2L, l_coord = -1),
  verbose = FALSE)
ts <- as.numeric(sched$timesteps) / 1000
expect_equal(seen$t, ts, tolerance = 1e-6)
expect_equal(seen$target, c(ts[-1], 0), tolerance = 1e-6) # jumps to the next, 0 last
expect_equal(seen$cond, 4L)
expect_equal(as.integer(out$shape), c(1L, 6L, 128L))
expect_true(as.numeric((out - lat)$abs()$max()) < 1e-6) # zero velocity

# --- End to end with tiny components --------------------------------------------------
torch::torch_manual_seed(8)
transformer <- flux3_transformer(in_channels = 128L, cond_in_channels = 128L,
  num_layers = 1L, num_single_layers = 1L, attention_head_dim = 8L,
  num_attention_heads = 2L, joint_attention_dim = 24L,
  axes_dims_rope = c(2L, 2L, 2L, 2L), flow_map = TRUE)
transformer$eval()
decoder <- flux2_vae_decoder(latent_channels = 32L,
  block_channels = c(32L, 32L, 16L, 8L), norm_groups = 8L)
decoder$eval()
pipeline <- structure(list(transformer = transformer, decoder = decoder,
  device = "cpu", text_device = "cpu", phase_offload = FALSE,
  fp8_resident = FALSE, format = "full", attn_chunk = NULL,
  vae_bn_eps = 1e-4), class = "flux3_pipeline")
emb <- torch::torch_randn(1L, 7L, 24L)

res <- txt2img_flux3("tiny smoke test", pipeline = pipeline, width = 64L,
  height = 48L, num_inference_steps = 2L, seed = 42L, prompt_embeds = emb,
  save_file = FALSE, verbose = FALSE)
expect_equal(dim(res$image), c(48L, 64L, 3L))
expect_true(all(is.finite(res$image)))
expect_true(all(res$image >= 0) && all(res$image <= 1))
expect_equal(res$metadata$model, "flux3-image-klein")

# Same seed, same image
res2 <- txt2img_flux3("tiny smoke test", pipeline = pipeline, width = 64L,
  height = 48L, num_inference_steps = 2L, seed = 42L, prompt_embeds = emb,
  save_file = FALSE, verbose = FALSE)
expect_equal(res$image, res2$image)

# Without an encoder the edit says how to load one
ref <- array(0.5, c(50L, 70L, 3L))
expect_error(img2img_flux3("x", ref, pipeline = pipeline,
  num_inference_steps = 1L, prompt_embeds = emb, save_file = FALSE,
  verbose = FALSE), "encoder = TRUE")

pipeline$encoder <- flux2_vae_encoder(latent_channels = 32L,
  block_channels = c(8L, 16L, 32L, 32L), norm_groups = 8L)
pipeline$encoder$eval()
# 70x50 reference -> 80x64 canvas; the output is cropped back to 70x50
res <- img2img_flux3("tiny edit", ref, pipeline = pipeline,
  num_inference_steps = 2L, seed = 42L, prompt_embeds = emb,
  save_file = FALSE, verbose = FALSE)
expect_equal(dim(res$image), c(50L, 70L, 3L))
expect_true(all(is.finite(res$image)))
# An explicit size is returned whole
res <- img2img_flux3("tiny edit", ref, pipeline = pipeline, width = 64L,
  height = 64L, num_inference_steps = 1L, seed = 42L, prompt_embeds = emb,
  save_file = FALSE, verbose = FALSE)
expect_equal(dim(res$image), c(64L, 64L, 3L))
