# Unit tests for the prefix-conditioning helpers (R/condition_ltx23.R):
# preprocessing geometry, i2v repeat semantics, prefix token placement,
# and the packed conditioning mask.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)
torch::torch_manual_seed(21)

# --- Preprocessing: resize + center-crop + range ------------------------------------

img <- array(runif(48 * 80 * 3), dim = c(48L, 80L, 3L))
fr <- ltx23_preprocess_frames(img, height = 32L, width = 32L)
expect_equal(as.integer(fr$shape), c(1L, 3L, 1L, 32L, 32L))
expect_true(as.numeric(fr$max()) <= 1)
expect_true(as.numeric(fr$min()) >= -1)

# Multi-frame input keeps frame count
vid <- array(runif(5 * 40 * 40 * 3), dim = c(5L, 40L, 40L, 3L))
frv <- ltx23_preprocess_frames(vid, height = 32L, width = 32L)
expect_equal(as.integer(frv$shape), c(1L, 3L, 5L, 32L, 32L))

# RGBA alpha is dropped
rgba <- array(runif(32 * 32 * 4), dim = c(32L, 32L, 4L))
fra <- ltx23_preprocess_frames(rgba, height = 32L, width = 32L)
expect_equal(as.integer(fra$shape)[2], 3L)

# A [0,1] gray image maps to ~0 in [-1,1]
gray <- array(0.5, dim = c(32L, 32L, 3L))
frg <- ltx23_preprocess_frames(gray, height = 32L, width = 32L)
expect_true(as.numeric(frg$abs()$max()) < 1e-6)

# --- Conditioned latent init + mask -------------------------------------------------

lf <- 4L; lh <- 2L; lw <- 3L
noise <- torch::torch_randn(1L, 128L, lf, lh, lw)

# i2v: one conditioning latent frame, repeated then masked to frame 0
cond1 <- torch::torch_randn(1L, 128L, 1L, lh, lw)
prep1 <- ltx23_prepare_conditioned_latents(cond1, lf, lh, lw, noise)
expect_equal(as.integer(prep1$latents$shape), c(1L, lf * lh * lw, 128L))
expect_equal(as.integer(prep1$conditioning_mask$shape), c(1L, lf * lh * lw))
n_tok <- lh * lw
m <- as.numeric(prep1$conditioning_mask[1, ])
expect_true(all(m[1:n_tok] == 1))
expect_true(all(m[(n_tok + 1):(lf * n_tok)] == 0))
# Conditioned tokens equal the packed condition; the rest equal noise
packed_cond <- diffuseR:::ltx23_pack_video_latents(cond1)
expect_true(as.numeric((prep1$latents$narrow(2L, 1L, n_tok) -
  packed_cond)$abs()$max()) < 1e-6)
packed_noise <- diffuseR:::ltx23_pack_video_latents(noise)
expect_true(as.numeric((prep1$latents$narrow(2L, n_tok + 1L, (lf - 1L) * n_tok) -
  packed_noise$narrow(2L, n_tok + 1L, (lf - 1L) * n_tok))$abs()$max()) < 1e-6)

# Continuation: two conditioning latent frames
cond2 <- torch::torch_randn(1L, 128L, 2L, lh, lw)
prep2 <- ltx23_prepare_conditioned_latents(cond2, lf, lh, lw, noise)
m2 <- as.numeric(prep2$conditioning_mask[1, ])
expect_true(all(m2[1:(2L * n_tok)] == 1))
expect_true(all(m2[(2L * n_tok + 1L):(lf * n_tok)] == 0))
expect_true(as.numeric((prep2$latents$narrow(2L, 1L, 2L * n_tok) -
  diffuseR:::ltx23_pack_video_latents(cond2))$abs()$max()) < 1e-6)

# cond_noise_scale partially noises ONLY the conditioned tokens
prep3 <- ltx23_prepare_conditioned_latents(cond1, lf, lh, lw, noise,
  cond_noise_scale = 0.5)
d_cond <- as.numeric((prep3$latents$narrow(2L, 1L, n_tok) -
  packed_cond)$abs()$max())
expect_true(d_cond > 1e-3)
d_free <- as.numeric((prep3$latents$narrow(2L, n_tok + 1L, (lf - 1L) * n_tok) -
  packed_noise$narrow(2L, n_tok + 1L, (lf - 1L) * n_tok))$abs()$max())
expect_true(d_free < 1e-6)

# Expected blend value: noise*0.5 + cond*0.5 on conditioned tokens
expected <- packed_noise$narrow(2L, 1L, n_tok)$mul(0.5) +
packed_cond$mul(0.5)
expect_true(as.numeric((prep3$latents$narrow(2L, 1L, n_tok) -
  expected)$abs()$max()) < 1e-6)

# --- Encoding: argmax + normalization round trip ------------------------------------

vae <- ltx23_video_vae(
  latent_channels = 4L,
  block_out_channels = c(8L, 8L, 8L, 8L),
  decoder_block_out_channels = c(4L, 8L, 8L, 16L),
  layers_per_block = c(1L, 1L, 1L, 1L, 1L),
  decoder_layers_per_block = c(1L, 1L, 1L, 1L, 1L)
)
vae$eval()
torch::with_no_grad({
  vae$latents_mean$copy_(torch::torch_randn(4L))
  vae$latents_std$copy_(torch::torch_rand(4L) + 0.5)
})
px <- torch::torch_rand(1L, 3L, 1L, 32L, 32L)$mul(2)$sub(1)
lat <- ltx23_encode_video_frames(vae, px)
expect_equal(as.integer(lat$shape), c(1L, 4L, 1L, 1L, 1L))
# Matches manual encode + normalize
torch::with_no_grad(ref_moments <- vae$encode(px))
ref <- ltx23_normalize_latents(ref_moments$mean, vae$latents_mean,
  vae$latents_std)
expect_true(as.numeric((lat - ref)$abs()$max()) < 1e-5)
