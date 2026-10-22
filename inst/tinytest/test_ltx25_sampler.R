# The LTX sampler step: plain Euler for 2.3, rectified-flow ancestral
# Euler (eta 1) for 2.5, and the loop's seeded, state-restoring noise.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)
step <- diffuseR:::.ltx23_step
max_abs_diff <- function(a, b) as.numeric((a - b)$abs()$max())

torch::torch_manual_seed(11)
x <- torch::torch_randn(2L, 7L, 4L)
v <- torch::torch_randn(2L, 7L, 4L)

# eta = 0 is the 2.3 Euler step, whatever noise is passed
expect_true(max_abs_diff(step(x, v, 0.9, 0.725), x + v * (0.725 - 0.9)) < 1e-7)
expect_true(max_abs_diff(step(x, v, 0.9, 0.725, eta = 0, noise = torch::torch_randn(2L, 7L, 4L)),
                         x + v * (0.725 - 0.9)) < 1e-7)
# The last ancestral step lands on the denoised estimate x - sigma * v
expect_true(max_abs_diff(step(x, v, 0.421875, 0, eta = 1, noise = x), x - v * 0.421875) < 1e-6)

# Marginal check: x_t = (1 - s) x0 + s eps with the exact velocity
# v = eps - x0 must step to (1 - s_next) x0 + N(0, s_next^2)
local({
  torch::torch_manual_seed(12)
  n <- 400000L
  x0 <- 0.8
  for (pair in list(c(1.0, 0.99375), c(0.909375, 0.725), c(0.725, 0.421875))) {
    s <- pair[1]; sn <- pair[2]
    eps <- torch::torch_randn(n)
    xt <- eps * s + x0 * (1 - s)
    out <- step(xt, eps - x0, s, sn, eta = 1, noise = torch::torch_randn(n))
    expect_equal(out$mean()$item(), (1 - sn) * x0, tolerance = 0.01, info = s)
    expect_equal(out$std()$item(), sn, tolerance = 0.01, info = s)
  }
  # Euler on the same input keeps the original noise draw scaled to s_next
  eps <- torch::torch_randn(n)
  xt <- eps * 0.9 + x0 * 0.1
  expect_true(max_abs_diff(step(xt, eps - x0, 0.9, 0.725),
                           eps * 0.725 + x0 * 0.275) < 1e-5)
})

# --- Sampler selection -----------------------------------------------------------

eta_for <- diffuseR:::.ltx23_sampler_eta
expect_equal(eta_for("auto", "2.3.0"), 0)
expect_equal(eta_for("auto", "2.5.0"), 1)
expect_equal(eta_for("auto", "2.10.1"), 1)
expect_equal(eta_for("auto", NULL), 0)
expect_equal(eta_for("euler", "2.5.0"), 0)
expect_equal(eta_for("euler_ancestral", "2.3.0"), 1)

# --- The denoise loop on a tiny transformer ------------------------------------

model <- ltx23_transformer(
  in_channels = 4L, out_channels = 4L,
  num_attention_heads = 2L, attention_head_dim = 8L,
  cross_attention_dim = 16L,
  audio_in_channels = 4L, audio_out_channels = 4L,
  audio_num_attention_heads = 2L, audio_attention_head_dim = 4L,
  audio_cross_attention_dim = 8L,
  num_layers = 1L
)
model$eval()
torch::torch_manual_seed(13)
lat <- torch::torch_randn(1L, 24L, 4L)
alat <- torch::torch_randn(1L, 5L, 4L)
venc <- torch::torch_randn(1L, 6L, 16L)
aenc <- torch::torch_randn(1L, 6L, 8L)
tmask <- torch::torch_ones(1L, 6L)
denoise <- function(eta, noise_seed = NULL) {
  diffuseR:::.ltx23_denoise(
    model, lat, alat, c(1.0, 0.909375, 0.725, 0.0),
    venc, aenc, tmask,
    latent_frames = 2L, latent_height = 3L, latent_width = 4L,
    audio_num_frames = 5L, frame_rate = 24, device = "cpu",
    compute_dtype = torch::torch_float32(), verbose = FALSE,
    eta = eta, noise_seed = noise_seed
  )
}

a1 <- denoise(1, noise_seed = 10007)
a2 <- denoise(1, noise_seed = 10007)
a3 <- denoise(1, noise_seed = 10008)
e0 <- denoise(0)
expect_true(max_abs_diff(a1$latents, a2$latents) == 0)
expect_true(max_abs_diff(a1$audio_latents, a2$audio_latents) == 0)
expect_true(max_abs_diff(a1$latents, a3$latents) > 1e-3)
expect_true(max_abs_diff(a1$latents, e0$latents) > 1e-3)
expect_true(as.logical(a1$latents$isfinite()$all()$item()))

# A seeded ancestral run restores the caller's RNG stream
torch::torch_manual_seed(99)
invisible(denoise(1, noise_seed = 10007))
after <- torch::torch_randn(5L)
torch::torch_manual_seed(99)
expect_true(max_abs_diff(after, torch::torch_randn(5L)) == 0)
