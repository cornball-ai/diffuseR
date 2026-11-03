# LTX-2.5 diffusion video decoder: tiled neighborhood attention against a
# dense masked reference, and the whole decoder against diffusers
# fixtures (tools/gen_fixtures_ltx25_diffusion_decoder.py).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

max_abs_diff <- function(a, b) (a - b)$abs()$max()$item()

# ---- neighborhood attention ------------------------------------------------
# Dense reference: every query against every key, masked to its window
# (diffusers _neighborhood_block_mask: centred, shifted inward at borders)
na_dense <- function(q, k, v, kernel) {
  s <- q$shape
  n <- s[2] * s[3] * s[4]
  pos <- expand.grid(w = seq_len(s[4]) - 1L, h = seq_len(s[3]) - 1L,
                     t = seq_len(s[2]) - 1L)
  start <- function(x, n, kk) pmin(pmax(x - kk %/% 2L, 0L), n - kk)
  inwin <- function(x, n, kk) {
    st <- start(x, n, kk)
    outer(st, x, function(a, b) b >= a & b < a + kk)
  }
  m <- inwin(pos$t, s[2], kernel[1]) & inwin(pos$h, s[3], kernel[2]) &
    inwin(pos$w, s[4], kernel[3])
  flat <- function(x) x$reshape(c(s[1], n, s[5], s[6]))$permute(c(1L, 3L, 2L, 4L))
  o <- torch::torch_scaled_dot_product_attention(
    flat(q), flat(k), flat(v), attn_mask = torch::torch_tensor(m), scale = 1)
  o$permute(c(1L, 3L, 2L, 4L))$reshape(c(s[1:4], s[5] * s[6]))
}

torch::torch_manual_seed(3)
for (case in list(list(grid = c(5L, 9L, 11L), kernel = c(3L, 5L, 7L)),
                  list(grid = c(11L, 6L, 7L), kernel = c(11L, 3L, 5L)),
                  list(grid = c(4L, 4L, 4L), kernel = c(3L, 3L, 3L)))) {
  g <- case$grid
  q <- torch::torch_randn(2L, g[1], g[2], g[3], 2L, 8L)
  k <- torch::torch_randn_like(q)
  v <- torch::torch_randn_like(q)
  # small tiles so every grid has interior, border and short tiles; a
  # small budget so classes split across several calls
  plan <- diffuseR:::.ltx25_na_plan(g, case$kernel, "cpu", tile = c(3L, 4L, 4L),
                                    budget = 2000)
  got <- diffuseR:::.ltx25_na3d(q, k, v, plan)
  expect_true(max_abs_diff(got, na_dense(q, k, v, case$kernel)) < 1e-5,
              info = paste(g, collapse = "x"))
}
expect_error(diffuseR:::.ltx25_na_plan(c(2L, 9L, 9L), c(3L, 3L, 3L), "cpu"),
             "at least its kernel")

# ---- patchify is a bijection -----------------------------------------------
x <- torch::torch_randn(1L, 3L, 5L, 8L, 12L)
px <- diffuseR:::.ltx25_dd_patchify(x, 4L)
expect_equal(as.integer(px$shape), c(1L, 48L, 5L, 2L, 3L))
expect_equal(max_abs_diff(diffuseR:::.ltx25_dd_unpatchify(px, 4L), x), 0)

# ---- key mapping -----------------------------------------------------------
mk <- diffuseR:::.ltx25_dd_map_key
expect_equal(mk("decoder.diff_blocks.3.attn.proj.weight"),
             "diff_blocks.3.attn.to_out.0.weight")
expect_equal(mk("decoder.t_embedder.mlp.2.bias"),
             "t_embedder.timestep_embedder.linear_2.bias")
expect_equal(mk("per_channel_statistics.std-of-means"), "latents_std")
expect_true(is.na(mk("decoder.type_emb")))
expect_true(is.na(mk("encoder.conv_in.conv.weight")))

# ---- the decoder against diffusers ------------------------------------------
fx_path <- function(f) {
  p <- system.file("tinytest", "fixtures", f, package = "diffuseR")
  if (p == "") p <- file.path("fixtures", f)
  p
}
ckpt <- fx_path("ltx25_diffusion_decoder_tiny.safetensors")
io_path <- fx_path("ltx25_diffusion_decoder_io.safetensors")
if (!file.exists(ckpt) || !file.exists(io_path)) exit_file("fixtures missing")
io <- safetensors::safe_load_file(io_path, framework = "torch")
f32 <- torch::torch_float32()
noise <- io$noise$to(dtype = f32)

m <- ltx25_load_diffusion_decoder(
  ckpt, dtype = f32, latent_channels = 8L, head_dim = 16L,
  stage_channels = c(128L, 64L, 32L, 32L, 16L),
  stage_depths = c(1L, 1L, 1L, 1L, 2L),
  stage_kernels = list(c(3L, 3L, 3L), c(3L, 3L, 3L), c(3L, 5L, 5L), c(3L, 3L, 3L)),
  stage5_kernel = c(5L, 7L, 5L), t_emb_dim = 32L)
torch::with_no_grad({
  ctx <- m$context(io$z)
  x1 <- m$decode(io$z, noise = noise)
  x2 <- m$denoise(io$context, noise, 2L)
})
expect_equal(as.integer(ctx$shape), c(1L, 17L, 24L, 24L, 16L))
expect_true(max_abs_diff(ctx, io$context) < 1e-5)
# 3 latent frames -> 17 pixel frames; pixels stored as fp16
expect_equal(as.integer(x1$shape), c(1L, 3L, 17L, 96L, 96L))
expect_true(max_abs_diff(x1, io$x1$to(dtype = f32)) < 5e-3)
expect_true(max_abs_diff(x2, io$x2$to(dtype = f32)) < 5e-3)
# tiled decode (diffusers tiled_decode): every axis split, remnants
# merged, origin/trailing temporal tiles, blended seams
m$enable_tiling(height = 48L, width = 48L, frames = 8L, stride_height = 32L,
                stride_width = 32L, stride_frames = 4L)
expect_equal(diffuseR:::.ltx25_tile_intervals(9L, 4L, 2L, 3L),
             list(c(0L, 4L), c(2L, 6L), c(4L, 8L), c(6L, 9L)))
xt <- torch::with_no_grad(m$decode(io$z, steps = 2L,
                                   noise = io$tiled_noise$to(dtype = f32)))
expect_equal(as.integer(xt$shape), c(1L, 3L, 17L, 96L, 96L))
expect_true(max_abs_diff(xt, io$tiled$to(dtype = f32)) < 5e-3)
# single-step tiled decode draws per-tile noise: right shape, finite
x1t <- torch::with_no_grad(m$decode(io$z))
expect_equal(as.integer(x1t$shape), c(1L, 3L, 17L, 96L, 96L))
expect_true(is.finite(x1t$sum()$item()))
m$disable_tiling()

# the stats come from per_channel_statistics, the encoder and type_emb
# keys are skipped
expect_true(m$latents_std$min()$item() > 1)
