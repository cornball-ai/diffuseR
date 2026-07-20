# Parity tests for the LTX-2.3 video VAE port against diffusers
# reference fixtures (tools/gen_fixtures_vae.py).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

fixture_path <- system.file("tinytest", "fixtures", "vae_ltx23.safetensors",
  package = "diffuseR")
if (fixture_path == "") fixture_path <- "fixtures/vae_ltx23.safetensors"
if (!file.exists(fixture_path)) exit_file("vae fixtures missing")

fx <- safetensors::safe_load_file(fixture_path, framework = "torch")

max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(
    a$to(dtype = torch::torch_float32()) - b$to(dtype = torch::torch_float32())
  )))
}

load_group <- function(module, prefix) {
  keys <- grep(paste0("^", prefix, "\\."), names(fx), value = TRUE)
  w <- fx[keys]
  names(w) <- sub(paste0("^", prefix, "\\."), "", keys)
  dests <- c(module$named_parameters(), module$named_buffers())
  if (!setequal(names(w), names(dests))) {
    stop(
      prefix, ": name mismatch. missing dest: ",
      paste(utils::head(setdiff(names(w), names(dests)), 3), collapse = ", "),
      " | unfilled: ",
      paste(utils::head(setdiff(names(dests), names(w)), 3), collapse = ", ")
    )
  }
  torch::with_no_grad({
    for (name in names(w)) dests[[name]]$copy_(w[[name]])
  })
  module$eval()
  module
}

# --- Causal conv -----------------------------------------------------------------

cc <- load_group(ltx23_causal_conv3d(4L, 6L, kernel_size = 3L), "cc")
torch::with_no_grad({
  out_c <- cc(fx$cc_x, causal = TRUE)
  out_nc <- cc(fx$cc_x, causal = FALSE)
})
expect_true(max_abs_diff(out_c, fx$cc_out_causal) < 1e-5)
expect_true(max_abs_diff(out_nc, fx$cc_out_noncausal) < 1e-5)

# --- Resnet block with channel-change shortcut -------------------------------------

rb <- load_group(ltx23_video_resnet_block3d(4L, 8L), "rb")
torch::with_no_grad(out_rb <- rb(fx$cc_x, causal = TRUE))
expect_true(max_abs_diff(out_rb, fx$rb_out) < 1e-5)

# --- Down/upsamplers ---------------------------------------------------------------

ds <- load_group(
  ltx23_video_downsampler3d(8L, 16L, stride = c(2L, 1L, 1L)), "ds"
)
torch::with_no_grad(out_ds <- ds(fx$ds_x, causal = TRUE))
expect_equal(as.integer(out_ds$shape), as.integer(fx$ds_out$shape))
expect_true(max_abs_diff(out_ds, fx$ds_out) < 1e-5)

us <- load_group(
  ltx23_video_upsampler3d(16L, stride = c(2L, 2L, 2L), residual = TRUE, upscale_factor = 2L),
  "us"
)
torch::with_no_grad(out_us <- us(fx$us_x, causal = FALSE))
expect_equal(as.integer(out_us$shape), as.integer(fx$us_out$shape))
expect_true(max_abs_diff(out_us, fx$us_out) < 1e-5)

# --- Tiny encoder (2.3 structure) --------------------------------------------------

enc <- load_group(
  ltx23_video_encoder3d(
    in_channels = 3L,
    out_channels = 4L,
    block_out_channels = c(8L, 16L, 32L, 32L),
    layers_per_block = c(1L, 1L, 1L, 1L, 1L),
    downsample_type = c("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
    patch_size = 4L
  ),
  "enc"
)
torch::with_no_grad(out_enc <- enc(fx$enc_x))
expect_equal(as.integer(out_enc$shape), as.integer(fx$enc_out$shape))
expect_true(max_abs_diff(out_enc, fx$enc_out) < 1e-5)

# --- Tiny decoder (2.3 structure) --------------------------------------------------

dec <- load_group(
  ltx23_video_decoder3d(
    in_channels = 4L,
    out_channels = 3L,
    block_out_channels = c(16L, 32L, 32L, 64L),
    layers_per_block = c(1L, 1L, 1L, 1L, 1L),
    upsample_type = c("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
    upsample_residual = c(FALSE, FALSE, FALSE, FALSE),
    upsample_factor = c(2L, 2L, 1L, 2L),
    patch_size = 4L
  ),
  "dec"
)
torch::with_no_grad(out_dec <- dec(fx$dec_x))
expect_equal(as.integer(out_dec$shape), c(1L, 3L, 17L, 64L, 64L))
expect_true(max_abs_diff(out_dec, fx$dec_out) < 1e-5)

# --- Latent (de)normalization round trip -------------------------------------------

lat <- torch::torch_randn(1L, 4L, 2L, 2L, 2L)
mean <- torch::torch_randn(4L)
std <- torch::torch_rand(4L) + 0.5
lat_n <- ltx23_normalize_latents(lat, mean, std)
lat_rt <- ltx23_denormalize_latents(lat_n, mean, std)
expect_true(max_abs_diff(lat, lat_rt) < 1e-5)

# --- Key mapper --------------------------------------------------------------------

expect_equal(
  ltx23_map_vae_key("vae.decoder.up_blocks.1.conv.conv.weight"),
  "decoder.up_blocks.0.upsamplers.0.conv.conv.weight"
)
expect_equal(
  ltx23_map_vae_key("vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"),
  "decoder.mid_block.resnets.0.conv1.conv.weight"
)
expect_equal(
  ltx23_map_vae_key("vae.decoder.up_blocks.8.res_blocks.1.norm3.weight"),
  "decoder.up_blocks.3.resnets.1.norm3.weight"
)
expect_equal(
  ltx23_map_vae_key("vae.encoder.down_blocks.8.res_blocks.0.conv2.conv.bias"),
  "encoder.mid_block.resnets.0.conv2.conv.bias"
)
expect_equal(
  ltx23_map_vae_key("vae.encoder.down_blocks.3.conv.conv.weight"),
  "encoder.down_blocks.1.downsamplers.0.conv.conv.weight"
)
expect_equal(
  ltx23_map_vae_key("vae.per_channel_statistics.mean-of-means"),
  "latents_mean"
)
expect_equal(ltx23_map_vae_key("vae.encoder.conv_in.conv.weight"), "encoder.conv_in.conv.weight")

# --- Tiled decode ------------------------------------------------------------------

# Reuse the tiny decoder inside a vae wrapper with small tile settings
vae_t <- ltx23_video_vae(
  latent_channels = 4L,
  block_out_channels = c(8L, 8L, 8L, 8L),
  decoder_block_out_channels = c(16L, 32L, 32L, 64L),
  layers_per_block = c(1L, 1L, 1L, 1L, 1L),
  decoder_layers_per_block = c(1L, 1L, 1L, 1L, 1L)
)
# Adopt the fixture decoder weights (same architecture)
dests_t <- vae_t$decoder$named_parameters()
w_dec <- fx[grep("^dec\\.", names(fx))]
names(w_dec) <- sub("^dec\\.", "", names(w_dec))
torch::with_no_grad({
  for (name in names(w_dec)) dests_t[[name]]$copy_(w_dec[[name]])
})
vae_t$eval()

z_big <- torch::torch_randn(1L, 4L, 5L, 4L, 6L)
torch::with_no_grad(ref_full <- vae_t$decode(z_big))
expect_equal(as.integer(ref_full$shape), c(1L, 3L, 33L, 128L, 192L))

# Spatial tiling: tiles of 2x2 latent with stride 1
vae_t$enable_tiling(spatial = TRUE, temporal = FALSE)
vae_t$tile_sample_min_height <- 64L
vae_t$tile_sample_min_width <- 64L
vae_t$tile_sample_stride_height <- 32L
vae_t$tile_sample_stride_width <- 32L
torch::with_no_grad(tiled <- vae_t$decode(z_big))
expect_equal(as.integer(tiled$shape), as.integer(ref_full$shape))
expect_true(as.logical(tiled$isfinite()$all()$item()))
# With random weights, tile borders dominate; value similarity to the
# full decode is only meaningful with trained weights. Instead check the
# degenerate case: a tile that covers the whole input must be identical.
vae_t$tile_sample_min_height <- 128L
vae_t$tile_sample_min_width <- 192L
vae_t$tile_sample_stride_height <- 96L
vae_t$tile_sample_stride_width <- 160L
torch::with_no_grad(tiled_one <- vae_t$decode(z_big))
expect_true(max_abs_diff(tiled_one, ref_full) < 1e-5)
vae_t$tile_sample_min_height <- 64L
vae_t$tile_sample_min_width <- 64L
vae_t$tile_sample_stride_height <- 32L
vae_t$tile_sample_stride_width <- 32L

# Temporal + spatial tiling
vae_t$enable_tiling(spatial = TRUE, temporal = TRUE)
vae_t$tile_sample_min_num_frames <- 16L
vae_t$tile_sample_stride_num_frames <- 8L
torch::with_no_grad(tiled_t <- vae_t$decode(z_big))
expect_equal(as.integer(tiled_t$shape), as.integer(ref_full$shape))
expect_true(as.logical(tiled_t$isfinite()$all()$item()))

# Small inputs bypass tiling entirely (identical output)
z_small <- torch::torch_randn(1L, 4L, 2L, 2L, 2L)
torch::with_no_grad({
  a <- vae_t$decode(z_small)
  vae_t$enable_tiling(spatial = FALSE, temporal = FALSE)
  b <- vae_t$decode(z_small)
})
expect_true(as.numeric((a - b)$abs()$max()) == 0)

# --- Untiled-when-fits dispatch ----------------------------------------------------

# Force-untiled: with tiling enabled and tiles smaller than the input,
# the option must route to the single full-latent forward (identical
# to the untiled reference, no seam blending)
vae_t$enable_tiling(spatial = TRUE, temporal = TRUE)
vae_t$tile_sample_min_height <- 64L
vae_t$tile_sample_min_width <- 64L
vae_t$tile_sample_stride_height <- 32L
vae_t$tile_sample_stride_width <- 32L
vae_t$tile_sample_min_num_frames <- 16L
vae_t$tile_sample_stride_num_frames <- 8L
options(diffuseR.vae_untiled = TRUE)
torch::with_no_grad(forced <- vae_t$decode(z_big))
expect_true(max_abs_diff(forced, ref_full) < 1e-6)

# Force-tiled: FALSE must reproduce the tiled path exactly
options(diffuseR.vae_untiled = FALSE)
torch::with_no_grad(forced_tiled <- vae_t$decode(z_big))
expect_true(max_abs_diff(forced_tiled, tiled_t) < 1e-6)

# "auto" on CPU keeps the tiled path (no VRAM to test against)
options(diffuseR.vae_untiled = NULL)
torch::with_no_grad(auto_cpu <- vae_t$decode(z_big))
expect_true(max_abs_diff(auto_cpu, tiled_t) < 1e-6)
