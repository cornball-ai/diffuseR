# Traced-decode parity (R/jit_vae_ltx23.R): the per-shape jit_trace
# cache must reproduce the eager video decoder, audio decoder, and
# vocoder exactly, re-trace on new shapes, and honor the off switch.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)
torch::torch_manual_seed(11)

old_opt <- options(diffuseR.jit_vae = TRUE)
on.exit(options(old_opt), add = TRUE)

max_abs_diff <- function(a, b) as.numeric((a - b)$abs()$max())
trace_count <- function() length(ls(diffuseR:::.ltx23_vae_traces))

diffuseR:::.ltx23_release_vae_traces()

# --- Video decoder through the vae wrapper (tiled + direct paths) -------------------

vae <- ltx23_video_vae(
  latent_channels = 4L,
  block_out_channels = c(8L, 8L, 8L, 8L),
  decoder_block_out_channels = c(16L, 32L, 32L, 64L),
  layers_per_block = c(1L, 1L, 1L, 1L, 1L),
  decoder_layers_per_block = c(1L, 1L, 1L, 1L, 1L)
)
vae$eval()

z <- torch::torch_randn(1L, 4L, 3L, 4L, 6L)
torch::with_no_grad({
  op <- options(diffuseR.jit_vae = FALSE)
  ref <- vae$decode(z)
  options(op)
  op <- options(diffuseR.jit_vae = TRUE)
  out <- vae$decode(z)
  options(op)
})
expect_equal(as.integer(out$shape), as.integer(ref$shape))
expect_true(max_abs_diff(out, ref) < 1e-6)
expect_equal(trace_count(), 1L)

# New shape -> new trace, still exact
z2 <- torch::torch_randn(1L, 4L, 3L, 6L, 4L)
torch::with_no_grad({
  op <- options(diffuseR.jit_vae = FALSE)
  ref2 <- vae$decode(z2)
  options(op)
  op <- options(diffuseR.jit_vae = TRUE)
  out2 <- vae$decode(z2)
  options(op)
})
expect_true(max_abs_diff(out2, ref2) < 1e-6)
expect_equal(trace_count(), 2L)

# Same shape again reuses the cache (no growth)
torch::with_no_grad({
  op <- options(diffuseR.jit_vae = TRUE)
  invisible(vae$decode(z))
  options(op)
})
expect_equal(trace_count(), 2L)

# Tiled decode: traced tiles must equal eager tiles exactly
vae$enable_tiling(spatial = TRUE, temporal = TRUE)
vae$tile_sample_min_height <- 64L
vae$tile_sample_min_width <- 64L
vae$tile_sample_stride_height <- 32L
vae$tile_sample_stride_width <- 32L
vae$tile_sample_min_num_frames <- 16L
vae$tile_sample_stride_num_frames <- 8L
z_big <- torch::torch_randn(1L, 4L, 5L, 4L, 6L)
torch::with_no_grad({
  op <- options(diffuseR.jit_vae = FALSE)
  ref_t <- vae$decode(z_big)
  options(op)
  op <- options(diffuseR.jit_vae = TRUE)
  out_t <- vae$decode(z_big)
  options(op)
})
expect_true(max_abs_diff(out_t, ref_t) < 1e-6)

# Release drops every cached trace
diffuseR:::.ltx23_release_vae_traces()
expect_equal(trace_count(), 0L)

# --- Audio decoder ---------------------------------------------------------------------

adec <- ltx23_audio_decoder(
  base_channels = 8L, output_channels = 2L, num_res_blocks = 1L,
  latent_channels = 4L, ch_mult = c(1L, 2L), mel_bins = 8L
)
adec$eval()
za <- torch::torch_randn(1L, 4L, 5L, 2L)
torch::with_no_grad({
  ref_a <- adec(za)
  out_a <- diffuseR:::.ltx23_traced_call(adec, za)
})
expect_true(max_abs_diff(out_a, ref_a) < 1e-6)

# --- Vocoder with BWE ------------------------------------------------------------------

bwe <- ltx23_vocoder_with_bwe(
  in_channels = 8L, hidden_channels = 16L, out_channels = 2L,
  upsample_kernel_sizes = c(4L, 4L), upsample_factors = c(2L, 2L),
  resnet_kernel_sizes = c(3L), resnet_dilations = list(c(1L, 3L)),
  bwe_in_channels = 16L, bwe_hidden_channels = 8L,
  bwe_upsample_kernel_sizes = c(8L, 4L), bwe_upsample_factors = c(4L, 2L),
  bwe_resnet_kernel_sizes = c(3L), bwe_resnet_dilations = list(c(1L, 3L)),
  filter_length = 8L, hop_length = 2L, window_length = 8L,
  num_mel_channels = 8L,
  input_sampling_rate = 100L, output_sampling_rate = 400L
)
bwe$eval()
# Vocoder input is [B, C, T, M] with C * M == in_channels
mel <- torch::torch_randn(1L, 2L, 6L, 4L)
torch::with_no_grad({
  ref_v <- bwe(mel)
  out_v <- diffuseR:::.ltx23_traced_call(bwe, mel)
})
expect_true(max_abs_diff(out_v, ref_v) < 1e-6)

diffuseR:::.ltx23_release_vae_traces()
