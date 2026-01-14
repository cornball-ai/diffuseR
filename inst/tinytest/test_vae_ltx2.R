# Tests for LTX2 Video VAE modules

# Test 1: PerChannelRMSNorm
cat("Test 1: PerChannelRMSNorm initialization and forward\n")
norm <- per_channel_rms_norm()
expect_true(!is.null(norm), info = "RMS norm should initialize")

x <- torch::torch_randn(c(2, 4, 3, 8, 8))
y <- norm(x)
expect_equal(as.numeric(x$shape), as.numeric(y$shape), info = "Shape should be preserved")

# RMS along channel dim should be approximately 1
rms_out <- torch::torch_sqrt(torch::torch_mean(y^2, dim = 2, keepdim = TRUE) + 1e-8)
expect_true(abs(rms_out$mean()$item() - 1.0) < 0.1, info = "Output RMS should be ~1")

# Test 2: LTX2VideoCausalConv3d
cat("Test 2: LTX2VideoCausalConv3d initialization\n")
conv <- ltx2_video_causal_conv3d(
  in_channels = 4L,
  out_channels = 8L,
  kernel_size = 3L
)
expect_true(!is.null(conv), info = "Causal conv should initialize")
expect_equal(conv$kernel_size, c(3L, 3L, 3L), info = "Kernel size should be 3x3x3")

# Test 3: Causal conv forward (causal mode)
cat("Test 3: LTX2VideoCausalConv3d forward (causal)\n")
x <- torch::torch_randn(c(2, 4, 5, 8, 8))  # B, C, T, H, W
y <- conv(x, causal = TRUE)
expect_equal(as.numeric(y$shape[1]), 2, info = "Batch dim preserved")
expect_equal(as.numeric(y$shape[2]), 8, info = "Output channels correct")
expect_equal(as.numeric(y$shape[3]), 5, info = "Temporal dim preserved (stride=1)")
expect_equal(as.numeric(y$shape[4]), 8, info = "Height preserved")
expect_equal(as.numeric(y$shape[5]), 8, info = "Width preserved")

# Test 4: Causal conv forward (non-causal mode)
cat("Test 4: LTX2VideoCausalConv3d forward (non-causal)\n")
y_nc <- conv(x, causal = FALSE)
expect_equal(as.numeric(y_nc$shape), as.numeric(y$shape), info = "Non-causal shape matches causal")

# Test 5: LTX2VideoResnetBlock3d
cat("Test 5: LTX2VideoResnetBlock3d initialization and forward\n")
resnet <- ltx2_video_resnet_block3d(
  in_channels = 8L,
  out_channels = 8L
)
expect_true(!is.null(resnet), info = "ResNet block should initialize")

x <- torch::torch_randn(c(2, 8, 4, 8, 8))
y <- resnet(x, causal = TRUE)
expect_equal(as.numeric(y$shape), as.numeric(x$shape), info = "ResNet should preserve shape")

# Test 6: ResNet block with channel change
cat("Test 6: LTX2VideoResnetBlock3d with channel change\n")
resnet_change <- ltx2_video_resnet_block3d(
  in_channels = 8L,
  out_channels = 16L
)
y <- resnet_change(x, causal = TRUE)
expect_equal(as.numeric(y$shape[2]), 16, info = "Output channels should change")

# Test 7: LTXVideoDownsampler3d
# Note: For stride (2,2,2), temporal dim T must satisfy (T + stride-1) % stride == 0
# So T % 2 == 1 (T must be odd for temporal stride 2)
cat("Test 7: LTXVideoDownsampler3d\n")
downsampler <- ltx_video_downsampler3d(
  in_channels = 8L,
  out_channels = 16L,
  stride = c(2L, 2L, 2L)
)
x <- torch::torch_randn(c(2, 8, 5, 16, 16))  # T=5 (odd for stride 2 compatibility)
y <- downsampler(x, causal = TRUE)
expect_equal(as.numeric(y$shape[1]), 2, info = "Batch preserved")
expect_equal(as.numeric(y$shape[2]), 16, info = "Channels increased")
# Output T = (T_in + stride - 1) / stride = (5 + 1) / 2 = 3
expect_equal(as.numeric(y$shape[3]), 3, info = "Temporal: (5+1)/2=3")
expect_equal(as.numeric(y$shape[4]), 8, info = "Height halved")
expect_equal(as.numeric(y$shape[5]), 8, info = "Width halved")

# Test 8: LTXVideoUpsampler3d
cat("Test 8: LTXVideoUpsampler3d\n")
upsampler <- ltx_video_upsampler3d(
  in_channels = 16L,
  stride = c(2L, 2L, 2L),
  residual = TRUE,
  upscale_factor = 1L
)
x <- torch::torch_randn(c(2, 16, 2, 8, 8))
y <- upsampler(x, causal = TRUE)
expect_equal(as.numeric(y$shape[1]), 2, info = "Batch preserved")
# Output channels = in_channels * stride_prod / upscale_factor = 16 * 8 / 1 = 128
expect_equal(as.numeric(y$shape[3]), 3, info = "Temporal doubled (minus 1 for causal)")
expect_equal(as.numeric(y$shape[4]), 16, info = "Height doubled")
expect_equal(as.numeric(y$shape[5]), 16, info = "Width doubled")

# Test 9: LTX2VideoDownBlock3D
cat("Test 9: LTX2VideoDownBlock3D\n")
down_block <- ltx2_video_down_block3d(
  in_channels = 8L,
  out_channels = 16L,
  num_layers = 2L,
  spatio_temporal_scale = TRUE,
  downsample_type = "conv"
)
x <- torch::torch_randn(c(2, 8, 4, 16, 16))
y <- down_block(x, causal = TRUE)
expect_equal(as.numeric(y$shape[1]), 2, info = "Batch preserved")
expect_equal(as.numeric(y$shape[3]), 2, info = "Temporal halved")
expect_equal(as.numeric(y$shape[4]), 8, info = "Height halved")
expect_equal(as.numeric(y$shape[5]), 8, info = "Width halved")

# Test 10: LTX2VideoMidBlock3d
cat("Test 10: LTX2VideoMidBlock3d\n")
mid_block <- ltx2_video_mid_block3d(
  in_channels = 8L,
  num_layers = 2L
)
x <- torch::torch_randn(c(2, 8, 4, 8, 8))
y <- mid_block(x, causal = TRUE)
expect_equal(as.numeric(y$shape), as.numeric(x$shape), info = "Mid block preserves shape")

# Test 11: LTX2VideoUpBlock3d
# Note: In LTX2 decoder, in_channels always equals out_channels.
# The upsampler expects out_channels * upscale_factor as input.
# So input tensor channels = out_channels * upscale_factor, output = out_channels
cat("Test 11: LTX2VideoUpBlock3d\n")
up_block <- ltx2_video_up_block3d(
  in_channels = 8L,       # Same as out_channels (normal LTX2 usage)
  out_channels = 8L,
  num_layers = 2L,
  spatio_temporal_scale = TRUE,
  upsample_residual = TRUE,
  upscale_factor = 2L
)
# Input has out_channels * upscale_factor = 8 * 2 = 16 channels
x <- torch::torch_randn(c(2, 16, 2, 8, 8))
y <- up_block(x, causal = TRUE)
expect_equal(as.numeric(y$shape[1]), 2, info = "Batch preserved")
# Output has out_channels = 8 channels
expect_equal(as.numeric(y$shape[2]), 8, info = "Channels: 16 -> 8")

# Test 12: LTX2VideoEncoder3d (small config)
# Note: For spatiotemporal downsampling, T must be odd so that (T+1)/2 is integer.
# For 2 spatiotemporal downs: T=5 -> (5+1)/2=3 -> (3+1)/2=2
cat("Test 12: LTX2VideoEncoder3d\n")
encoder <- ltx2_video_encoder3d(
  in_channels = 3L,
  out_channels = 32L,
  block_out_channels = c(32L, 64L),
  spatio_temporal_scaling = c(TRUE, TRUE),
  layers_per_block = c(1L, 1L, 1L),
  downsample_type = c("spatiotemporal", "spatiotemporal"),
  patch_size = 2L,
  patch_size_t = 1L
)
# Input: T=5 (odd for spatiotemporal compat), H=32, W=32
x <- torch::torch_randn(c(1, 3, 5, 32, 32))
y <- encoder(x, causal = TRUE)
expect_equal(as.numeric(y$shape[1]), 1, info = "Encoder batch preserved")
cat(sprintf("  Encoder output shape: [%s]\n", paste(as.numeric(y$shape), collapse=", ")))

# Test 13: LTX2VideoDecoder3d (small config)
cat("Test 13: LTX2VideoDecoder3d\n")
decoder <- ltx2_video_decoder3d(
  in_channels = 32L,
  out_channels = 3L,
  block_out_channels = c(32L, 64L),
  spatio_temporal_scaling = c(TRUE, TRUE),
  layers_per_block = c(1L, 1L, 1L),
  patch_size = 2L,
  patch_size_t = 1L,
  upsample_residual = c(TRUE, TRUE),
  upsample_factor = c(2L, 2L)
)
# Use a small latent input
z <- torch::torch_randn(c(1, 32, 1, 4, 4))
out <- decoder(z, causal = TRUE)
expect_equal(as.numeric(out$shape[1]), 1, info = "Decoder batch preserved")
cat(sprintf("  Decoder output shape: [%s]\n", paste(as.numeric(out$shape), collapse=", ")))

# Test 14: DiagonalGaussianDistribution
cat("Test 14: DiagonalGaussianDistribution\n")
# Create parameters tensor (mean and logvar concatenated along channel dim)
params <- torch::torch_randn(c(2, 64, 2, 4, 4))  # 32 mean + 32 logvar
dist <- diagonal_gaussian_distribution(params)
expect_equal(as.numeric(dist$mean$shape[2]), 32, info = "Mean has half channels")
expect_equal(as.numeric(dist$logvar$shape[2]), 32, info = "Logvar has half channels")
sample <- dist$sample()
expect_equal(as.numeric(sample$shape), as.numeric(dist$mean$shape), info = "Sample shape matches mean")
mode <- dist$mode()
expect_equal(as.numeric(mode$shape), as.numeric(dist$mean$shape), info = "Mode shape matches mean")

# Test 15: Full VAE instantiation (small config)
cat("Test 15: Full ltx2_video_vae instantiation\n")
vae <- ltx2_video_vae(
  in_channels = 3L,
  out_channels = 3L,
  latent_channels = 32L,
  block_out_channels = c(32L, 64L),
  decoder_block_out_channels = c(32L, 64L),
  layers_per_block = c(1L, 1L, 1L),
  decoder_layers_per_block = c(1L, 1L, 1L),
  spatio_temporal_scaling = c(TRUE, TRUE),
  decoder_spatio_temporal_scaling = c(TRUE, TRUE),
  downsample_type = c("spatiotemporal", "spatiotemporal"),
  upsample_residual = c(TRUE, TRUE),
  upsample_factor = c(2L, 2L),
  patch_size = 2L,
  patch_size_t = 1L
)
expect_true(!is.null(vae), info = "VAE should instantiate")
expect_true(!is.null(vae$encoder), info = "VAE should have encoder")
expect_true(!is.null(vae$decoder), info = "VAE should have decoder")

# Test 16: VAE encode
cat("Test 16: VAE encode\n")
x <- torch::torch_randn(c(1, 3, 5, 32, 32))  # T=5 (odd for spatiotemporal)
torch::with_no_grad({
  posterior <- vae$encode(x, causal = TRUE)
})
expect_true(!is.null(posterior$mean), info = "Posterior should have mean")
expect_true(!is.null(posterior$std), info = "Posterior should have std")
cat(sprintf("  Latent mean shape: [%s]\n", paste(as.numeric(posterior$mean$shape), collapse=", ")))

# Test 17: VAE decode
cat("Test 17: VAE decode\n")
torch::with_no_grad({
  z <- posterior$sample()
  decoded <- vae$decode(z, causal = TRUE)
})
cat(sprintf("  Decoded shape: [%s]\n", paste(as.numeric(decoded$shape), collapse=", ")))

# Test 18: VAE enable_tiling
cat("Test 18: VAE enable_tiling\n")
vae$enable_tiling(tile_sample_min_height = 64L, tile_sample_min_width = 64L)
expect_true(vae$use_tiling, info = "Tiling should be enabled")
expect_equal(vae$tile_sample_min_height, 64L, info = "Tile height should be updated")

vae$disable_tiling()
expect_false(vae$use_tiling, info = "Tiling should be disabled")

cat("\nAll LTX2 VAE module tests completed\n")
