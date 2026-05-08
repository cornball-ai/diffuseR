# Tests for LTX-2 latent upsampler (R/upsampler_ltx2.R)

library(torch)

# --- pixel_shuffle_2d --------------------------------------------------------
ps <- diffuseR:::pixel_shuffle_2d(2L)

# [B, C*r*r, H, W] -> [B, C, H*r, W*r]
x <- torch_randn(c(1L, 16L, 3L, 3L))
y <- ps(x)
expect_equal(y$shape, c(1, 4, 6, 6))

# Values check: channel-to-spatial rearrangement
x2 <- torch_arange(0, 15)$view(c(1L, 16L, 1L, 1L))$to(dtype = torch_float32())
y2 <- ps(x2)
expect_equal(y2$shape, c(1, 4, 2, 2))

# --- upsampler_res_block -----------------------------------------------------
rb <- diffuseR:::upsampler_res_block(channels = 32L)
x <- torch_randn(c(1L, 32L, 2L, 4L, 4L))
with_no_grad({
    y <- rb(x)
})
# ResBlock preserves shape
expect_equal(y$shape, c(1, 32, 2, 4, 4))

# --- spatial_rational_resampler -----------------------------------------------
# mid_channels must be >= 32 for GroupNorm(32)
sr <- diffuseR:::spatial_rational_resampler(mid_channels = 32L, scale = 2.0)
x <- torch_randn(c(1L, 32L, 2L, 4L, 4L))
with_no_grad({
    y <- sr(x)
})
# 2x spatial upscale: [1, 32, 2, 4, 4] -> [1, 32, 2, 8, 8]
expect_equal(y$shape, c(1, 32, 2, 8, 8))

# --- latent_upsampler (small config) -----------------------------------------
# Use small config with mid_channels=32 (minimum for GroupNorm(32))
us <- diffuseR:::latent_upsampler(in_channels = 8L,
                                   mid_channels = 32L,
                                   num_blocks_per_stage = 1L,
                                   spatial_scale = 2.0)
x <- torch_randn(c(1L, 8L, 2L, 4L, 4L))
with_no_grad({
    y <- us(x)
})
# 2x spatial: [1, 8, 2, 4, 4] -> [1, 8, 2, 8, 8]
expect_equal(y$shape, c(1, 8, 2, 8, 8))

# --- upsample_video_latents --------------------------------------------------
lat_mean <- torch_zeros(8L)
lat_std <- torch_ones(8L)
x <- torch_randn(c(1L, 8L, 2L, 4L, 4L))
with_no_grad({
    y <- diffuseR:::upsample_video_latents(x, us, lat_mean, lat_std)
})
expect_equal(y$shape, c(1, 8, 2, 8, 8))

# --- load_ltx2_upsampler (needs weights, at_home only) -----------------------
if (at_home()) {
    weights_path <- "/home/troy/Wan2GP_api/models/ckpts/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    if (file.exists(weights_path)) {
        model <- diffuseR::load_ltx2_upsampler(weights_path, device = "cpu",
                                                dtype = "float32",
                                                verbose = FALSE)
        x <- torch_randn(c(1L, 128L, 2L, 4L, 4L))
        with_no_grad({
            y <- model(x)
        })
        expect_equal(y$shape, c(1, 128, 2, 8, 8))
    }
}
