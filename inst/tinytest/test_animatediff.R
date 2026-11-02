# AnimateDiff motion modules, SD 1.5 head layout, DDIM timesteps and prompt
# keyframes. Tiny random modules; no weights needed.

library(tinytest)
library(diffuseR)

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
    exit_file("torch not fully installed")
}

# SD 2.1 fixes the head size, SD 1.5 the head count
expect_equal(diffuseR:::.unet_heads(640L, 64L), c(10L, 64L))
expect_equal(diffuseR:::.unet_heads(640L, 64L, 8L), c(8L, 80L))

# Original AnimateDiff keys map onto the diffusers adapter layout
remap <- diffuseR:::.motion_remap
expect_equal(remap("down_blocks.1.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.1.to_q.weight"),
             "down_blocks.1.motion_modules.0.transformer_blocks.0.attn2.to_q.weight")
expect_equal(remap("up_blocks.0.motion_modules.2.temporal_transformer.transformer_blocks.0.ff_norm.bias"),
             "up_blocks.0.motion_modules.2.transformer_blocks.0.norm3.bias")
expect_equal(remap("mid_block.motion_modules.0.temporal_transformer.transformer_blocks.0.norms.0.weight"),
             "mid_block.motion_modules.0.transformer_blocks.0.norm1.weight")
expect_equal(remap("down_blocks.0.motion_modules.0.proj_in.weight"),
             "down_blocks.0.motion_modules.0.proj_in.weight")

# Sinusoidal frame positions: sin on even channels, cos on odd
pe <- as.array(diffuseR:::.motion_positions(8L, 4L))
expect_equal(dim(pe), c(1L, 4L, 8L))
expect_equal(pe[1, 3, 1], sin(2), tolerance = 1e-6)
expect_equal(pe[1, 3, 2], cos(2), tolerance = 1e-6)
expect_equal(pe[1, 2, 3], sin(1 * exp(2 * -log(10000) / 8)), tolerance = 1e-6)

# Adapter layout: up blocks carry one more module than down blocks
ad <- motion_adapter(c(32L, 64L), layers_per_block = 1L, use_mid_block = TRUE,
                     num_heads = 2L, max_len = 8L)
nm <- names(ad$named_parameters())
expect_true("down_blocks.0.motion_modules.0.norm.weight" %in% nm)
expect_true("down_blocks.1.motion_modules.0.transformer_blocks.0.attn1.to_out.0.weight" %in% nm)
expect_true("up_blocks.1.motion_modules.1.transformer_blocks.0.ff.net.2.bias" %in% nm)
expect_false(any(grepl("^up_blocks\\.0\\.motion_modules\\.2\\.", nm)))
expect_true(any(startsWith(nm, "mid_block.")))
expect_false(any(grepl("\\.pe$", names(ad$state_dict()))))
ad_nomid <- motion_adapter(c(32L, 64L), layers_per_block = 1L, use_mid_block = FALSE,
                           num_heads = 2L, max_len = 8L)
expect_false(any(startsWith(names(ad_nomid$named_parameters()), "mid_block.")))

# A motion module keeps [B*F, C, H, W] and mixes frames
torch::torch_manual_seed(1)
mm <- diffuseR:::motion_transformer_3d(32L, heads = 2L, max_len = 8L)
x <- torch::torch_randn(2L * 3L, 32L, 4L, 4L)
y <- torch::with_no_grad(mm(x, 3L))
expect_equal(as.integer(y$shape), c(6L, 32L, 4L, 4L))
x2 <- x$clone()
torch::with_no_grad(x2[2, , , ]$add_(1))
y2 <- torch::with_no_grad(mm(x2, 3L))
# frame 2 of clip 1 changed: frames 1 and 3 of clip 1 move, clip 2 does not
expect_true((y2[1, ..] - y[1, ..])$abs()$max()$item() > 1e-4)
expect_true((y2[3, ..] - y[3, ..])$abs()$max()$item() > 1e-4)
expect_true((y2[4:6, ..] - y[4:6, ..])$abs()$max()$item() < 1e-5)

# Tiny SD 1.5-shaped UNet + adapter. With every motion proj_out zeroed the
# modules are identities, so each frame equals the image UNet's output.
unet <- unet_native(block_out_channels = c(32L, 32L, 64L, 64L), layers_per_block = 1L,
                    cross_attention_dim = 16L, num_attention_heads = 2L)
ad <- motion_adapter(c(32L, 32L, 64L, 64L), layers_per_block = 1L,
                     use_mid_block = TRUE, num_heads = 2L, max_len = 8L)
um <- unet_motion(unet, ad)
um$eval()
lat <- torch::torch_randn(1L, 4L, 3L, 16L, 16L)
ctx <- torch::torch_randn(1L, 77L, 16L)
t <- torch::torch_tensor(500L)
out <- torch::with_no_grad(um(lat, t, ctx))
expect_equal(as.integer(out$shape), c(1L, 4L, 3L, 16L, 16L))
torch::with_no_grad({
    for (p in names(ad$named_parameters())) {
        if (grepl("\\.proj_out\\.", p)) ad$named_parameters()[[p]]$zero_()
    }
    out0 <- um(lat, t, ctx)
    img <- unet(lat[, , 2, , ], t, ctx)
})
expect_true((out0[, , 2, , ] - img)$abs()$max()$item() < 1e-5)
# per-frame context: [F, 77, D] rows condition each frame
ctx_f <- torch::torch_randn(3L, 77L, 16L)
out_f <- torch::with_no_grad(um(lat, t, ctx_f))
img3 <- torch::with_no_grad(unet(lat[, , 3, , ], t, ctx_f[3:3, , ]))
expect_true((out_f[, , 3, , ] - img3)$abs()$max()$item() < 1e-5)
expect_error(um(torch::torch_randn(1L, 4L, 9L, 16L, 16L), t, ctx), "8-frame window")

# DDIM timesteps as diffusers: linspace (rounded half to even) and leading
expect_equal(diffuseR:::.ddim_timesteps(6L, "linspace"), c(999L, 799L, 599L, 400L, 200L, 0L))
lead <- diffuseR:::.ddim_timesteps(50L, "leading")
expect_equal(c(lead[1], lead[50]), c(981L, 1L))
ac <- diffuseR:::.ddim_alphas_cumprod("linear")
expect_equal(length(ac), 1000L)
s <- torch::torch_ones(2)
e <- torch::torch_zeros(2)
# zero noise prediction: x0 = x / sqrt(a_t); last step lands on alphas_cumprod[1]
last <- diffuseR:::.ddim_step(s, e, 0L, 6L, ac)
expect_equal(as.numeric(last[1]), sqrt(ac[1]) / sqrt(ac[1]), tolerance = 1e-6)
mid <- diffuseR:::.ddim_step(s, e, 599L, 6L, ac)
# previous timestep is t - 1000 %/% n = 433, i.e. alphas_cumprod[434]
expect_equal(as.numeric(mid[1]), sqrt(ac[434]) / sqrt(ac[600]), tolerance = 1e-6)

# Prompt keyframes blend linearly and hold outside the keyframes
enc <- function(p) torch::torch_full(c(1L, 2L, 3L), switch(p, a = 0, b = 10))
pe <- diffuseR:::.animatediff_prompt_embeds(c("2" = "a", "6" = "b"), 8L, enc)
v <- as.numeric(pe[, 1, 1])
expect_equal(v, c(0, 0, 2.5, 5, 7.5, 10, 10, 10))
expect_equal(as.integer(diffuseR:::.animatediff_prompt_embeds("a", 8L, enc)$shape[1]), 1L)
expect_error(diffuseR:::.animatediff_prompt_embeds(c("0" = "a"), 8L, enc), "frame number")
