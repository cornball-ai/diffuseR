# Parity: yq_sdxl_unet (jitted, full SDXL UNet) vs the diffuseR/torch
# reference from tools/gen_fixture_sdxl_unet.R, on RANDOM-INIT weights
# (architecture parity). Small spatial input (1x4x16x16, 16 text tokens),
# but the full production config (channels 320/640/1280, transformer
# depth 0/2/10, cross-attn 2048, add-embedding conditioning).

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
weights_file <- file.path(fixture_dir, "sdxl_unet_weights.safetensors")
io_file <- file.path(fixture_dir, "sdxl_unet_io.safetensors")
if (!file.exists(weights_file) || !file.exists(io_file)) {
    exit_file("fixture missing; run tools/gen_fixture_sdxl_unet.R")
}

f <- anvl::nv_read(io_file)
w <- yq_sdxl_unet_load_weights(weights_file, device = "cpu")

net <- yq_sdxl_unet()
fj <- anvl::jit(net)
out <- fj(f$sample, f$t_sin, f$time_ids_sin, f$text_embeds, f$context, w)

got <- as.array(out); want <- as.array(f$out)
max_abs <- max(abs(got - want))
scale <- max(abs(want))
cat(sprintf("sdxl unet parity: max %.3e mean %.3e cor %.6f (out scale %.3f)\n",
            max_abs, mean(abs(got - want)),
            cor(as.vector(got), as.vector(want)), scale))
expect_equal(dim(got), c(1L, 4L, 16L, 16L))
expect_true(cor(as.vector(got), as.vector(want)) > 0.99999)
# f32 tolerance relative to output scale (accumulated over a deep conv net
# with depth-10 transformer stacks)
expect_true(max_abs < 1e-3 * scale)
expect_true(mean(abs(got - want)) < 1e-4 * scale)

# Host-side sinusoids match the reference's internal timestep embeddings
# (f32 tolerance: cos/sin of large arguments round differently in f32 vs
# f64-then-cast, but the values are otherwise identical, cor == 1).
t_sin_host <- as.array(yq_sdxl_time_embed(as.numeric(as.array(f$timestep)),
                                          dim = 320L))
expect_true(max(abs(t_sin_host - as.array(f$t_sin))) < 1e-4)

time_ids_host <- as.array(yq_sdxl_time_ids_embed(as.array(f$time_ids),
                                                 dim = 256L))
expect_true(max(abs(time_ids_host - as.array(f$time_ids_sin))) < 1e-4)
