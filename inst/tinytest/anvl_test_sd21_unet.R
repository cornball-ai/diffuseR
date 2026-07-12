# Parity: yq_sd_unet (jitted, full SD 2.1 UNet) vs the diffuseR/torch
# reference from tools/gen_fixture_sd21_unet.R, on real F16 weights.
# Small (1x4x32x32 latent, 16 text tokens), so it runs quickly on CPU.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
ckpt <- file.path(Sys.getenv("HOME"),
                  ".local/share/R/diffuseR/sd21-diffusers/unet",
                  "diffusion_pytorch_model.safetensors")
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/sd21_unet.safetensors")
if (!file.exists(ckpt) || !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing; run tools/gen_fixture_sd21_unet.R")
}

f <- anvl::nv_read(fixture)
w <- yq_sd_unet_load_weights(ckpt, device = "cpu")

net <- yq_sd_unet()
fj <- anvl::jit(net)
out <- fj(f$sample, f$t_sin, f$context, w)

got <- as.array(out); want <- as.array(f$out)
max_abs <- max(abs(got - want))
scale <- max(abs(want))
cat(sprintf("sd21 unet parity: max %.3e mean %.3e cor %.6f (out scale %.3f)\n",
            max_abs, mean(abs(got - want)),
            cor(as.vector(got), as.vector(want)), scale))
expect_equal(dim(got), c(1L, 4L, 32L, 32L))
expect_true(cor(as.vector(got), as.vector(want)) > 0.99999)
# f32 tolerance relative to output scale (accumulated over a deep conv net)
expect_true(max_abs < 1e-3 * scale)
expect_true(mean(abs(got - want)) < 1e-4 * scale)

# Host-side sinusoid matches the reference's internal timestep embedding
# (f32 tolerance: cos/sin of a ~500 argument round differently in f32 vs
# f64-then-cast, but the values are otherwise identical, cor == 1).
t_sin_host <- as.array(yq_sd_time_embed(as.numeric(as.array(f$timestep)),
                                        dim = 320L))
expect_true(max(abs(t_sin_host - as.array(f$t_sin))) < 1e-4)
