# Parity: yq_sd_vae_decode (jitted, full SD 2.1 AutoencoderKL decoder)
# vs the diffuseR/torch reference from tools/gen_fixture_sd21_vae.R, on
# real F16 VAE weights. Small (1x4x8x8 -> 1x3x64x64), so it runs quickly
# on CPU.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
vae <- file.path(Sys.getenv("HOME"),
                 ".local/share/R/diffuseR/sd21-diffusers/vae",
                 "diffusion_pytorch_model.safetensors")
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/sd21_vae.safetensors")
if (!file.exists(vae) || !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing; run tools/gen_fixture_sd21_vae.R")
}

f <- anvl::nv_read(fixture)
w <- yq_sd_vae_load_weights(vae, device = "cpu")

fj <- anvl::jit(function(z) yq_sd_vae_decode(z, w))
out <- fj(f$z)

got <- as.array(out); want <- as.array(f$out)
max_abs <- max(abs(got - want))
scale <- max(abs(want))
cat(sprintf("sd21 vae parity: max %.3e mean %.3e cor %.6f (out scale %.3f)\n",
            max_abs, mean(abs(got - want)),
            cor(as.vector(got), as.vector(want)), scale))
expect_equal(dim(got), c(1L, 3L, 64L, 64L))
expect_true(cor(as.vector(got), as.vector(want)) > 0.99999)
# f32 tolerance relative to output scale (accumulated over a deep conv net)
expect_true(max_abs < 1e-3 * scale)
expect_true(mean(abs(got - want)) < 1e-4 * scale)

# yq_sd_vae_prepare applies the scaling-factor rescale (z / 0.18215).
zp <- as.array(yq_sd_vae_prepare(f$z))
expect_true(max(abs(zp - as.array(f$z) / 0.18215)) < 1e-4)
