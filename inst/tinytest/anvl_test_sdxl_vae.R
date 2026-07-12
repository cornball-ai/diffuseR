# Parity: yq_sdxl_vae_decode (jitted, full 4-channel AutoencoderKL decoder
# + post_quant_conv) vs the torch reference from tools/gen_fixture_sdxl_vae.R,
# on REAL SDXL VAE weights. Small (8x8x4 -> 64x64x3), so it runs quickly.
# The anvl decoder reads the diffusers vae keys directly (post_quant_conv.*,
# decoder.*), so no native-format weight fixture is needed.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
vae <- Sys.glob(file.path(
    Sys.getenv("HOME"),
    ".cache/huggingface/hub/datasets--cornball-ai--sdxl-R",
    "snapshots/*/diffusers/vae/diffusion_pytorch_model.safetensors"))[1]
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/sdxl_vae.safetensors")
if (is.na(vae) || !file.exists(vae) || !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing; run tools/gen_fixture_sdxl_vae.R")
}

f <- anvl::nv_read(fixture)
w <- yq_sdxl_vae_load_weights(vae, device = "cpu")
fj <- anvl::jit(function(z) yq_sdxl_vae_decode(z, w))
out <- fj(f$z)

got <- as.array(out); want <- as.array(f$out)
max_abs <- max(abs(got - want)); scale <- max(abs(want))
cat(sprintf("sdxl vae decode parity: max %.2e mean %.2e cor %.6f (scale %.3f)\n",
            max_abs, mean(abs(got - want)),
            cor(as.vector(got), as.vector(want)), scale))
expect_equal(dim(got), c(1L, 3L, 64L, 64L))
expect_true(cor(as.vector(got), as.vector(want)) > 0.99999)
expect_true(max_abs < 1e-3 * scale)
expect_true(mean(abs(got - want)) < 1e-4 * scale)
