# Parity: yq_flux2_vae_decode (jitted, full AutoencoderKLFlux2 decoder)
# vs the torch reference from tools/gen_fixture_vae.R, on real VAE
# weights. Small (8x8x32 -> 64x64x3), so it runs quickly.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
vae <- Sys.glob(file.path(
    Sys.getenv("HOME"),
    ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
    "snapshots/*/vae/diffusion_pytorch_model.safetensors"))[1]
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/yunque/fixtures/flux2_vae.safetensors")
if (is.na(vae) || !file.exists(vae) || !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing")
}

f <- anvl::nv_read(fixture)
w <- yq_flux2_load_vae(vae, device = "cpu")
fj <- anvl::jit(function(z) yq_flux2_vae_decode(z, w))
out <- fj(f$z)

got <- as.array(out); want <- as.array(f$out)
max_abs <- max(abs(got - want))
cat(sprintf("vae decode parity: max %.2e mean %.2e cor %.6f\n",
            max_abs, mean(abs(got - want)),
            cor(as.vector(got), as.vector(want))))
expect_equal(dim(got), c(1L, 3L, 64L, 64L))
expect_true(max_abs < 1e-3)
expect_true(mean(abs(got - want)) < 1e-4)

# group norm + nearest upsample unit checks
x <- anvl::nv_array(array(rnorm(2 * 8 * 4 * 4), c(2, 8, 4, 4)), dtype = "f32")
up <- as.array(yunque::upsample_nearest2d(x))
expect_equal(dim(up), c(2L, 8L, 8L, 8L))
xa <- as.array(x)
expect_equal(up[1, 1, 1:2, 1:2], matrix(xa[1, 1, 1, 1], 2, 2))
