# Parity: yq_ltx_vae_decode (jitted LTX-2.3 3D video VAE decoder, causal
# 3D conv via prim_convolution) + yq_ltx_vae_prepare (per-channel latent
# de-normalization) vs the torch reference from tools/gen_fixture_ltx_vae.R.
# RANDOM-INIT weights, SMALL config (latent 8ch, F=2, 4x4 -> 3ch, 9 frames,
# 128x128). Architecture parity; runs in seconds on CPU f32.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
weights <- file.path(fixture_dir, "ltx_vae_weights.safetensors")
fixture <- file.path(fixture_dir, "ltx_vae.safetensors")
if (!file.exists(weights) || !file.exists(fixture)) {
    exit_file("weights or fixture missing")
}

f <- anvl::nv_read(fixture)
w <- yq_ltx_vae_load_weights(weights, device = "cpu")

# Latent de-normalization (host stats) then the jitted decoder.
zin <- yq_ltx_vae_prepare(f$z_raw, as.double(as.array(f$latents_mean)),
                          as.double(as.array(f$latents_std)))
fj <- anvl::jit(function(z) yq_ltx_vae_decode(z, w))
out <- fj(zin)

got <- as.array(out)
want <- as.array(f$out)
max_abs <- max(abs(got - want))
scale <- max(abs(want))
correlation <- cor(as.vector(got), as.vector(want))
cat(sprintf("ltx vae decode parity: max %.2e (rel %.2e) mean %.2e cor %.6f\n",
            max_abs, max_abs / scale, mean(abs(got - want)), correlation))

expect_equal(dim(got), c(1L, 3L, 9L, 128L, 128L))
expect_true(correlation > 0.999999)
expect_true(max_abs / scale < 1e-4)   # scale-relative max-abs

# Standalone check of the per-channel de-normalization (z * std + mean).
prep <- as.array(zin)
zr <- as.array(f$z_raw)
mn <- as.array(f$latents_mean)
sdv <- as.array(f$latents_std)
want_prep <- sweep(sweep(zr, 2L, sdv, `*`), 2L, mn, `+`)
expect_true(max(abs(prep - want_prep)) < 1e-5)
