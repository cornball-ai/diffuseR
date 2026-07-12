# Parity: yq_zimage_vae_decode (jitted, full AutoencoderKL 16-channel
# decoder) + yq_zimage_vae_prepare (scalar shift+scale de-normalization)
# vs the torch reference from tools/gen_fixture_zimage_vae.R. RANDOM-INIT
# weights (architecture parity). Small (8x8x16 -> 64x64x3), runs quickly.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
weights <- file.path(fixture_dir, "zimage_vae_weights.safetensors")
fixture <- file.path(fixture_dir, "zimage_vae.safetensors")
if (!file.exists(weights) || !file.exists(fixture)) {
    exit_file("weights or fixture missing")
}

f <- anvl::nv_read(fixture)
w <- yq_zimage_vae_load_weights(weights, device = "cpu")

# Full decode path: prepare (eager scalar de-norm) then jitted decoder.
zin <- yq_zimage_vae_prepare(f$z_raw)
fj <- anvl::jit(function(z) yq_zimage_vae_decode(z, w))
out <- fj(zin)

got <- as.array(out)
want <- as.array(f$out)
max_abs <- max(abs(got - want))
scale <- max(abs(want))
correlation <- cor(as.vector(got), as.vector(want))
cat(sprintf("zimage vae decode parity: max %.2e (rel %.2e) mean %.2e cor %.6f\n",
            max_abs, max_abs / scale, mean(abs(got - want)), correlation))

expect_equal(dim(got), c(1L, 3L, 64L, 64L))
expect_true(correlation > 0.999999)
expect_true(max_abs / scale < 1e-4)  # scale-relative max-abs

# Standalone check of the FLUX.1 scaling/shift de-normalization constants.
prep <- as.array(zin)
want_prep <- as.array(f$z_raw) / 0.3611 + 0.1159
expect_true(max(abs(prep - want_prep)) < 1e-5)
