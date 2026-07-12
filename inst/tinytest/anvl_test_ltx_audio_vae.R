# Parity: yq_ltx_audio_vae_decode (jitted LTX-2.3 audio VAE decoder, causal
# 2D conv via nv_pad + nv_conv2d) + yq_ltx_audio_vae_prepare (packed-latent
# de-normalization + unpack) vs the torch reference from
# tools/gen_fixture_ltx_audio_vae.R. RANDOM-INIT weights, SMALL config
# (base 8, latent 4, ch_mult c(1,2,4), mel_bins 16 -> out [1, 2, 17, 16]).
# Architecture parity; runs in seconds on CPU f32.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
weights <- file.path(fixture_dir, "ltx_audio_vae_weights.safetensors")
fixture <- file.path(fixture_dir, "ltx_audio_vae.safetensors")
if (!file.exists(weights) || !file.exists(fixture)) {
    exit_file("weights or fixture missing")
}

f <- anvl::nv_read(fixture)
num_mel_bins <- as.integer(as.array(f$num_mel_bins))
w <- yq_ltx_audio_vae_load_weights(weights, device = "cpu", mel_bins = 16L)

# De-normalize + unpack the packed latent (host stats) then the jitted decode.
zin <- yq_ltx_audio_vae_prepare(f$z_packed,
                                as.double(as.array(f$latents_mean)),
                                as.double(as.array(f$latents_std)),
                                num_mel_bins)
fj <- anvl::jit(function(z) yq_ltx_audio_vae_decode(z, w))
out <- fj(zin)

got <- as.array(out)
want <- as.array(f$out)
max_abs <- max(abs(got - want))
scale <- max(abs(want))
correlation <- cor(as.vector(got), as.vector(want))
cat(sprintf("ltx audio vae decode parity: max %.2e (rel %.2e) mean %.2e cor %.6f\n",
            max_abs, max_abs / scale, mean(abs(got - want)), correlation))

expect_equal(dim(got), c(1L, 2L, 17L, 16L))
expect_true(correlation > 0.999999)
expect_true(max_abs / scale < 1e-4) # scale-relative max-abs

# Standalone check of the de-normalize + unpack (z * std + mean, then
# [B, T, C*M] -> [B, C, T, M]).
prep <- as.array(zin)
zp <- as.array(f$z_packed)
mn <- as.array(f$latents_mean)
sdv <- as.array(f$latents_std)
denorm <- sweep(sweep(zp, 3L, sdv, `*`), 3L, mn, `+`) # [B, T, C*M]
d <- dim(denorm)
cm <- d[3L]
cc <- cm %/% num_mel_bins
# unpack row-major: [B, T, C*M] -> [B, T, C, M] -> [B, C, T, M]
flat <- aperm(denorm, c(3L, 2L, 1L)) # reverse for row-major reshape
un <- array(as.vector(flat), dim = c(num_mel_bins, cc, d[2L], d[1L]))
un <- aperm(un, c(4L, 3L, 2L, 1L)) # [B, T, C, M]
want_prep <- aperm(un, c(1L, 3L, 2L, 4L)) # [B, C, T, M]
expect_true(max(abs(prep - want_prep)) < 1e-5)
