# Parity: yq_zimage_dit (jitted anvl Z-Image DiT) vs the torch reference
# fixture from tools/gen_fixture_zimage_dit.R (random-init weights). CPU
# f32. Also cross-checks the host-side time-embed and RoPE tables.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/zimage_dit.safetensors")
if (!file.exists(fixture)) {
    exit_file("fixture missing (run tools/gen_fixture_zimage_dit.R)")
}

# fixture config (must match tools/gen_fixture_zimage_dit.R)
heads <- 6L
axes_dims <- c(16L, 20L, 20L)
theta <- 256
cap_len <- 5L
h_tokens <- 4L; w_tokens <- 6L; f_tokens <- 1L

f <- anvl::nv_read(fixture)
w <- yq_zimage_load_weights(fixture, device = "cpu")

dit <- anvl::jit(yq_zimage_dit(heads = heads, precision = "highest"))
out <- dit(f$tokens, f$cap_feats, f$t_freq,
           f$cos_img, f$sin_img, f$cos_cap, f$sin_cap, w)

# Compare the image span of the packed final-layer output against the
# reference (out_img = patchify of the [C,F,H,W] forward output).
got_full <- as.array(out)
want <- as.array(f$out_img)
img_len <- dim(want)[1L]
got <- got_full[1L, seq_len(img_len), , drop = TRUE]

max_abs <- max(abs(got - want))
sd_out <- sd(as.vector(want))
correlation <- cor(as.vector(got), as.vector(want))
cat(sprintf("zimage DiT parity: max %.2e  mean %.2e  cor %.6f  (out sd %.3f)\n",
            max_abs, mean(abs(got - want)), correlation, sd_out))

expect_equal(dim(got), dim(want))
expect_true(correlation > 0.999999)
# f32 tolerance relative to the output scale
expect_true(max_abs < 1e-4 * max(1, sd_out))
expect_true(mean(abs(got - want)) < 1e-5 * max(1, sd_out))

# ---- host-side time-embed cross-check ----
tp <- as.array(yq_zimage_time_embed(0.7, freq_size = 256L, t_scale = 1000))
tp_ref <- as.array(f$t_freq)
cat(sprintf("time-embed parity: max %.2e\n", max(abs(tp - tp_ref))))
# f32 rounding of the large angle (t * 1000) lands ~2.5e-5
expect_true(max(abs(tp - tp_ref)) < 1e-4)

# ---- host-side RoPE cross-check ----
rope <- yq_zimage_rope(h_tokens, w_tokens, cap_len, f_tokens = f_tokens,
                       axes_dim = axes_dims, theta = theta)
rope_max <- max(
    max(abs(as.array(rope$cos_img) - as.array(f$cos_img))),
    max(abs(as.array(rope$sin_img) - as.array(f$sin_img))),
    max(abs(as.array(rope$cos_cap) - as.array(f$cos_cap))),
    max(abs(as.array(rope$sin_cap) - as.array(f$sin_cap)))
)
cat(sprintf("rope parity: max %.2e\n", rope_max))
expect_true(rope_max < 1e-5)
