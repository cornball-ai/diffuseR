# Parity: yq_clip_encoder (jitted, full SD 2.1 OpenCLIP ViT-H text
# encoder) vs the diffuseR/torch reference from
# tools/gen_fixture_sd21_clip.R, on real F16 weights. Small (16 tokens),
# so it runs quickly on CPU. CLIP hidden states carry large-magnitude
# outlier dims, so parity is judged by correlation + max-abs relative to
# the output scale, not a fixed absolute threshold.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
te <- file.path(Sys.getenv("HOME"),
                ".local/share/R/diffuseR/sd21-diffusers/text_encoder")
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/sd21_clip.safetensors")
if (!dir.exists(te) || !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing; run tools/gen_fixture_sd21_clip.R")
}

f <- anvl::nv_read(fixture)
ids0 <- matrix(as.integer(round(as.array(f$ids0))), nrow = 1L)
S <- ncol(ids0)

w <- yq_clip_load_weights(te, device = "cpu")
embeds <- yq_clip_embed(w$token_embedding, w$position_embedding, ids0,
                        device = "cpu")
mask <- yq_clip_mask(S, batch = 1L, device = "cpu")

ej <- anvl::jit(yq_clip_encoder(apply_final_ln = TRUE, precision = "highest"))
out <- ej(embeds, mask, w)

got <- as.array(out)
want <- as.array(f$out)
max_abs <- max(abs(got - want))
scale <- max(abs(want))
correlation <- cor(as.vector(got), as.vector(want))
cat(sprintf("sd21 clip parity: max %.3e mean %.3e cor %.6f (out scale %.3f)\n",
            max_abs, mean(abs(got - want)), correlation, scale))

expect_equal(dim(got), c(1L, S, 1024L))
expect_true(correlation > 0.99999)
# f32 tolerance relative to output scale (accumulated over 23 layers, with
# large-magnitude CLIP outlier dims setting the scale).
expect_true(max_abs < 1e-3 * scale)
expect_true(mean(abs(got - want)) < 1e-4 * scale)
