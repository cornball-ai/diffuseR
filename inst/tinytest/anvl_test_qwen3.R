# Parity: full yq_qwen3_encoder (jitted, 27 layers, real Qwen3-4B
# weights) vs the torch reference from tools/gen_fixture_qwen3.R. Loads
# the sharded text_encoder checkpoint (~84s), so at_home + gated.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
te <- file.path(
    Sys.getenv("HOME"),
    ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
    "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294", "text_encoder")
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/yunque/fixtures/qwen3_encoder.safetensors")
if (!dir.exists(te) || !file.exists(fixture)) {
    exit_file("checkpoint or fixture missing")
}

f <- anvl::nv_read(fixture)
ids0 <- matrix(as.integer(round(as.array(f$ids0))), nrow = 1L)
attn <- matrix(as.integer(round(as.array(f$attn))), nrow = 1L)
S <- ncol(ids0)

w <- yq_qwen3_load_weights(te, n_layers = 27L, device = "cpu")
embeds <- yq_qwen3_embed(w$embed, ids0, device = "cpu")
rope <- yq_qwen3_rope(S, 128L, 1e6, device = "cpu")
mask <- yq_qwen3_mask(attn, S, batch = 1L, device = "cpu")

ej <- anvl::jit(yq_qwen3_encoder(out_layers = c(9L, 18L, 27L),
                                 precision = "highest"))
out <- ej(embeds, rope$cos, rope$sin, mask, w)
got <- as.array(out); want <- as.array(f$out)

max_abs <- max(abs(got - want))
sdw <- sd(as.vector(want))
cat(sprintf("qwen3 encoder parity: max %.2e mean %.2e (rel %.1e) cor %.6f\n",
            max_abs, mean(abs(got - want)), max_abs / sdw,
            cor(as.vector(got), as.vector(want))))
expect_equal(dim(got), c(1L, S, 7680L))
expect_true(max_abs / sdw < 1e-4)          # f32 rounding at the state scale
expect_true(mean(abs(got - want)) < 1e-4)
