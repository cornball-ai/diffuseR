# Parity: full yq_gemma3_encoder (jitted, small random-init config) vs
# the torch reference diffuseR::gemma3_text_model, from
# tools/gen_fixture_ltx_gemma3.R. Random-init weights validate the
# architecture (dual RoPE, sandwich norms, (1+weight) RMSNorm, GeGLU,
# sqrt embedding scale, GQA) at f32 tolerance without any download.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/ltx_gemma3.safetensors")
if (!file.exists(fixture)) {
    exit_file("fixture missing (run tools/gen_fixture_ltx_gemma3.R)")
}

# ---- small config, must match tools/gen_fixture_ltx_gemma3.R ----
NUM_LAYERS <- 6L; NUM_HEADS <- 4L; NUM_KV <- 2L; HEAD_DIM <- 48L
EPS <- 1e-6; PATTERN <- 6L
ROPE_THETA <- 1e6; ROPE_SCALE <- 8.0; ROPE_LOCAL_THETA <- 1e4

f <- anvl::nv_read(fixture)
ids0 <- matrix(as.integer(round(as.array(f$ids0))), nrow = 1L)
attn <- matrix(as.integer(round(as.array(f$attn))), nrow = 1L)
S <- ncol(ids0)

w <- yq_gemma3_load_weights(fixture, num_layers = NUM_LAYERS, device = "cpu")
embeds <- yq_gemma3_embed(w$embed, ids0, device = "cpu")
rope_g <- yq_gemma3_rope(S, HEAD_DIM, ROPE_THETA, ROPE_SCALE, device = "cpu")
rope_l <- yq_gemma3_rope(S, HEAD_DIM, ROPE_LOCAL_THETA, 1.0, device = "cpu")
mask <- yq_gemma3_mask(attn, S, batch = 1L, device = "cpu")

ej <- anvl::jit(yq_gemma3_encoder(num_layers = NUM_LAYERS, num_heads = NUM_HEADS,
                                  num_kv = NUM_KV, head_dim = HEAD_DIM, eps = EPS,
                                  sliding_window_pattern = PATTERN,
                                  precision = "highest"))
out <- ej(embeds, rope_g, rope_l, mask, w)
got <- as.array(out); want <- as.array(f$out)

max_abs <- max(abs(got - want))
sdw <- sd(as.vector(want))
corr <- cor(as.vector(got), as.vector(want))
cat(sprintf("ltx gemma3 encoder parity: max %.2e mean %.2e (rel %.1e) cor %.6f\n",
            max_abs, mean(abs(got - want)), max_abs / sdw, corr))

expect_equal(dim(got), c(1L, S, 128L, NUM_LAYERS + 1L))
expect_true(corr > 0.999999)               # architecture parity
expect_true(max_abs / sdw < 1e-4)          # f32 rounding at the state scale
