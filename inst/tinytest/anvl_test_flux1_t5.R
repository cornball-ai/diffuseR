# Parity: yq_t5_encoder (jitted, small random-init T5-v1.1) vs the
# diffuseR/torch t5_encoder reference from tools/gen_fixture_flux1_t5.R.
# Self-contained: the fixture carries the random weights, so no external
# checkpoint is needed. at_home + anvl/yunque guards.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture <- file.path(Sys.getenv("HOME"),
                     ".local/share/R/diffuseR/fixtures/flux1_t5_small.safetensors")
if (!file.exists(fixture)) {
    exit_file("fixture missing; run tools/gen_fixture_flux1_t5.R")
}

# Must match the fixture generator's small config.
NUM_LAYERS <- 3L; NUM_HEADS <- 4L; D_KV <- 16L; D_MODEL <- 64L
NUM_BUCKETS <- 32L; MAX_DIST <- 128L; EPS <- 1e-6

f <- anvl::nv_read(fixture)
ids0 <- matrix(as.integer(round(as.array(f$input_ids))), nrow = 1L)
S <- ncol(ids0)

w <- yq_t5_load_weights(fixture, num_layers = NUM_LAYERS, device = "cpu")
embeds <- yq_t5_embed(w$embed, ids0, device = "cpu")
pbias <- yq_t5_rel_pos_bias(w$rel_bias, S, num_buckets = NUM_BUCKETS,
                            max_distance = MAX_DIST, device = "cpu")

ej <- anvl::jit(yq_t5_encoder(num_layers = NUM_LAYERS, num_heads = NUM_HEADS,
                              d_kv = D_KV, eps = EPS, precision = "highest"))
out <- ej(embeds, pbias, w)
got <- as.array(out); want <- as.array(f$output)

max_abs <- max(abs(got - want))
scale <- sd(as.vector(want))
cat(sprintf("t5 encoder parity: max %.2e mean %.2e (rel %.1e) cor %.6f\n",
            max_abs, mean(abs(got - want)), max_abs / scale,
            cor(as.vector(got), as.vector(want))))

expect_equal(dim(got), c(1L, S, D_MODEL))
expect_true(cor(as.vector(got), as.vector(want)) > 0.99999)
expect_true(max_abs / scale < 1e-4)         # f32 rounding at the state scale
