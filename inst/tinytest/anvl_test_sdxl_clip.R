# Parity: yq_sdxl_clip_encoders (jitted CLIP-L + OpenCLIP bigG) vs the
# diffuseR/torch reference from tools/gen_fixture_sdxl_clip.R, on
# RANDOM-INIT weights (architecture parity). Small (16 tokens) but the
# full production configs (CLIP-L 768/12/12 quick-GELU; bigG 1280/32/20
# exact GELU). Asserts correlation ~1.0 + scale-relative max-abs on BOTH
# the concatenated penultimate context [1, S, 2048] and the pooled bigG
# vector [1, 1280]. CLIP hidden states carry large-magnitude outlier dims,
# so parity is judged relative to the output scale, not a fixed threshold.

if (!tinytest::at_home()) {
    exit_file("at_home only")
}
if (!requireNamespace("anvl", quietly = TRUE) ||
    !requireNamespace("yunque", quietly = TRUE)) {
    exit_file("anvl not installed")
}
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
clipl_file <- file.path(fixture_dir, "sdxl_clip_l_weights.safetensors")
bigg_file <- file.path(fixture_dir, "sdxl_bigg_weights.safetensors")
io_file <- file.path(fixture_dir, "sdxl_clip_io.safetensors")
if (!file.exists(clipl_file) || !file.exists(bigg_file) ||
    !file.exists(io_file)) {
    exit_file("fixture missing; run tools/gen_fixture_sdxl_clip.R")
}

f <- anvl::nv_read(io_file)
ids0 <- matrix(as.integer(round(as.array(f$ids0))), nrow = 1L)
S <- ncol(ids0)
eos_index <- which.max(ids0[1, ])                # EOS = highest id, unambiguous

w_clipl <- yq_sdxl_clip_load_weights(clipl_file, num_layers = 12L,
                                     has_text_projection = FALSE, device = "cpu")
w_bigg <- yq_sdxl_clip_load_weights(bigg_file, num_layers = 32L,
                                    has_text_projection = TRUE, device = "cpu")

clipl_embeds <- yq_sdxl_clip_embed(w_clipl$token_embedding,
                                   w_clipl$position_embedding, ids0,
                                   device = "cpu")
bigg_embeds <- yq_sdxl_clip_embed(w_bigg$token_embedding,
                                  w_bigg$position_embedding, ids0,
                                  device = "cpu")
mask <- yq_sdxl_clip_mask(S, batch = 1L, device = "cpu")

res <- yq_sdxl_clip_encoders(clipl_embeds, bigg_embeds, mask, eos_index,
                             w_clipl, w_bigg, precision = "highest")

# ---- concatenated penultimate context parity ----
gc_ctx <- as.array(res$context)
wc_ctx <- as.array(f$context)
ctx_max <- max(abs(gc_ctx - wc_ctx))
ctx_scale <- max(abs(wc_ctx))
ctx_cor <- cor(as.vector(gc_ctx), as.vector(wc_ctx))
cat(sprintf("sdxl clip context: max %.3e mean %.3e cor %.6f (scale %.3f)\n",
            ctx_max, mean(abs(gc_ctx - wc_ctx)), ctx_cor, ctx_scale))
expect_equal(dim(gc_ctx), c(1L, S, 2048L))
expect_true(ctx_cor > 0.99999)
expect_true(ctx_max < 1e-3 * ctx_scale)
expect_true(mean(abs(gc_ctx - wc_ctx)) < 1e-4 * ctx_scale)

# ---- pooled bigG vector parity ----
gp <- as.array(res$pooled)
wp <- as.array(f$pooled)
p_max <- max(abs(gp - wp))
p_scale <- max(abs(wp))
p_cor <- cor(as.vector(gp), as.vector(wp))
cat(sprintf("sdxl clip pooled : max %.3e mean %.3e cor %.6f (scale %.3f)\n",
            p_max, mean(abs(gp - wp)), p_cor, p_scale))
expect_equal(dim(gp), c(1L, 1280L))
expect_true(p_cor > 0.99999)
expect_true(p_max < 1e-3 * p_scale)
expect_true(mean(abs(gp - wp)) < 1e-4 * p_scale)
