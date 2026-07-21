# gemma3_quantize_nf4 + load_gemma3_nf4: round-trip through a tiny
# on-disk HF-layout checkpoint, projection swap to NF4 modules, forward
# parity within NF4 quantization tolerance, dispatch through
# load_gemma3_text_encoder, and the unfilled-parameter hard error.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

config <- list(
  vocab_size = 64L, hidden_size = 16L, intermediate_size = 32L,
  num_hidden_layers = 2L, num_attention_heads = 2L,
  num_key_value_heads = 1L, head_dim = 8L, query_pre_attn_scalar = 8L,
  sliding_window = 4L, sliding_window_pattern = 2L
)

ref_config <- gemma3_config_ltx2()
for (nm in names(config)) ref_config[[nm]] <- config[[nm]]
torch::torch_manual_seed(23)
ref <- gemma3_text_model(ref_config)
ref$eval()

# HF-layout snapshot: config.json, one shard, and the weight-map index
# the quantizer's sharded-dir opener requires
src <- file.path(tempdir(), "gemma3-nf4-src")
dir.create(src, showWarnings = FALSE)
jsonlite::write_json(config, file.path(src, "config.json"), auto_unbox = TRUE)
params <- ref$parameters
weights <- stats::setNames(
  lapply(params, function(p) p$detach()$contiguous()),
  paste0("model.", names(params))
)
shard_name <- "model-00001-of-00001.safetensors"
safetensors::safe_save_file(weights, file.path(src, shard_name))
jsonlite::write_json(
  list(weight_map = stats::setNames(
    as.list(rep(shard_name, length(weights))), names(weights))),
  file.path(src, "model.safetensors.index.json"), auto_unbox = TRUE
)

# --- quantize -----------------------------------------------------------------------

art <- file.path(tempdir(), "gemma3-nf4-art")
unlink(art, recursive = TRUE)
# Pin residents to float32 so the exactness assertions below are
# hermetic across CRAN (no bf16 write) and fork safetensors
options(diffuseR.st_caps = list(bfloat16 = FALSE))
manifest <- gemma3_quantize_nf4(src, art, verbose = FALSE)
options(diffuseR.st_caps = NULL)
expect_equal(manifest$model, "gemma3")
# 2 layers x 7 projections
expect_equal(manifest$n_cast, 14L)
expect_true(file.exists(file.path(art, "manifest.json")))

# --- load + forward parity ----------------------------------------------------------

enc <- load_gemma3_nf4(art, device = "cpu", dtype = "float32",
                       verbose = FALSE)
expect_true(inherits(enc$layers[[1]]$self_attn$q_proj, "ltx23_nf4_linear"))
expect_true(inherits(enc$layers[[2]]$mlp$down_proj, "ltx23_nf4_linear"))
# Residents are exact
expect_true(as.numeric((enc$parameters[["embed_tokens.weight"]] -
                        params[["embed_tokens.weight"]])$abs()$max()) == 0)

ids <- torch::torch_randint(1L, 64L, size = c(1L, 5L), dtype = torch::torch_long())
mask <- torch::torch_ones(1L, 5L)
torch::with_no_grad({
  out_ref <- ref(ids, attention_mask = mask, output_hidden_states = TRUE)
  out_nf4 <- enc(ids, attention_mask = mask, output_hidden_states = TRUE)
})
h_ref <- out_ref$last_hidden_state
h_nf4 <- out_nf4$last_hidden_state
rel <- as.numeric((h_nf4 - h_ref)$abs()$max()) /
  as.numeric(h_ref$abs()$max())
# 4-bit weights: outputs agree to quantization noise, not float noise.
# On this tiny 2-layer model the worst element sits ~0.15 relative
# (measured 0.1505); the direction check below is the real quality
# gate, and full-model embedding parity is validated on GPU.
expect_true(rel < 0.25, info = sprintf("relative max diff %.4f", rel))
cs <- as.numeric(torch::nnf_cosine_similarity(
  h_ref$flatten()$unsqueeze(1L), h_nf4$flatten()$unsqueeze(1L)))
expect_true(cs > 0.99, info = sprintf("cosine %.5f", cs))

# --- dispatch through load_gemma3_text_encoder --------------------------------------

enc2 <- load_gemma3_text_encoder(art, device = "cpu", dtype = "float32",
                                 verbose = FALSE)
expect_true(inherits(enc2$layers[[1]]$self_attn$q_proj, "ltx23_nf4_linear"))

# --- an artifact missing a resident is a hard error ---------------------------------

art2 <- file.path(tempdir(), "gemma3-nf4-art2")
unlink(art2, recursive = TRUE)
dir.create(art2)
m2 <- jsonlite::fromJSON(file.path(art, "manifest.json"))
for (s in m2$shards) {
  w <- safetensors::safe_load_file(file.path(art, s), framework = "torch")
  w[["norm.weight"]] <- NULL
  safetensors::safe_save_file(w, file.path(art2, s))
}
jsonlite::write_json(m2, file.path(art2, "manifest.json"), auto_unbox = TRUE)
expect_error(
  load_gemma3_nf4(art2, device = "cpu", verbose = FALSE),
  "not filled"
)

unlink(c(src, art, art2), recursive = TRUE)
