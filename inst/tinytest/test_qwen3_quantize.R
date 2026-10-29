# Qwen3 text-encoder quantization (fp8 / NF4 artifacts) on the tiny
# random-init Qwen3-VL checkpoint: key mapping, the cast set, loading,
# outputs near the full-precision model, and layer streaming with fp8
# fields.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

ckpt_dir <- system.file("tinytest", "fixtures", "flux3_qwen_tiny_ckpt",
  package = "diffuseR")
if (ckpt_dir == "") ckpt_dir <- "fixtures/flux3_qwen_tiny_ckpt"
if (!dir.exists(ckpt_dir)) {
  exit_file("flux3 qwen fixtures missing")
}

# --- Key mapping and the cast set -------------------------------------------------

map <- diffuseR:::.qwen3_map_key
expect_equal(map("model.language_model.layers.0.mlp.up_proj.weight", TRUE),
  "model.layers.0.mlp.up_proj.weight")
expect_true(is.na(map("model.visual.blocks.0.attn.qkv.weight", TRUE)))
expect_true(is.na(map("lm_head.weight", FALSE)))
expect_equal(map("model.norm.weight", FALSE), "model.norm.weight")

is_q <- diffuseR:::.qwen3_is_quant_key
expect_true(all(is_q(c("model.layers.3.self_attn.q_proj.weight",
  "model.layers.3.self_attn.o_proj.weight",
  "model.layers.12.mlp.down_proj.weight"))))
expect_false(any(is_q(c("model.layers.3.self_attn.q_norm.weight",
  "model.layers.3.input_layernorm.weight", "model.embed_tokens.weight",
  "model.norm.weight"))))

# --- Reference ----------------------------------------------------------------------

ref_model <- load_qwen3_text_encoder(ckpt_dir, device = "cpu",
  dtype = "float32", verbose = FALSE)
n_layers <- length(ref_model$model$layers)
ids <- torch::torch_tensor(matrix(c(5L, 9L, 13L, 2L, 7L, 11L), nrow = 1),
  dtype = torch::torch_long())
ref <- torch::with_no_grad(ref_model(ids, out_layers = c(1L, 2L, n_layers)))
rel_err <- function(a, b) {
  (a - b)$norm()$item() / b$norm()$item()
}

formats <- if (diffuseR:::.st_can_write("float8_e4m3fn")) c("fp8", "nf4") else "nf4"
# Per-tensor fp8 keeps ~3% error on the real 4B; NF4 ~5%
tol <- c(fp8 = 0.1, nf4 = 0.25)
for (f in formats) {
  dir <- file.path(tempdir(), paste0("qwen3-tiny-", f))
  unlink(dir, recursive = TRUE)
  manifest <- qwen3_quantize(ckpt_dir, dir, format = f, verbose = FALSE)
  # 7 projections per layer
  expect_equal(manifest$cast, 7L * n_layers)
  expect_true(diffuseR:::.is_qwen3_artifact(dir, f))
  expect_false(diffuseR:::.is_qwen3_artifact(dir, setdiff(c("fp8", "nf4"), f)))
  keys <- diffuseR:::ltx23_open_fp8_checkpoint(dir)$keys
  expect_false(any(grepl("visual|language_model|lm_head", keys)))
  # A valid artifact is not rebuilt
  again <- qwen3_quantize(ckpt_dir, dir, format = f, verbose = FALSE)
  expect_equal(again$shards, manifest$shards)

  model <- load_qwen3_text_encoder(dir, device = "cpu", dtype = "float32",
    verbose = FALSE)
  q_proj <- model$model$layers[[1]]$self_attn$q_proj
  expect_true(inherits(q_proj, if (f == "fp8") "ltx23_fp8_linear" else "ltx23_nf4_linear"))
  expect_equal(model$rope_theta, ref_model$rope_theta)
  out <- torch::with_no_grad(model(ids, out_layers = c(1L, 2L, n_layers)))
  for (k in seq_along(ref)) {
    err <- rel_err(out[[k]], ref[[k]])
    expect_true(err > 0 && err < tol[[f]], info = sprintf("%s state %d: %.4f", f, k, err))
  }

  # Layer streaming stages the fp8 fields with each layer
  base <- diffuseR:::.flux3_text_stream(model, pin = FALSE)
  per_layer <- length(model$model$layers[[1]]$parameters) +
    length(model$model$layers[[1]]$buffers)
  expected <- if (f == "fp8") per_layer + 14L else per_layer
  expect_equal(length(model$layer_stream[[1]]), expected)
  diffuseR:::.staged_onload(base, "cpu")
  streamed <- torch::with_no_grad(model(ids, out_layers = c(1L, 2L, n_layers)))
  for (k in seq_along(out)) {
    expect_true((streamed[[k]] - out[[k]])$abs()$max()$item() == 0)
  }
  model$layer_stream <- NULL
}

# --- Pipeline resolution -------------------------------------------------------------

resolve <- diffuseR:::.flux3_resolve_text_precision
expect_equal(resolve("bf16"), "bf16")
expect_equal(resolve("nf4"), "nf4")
expect_true(resolve("auto") %in% c("fp8", "nf4", "bf16"))
