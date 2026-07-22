# load_gemma3_text_encoder(): skeleton construction at the target dtype,
# checkpoint fill, and the hard error when a parameter is not covered by
# the checkpoint (skeleton weights are uninitialized memory, so a silent
# partial load must be impossible).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

# Tiny model dir: config.json + one model-*.safetensors shard whose keys
# carry the HF "model." prefix, generated from a bare (initialized) model
config <- list(
  vocab_size = 64L, hidden_size = 16L, intermediate_size = 32L,
  num_hidden_layers = 2L, num_attention_heads = 2L,
  num_key_value_heads = 1L, head_dim = 8L, query_pre_attn_scalar = 8L,
  sliding_window = 4L, sliding_window_pattern = 2L
)

ref_config <- gemma3_config_ltx2()
for (nm in names(config)) ref_config[[nm]] <- config[[nm]]
torch::torch_manual_seed(11)
ref <- gemma3_text_model(ref_config)
ref$eval()

dir <- file.path(tempdir(), "gemma3-loader-test")
dir.create(dir, showWarnings = FALSE)
jsonlite::write_json(config, file.path(dir, "config.json"), auto_unbox = TRUE)

params <- ref$parameters
weights <- stats::setNames(
  lapply(params, function(p) p$detach()$contiguous()),
  paste0("model.", names(params))
)
safetensors::safe_save_file(weights, file.path(dir, "model-00001-of-00001.safetensors"))

# --- full checkpoint loads; weights and forward match the reference ----------------

enc <- load_gemma3_text_encoder(dir, device = "cpu", dtype = "float32",
                                verbose = FALSE)
expect_true(all(names(enc$parameters) %in% names(params)))
expect_true(as.numeric((enc$parameters[["embed_tokens.weight"]] -
                        params[["embed_tokens.weight"]])$abs()$max()) == 0)

ids <- torch::torch_randint(1L, 64L, size = c(1L, 5L), dtype = torch::torch_long())
mask <- torch::torch_ones(1L, 5L)
torch::with_no_grad({
  out_ref <- ref(ids, attention_mask = mask, output_hidden_states = TRUE)
  out_enc <- enc(ids, attention_mask = mask, output_hidden_states = TRUE)
})
expect_true(as.numeric((out_enc$last_hidden_state -
                        out_ref$last_hidden_state)$abs()$max()) < 1e-6)

# --- skeleton dtype follows the dtype argument -------------------------------------

enc16 <- load_gemma3_text_encoder(dir, device = "cpu", dtype = "float16",
                                  verbose = FALSE)
expect_true(enc16$parameters[[1]]$dtype == torch::torch_float16())

# --- a checkpoint that misses a parameter is a hard error --------------------------

partial <- weights[setdiff(names(weights), "model.norm.weight")]
unlink(file.path(dir, "model-00001-of-00001.safetensors"))
safetensors::safe_save_file(partial, file.path(dir, "model-00001-of-00001.safetensors"))
expect_error(
  load_gemma3_text_encoder(dir, device = "cpu", dtype = "float32",
                           verbose = FALSE),
  "not filled"
)

unlink(dir, recursive = TRUE)

# --- pinned staging surface ----------------------------------------------------------

# Both loaders take pin (default: the pin_staging option). Actual
# page-locking needs CUDA; validated on GPU outside R CMD check.
expect_true("pin" %in% names(formals(load_gemma3_text_encoder)))
expect_true("pin" %in% names(formals(load_gemma3_nf4)))
