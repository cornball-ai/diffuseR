# Parity of the Gemma 4 (LTX-2.5) text encoder and tokenizer against
# HF transformers / tokenizers fixtures (tools/gen_fixtures_gemma4.py).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

fixture <- function(name) {
  p <- system.file("tinytest", "fixtures", name, package = "diffuseR")
  if (p == "") p <- file.path("fixtures", name)
  p
}
if (!file.exists(fixture("gemma4_text.safetensors"))) exit_file("gemma4 fixtures missing")
max_abs_diff <- function(a, b) as.numeric((a - b)$abs()$max())

# --- Tokenizer -------------------------------------------------------------------

tok <- gemma4_tokenizer(fixture("gemma4_tiny_tokenizer.json"))
expect_inherits(tok, "gemma4_tokenizer")
# Gemma 4's post-processor adds no special tokens (no BOS)
expect_equal(length(tok$template$prefix) + length(tok$template$suffix), 0L)
cases <- jsonlite::fromJSON(fixture("gemma4_tokenizer_cases.json"),
                            simplifyVector = FALSE)
for (cs in cases) {
  got <- encode_gemma4(tok, cs$text, max_length = NULL)$input_ids[[1]]
  expect_identical(as.integer(got), as.integer(unlist(cs$ids)), info = cs$text)
}
# Left padding with <pad> and truncation keeping the start
enc <- encode_gemma4(tok, c("the cat sat", "tell"), max_length = 4L)
expect_identical(enc$input_ids[2, ], c(0L, 0L, 274L, 282L))
expect_identical(enc$attention_mask[2, ], c(0L, 0L, 1L, 1L))
expect_identical(enc$input_ids[1, ], c(274L, 279L, 287L, 288L))

# The real LTX-2.5 tokenizer, when a copy is available locally
real_tok <- Sys.getenv("GEMMA4_TOKENIZER_JSON")
if (at_home() && nzchar(real_tok) && file.exists(real_tok)) {
  rtok <- gemma4_tokenizer(real_tok)
  rcases <- jsonlite::fromJSON(fixture("gemma4_real_tokenizer_cases.json"),
                               simplifyVector = FALSE)
  for (cs in rcases) {
    got <- encode_gemma4(rtok, cs$text, max_length = NULL)$input_ids[[1]]
    expect_identical(as.integer(got), as.integer(unlist(cs$ids)), info = cs$text)
  }
}

# --- Model through the single-file loader ---------------------------------------

model <- load_gemma4_text_encoder(fixture("gemma4_tiny_ckpt.safetensors"),
                                  dtype = "float32", pin = FALSE, verbose = FALSE)
expect_inherits(model, "Gemma4TextModel")
blk_full <- model$layers[[3]]$self_attn
expect_true(blk_full$k_eq_v)
expect_null(blk_full$v_proj)
expect_equal(blk_full$head_dim, 16L)
expect_equal(blk_full$num_kv_heads, 1L)
expect_false(model$layers[[1]]$self_attn$k_eq_v)

fx <- safetensors::safe_load_file(fixture("gemma4_text.safetensors"),
                                  framework = "torch")
torch::with_no_grad({
  out <- model(fx$input_ids, attention_mask = fx$attention_mask,
               output_hidden_states = TRUE)
})
stacked <- torch::torch_stack(out$hidden_states, dim = -1L)
expect_equal(as.integer(stacked$shape), as.integer(fx$hidden_states$shape))
# Real tokens are the contract (the connectors replace pad positions)
valid <- fx$attention_mask$to(dtype = torch::torch_bool())
expect_true(max_abs_diff(stacked[valid], fx$hidden_states[valid]) < 1e-4)
expect_true(max_abs_diff(out$last_hidden_state[valid],
                         fx$last_hidden_state[valid]) < 1e-4)
expect_true(as.logical(stacked$isfinite()$all()$item()))

# The full pipeline step: left-padded encode returns the same stack layout
enc <- encode_with_gemma4(c("the cat sat", "tell"), model, tok,
                          max_sequence_length = 6L, device = "cpu",
                          verbose = FALSE)
expect_equal(as.integer(enc$prompt_embeds$shape), c(2L, 6L, 32L, 7L))
expect_identical(as.integer(as.array(enc$prompt_attention_mask[2, ])),
                 c(0L, 0L, 0L, 0L, 1L, 1L))
