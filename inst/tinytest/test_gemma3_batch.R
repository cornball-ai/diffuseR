# gemma3_encode_batch: sub-batched encoding matches solo encodes, disk
# caching returns paths, resumes without re-encoding, and round-trips
# through torch_load.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

config <- gemma3_config_ltx2()
for (nm in names(cfg <- list(
  vocab_size = 64L, hidden_size = 16L, intermediate_size = 32L,
  num_hidden_layers = 2L, num_attention_heads = 2L,
  num_key_value_heads = 1L, head_dim = 8L, query_pre_attn_scalar = 8L,
  sliding_window = 4L, sliding_window_pattern = 2L
))) config[[nm]] <- cfg[[nm]]
torch::torch_manual_seed(7)
model <- gemma3_text_model(config)
model$eval()

# A stub tokenizer: encode_with_gemma3 only needs tokenize_gemma3 to
# work, which needs ids within vocab; use the real path via a fake
# minimal bpe? Simpler: call encode_with_gemma3 through its tokens by
# monkeying is overkill - instead drive the batch helper with a real
# tokenizer-free path: fake tokenizer via the exported tokenize seam is
# not available, so tokenize through gemma3_tokenizer would need vocab
# files. Drive the model directly instead for parity, and the batch
# helper end-to-end with a trivial tokenizer object.

# Minimal tokenizer standing in for gemma3_tokenizer: the helper only
# passes it through to encode_with_gemma3 -> tokenize_gemma3, which
# calls encode_bpe; emulate the contract with a closure-based double.
fake_tok <- structure(list(), class = "fake_tok")
tokenize_real <- diffuseR:::tokenize_gemma3

# Patch tokenize_gemma3 inside the namespace for this test: map each
# prompt deterministically to ids from its characters (padded left)
fake_tokenize <- function(tokenizer, prompts, max_length = 16L,
                          padding = "max_length") {
  n <- length(prompts)
  ids <- torch::torch_zeros(n, max_length, dtype = torch::torch_long())
  mask <- torch::torch_zeros(n, max_length)
  for (i in seq_len(n)) {
    v <- utf8ToInt(prompts[[i]]) %% 60L + 1L
    v <- utils::tail(v, max_length)
    ids[i, (max_length - length(v) + 1L):max_length] <-
      torch::torch_tensor(v, dtype = torch::torch_long())
    mask[i, (max_length - length(v) + 1L):max_length] <- 1
  }
  list(input_ids = ids, attention_mask = mask)
}
# No on.exit here: at test-file top level tinytest evaluates each
# expression in its own frame, so on.exit would fire immediately and
# restore the real tokenizer before the calls below. Restored at the
# end of the file instead.
assignInNamespace("tokenize_gemma3", fake_tokenize, ns = "diffuseR")

prompts <- c("a cat on a mat", "neon fog rolling", "a robot dancing")

# --- no cache: list results match solo encodes -------------------------------------

batch <- gemma3_encode_batch(prompts, model = model, tokenizer = fake_tok,
                             batch_size = 2L, max_sequence_length = 16L,
                             device = "cpu", verbose = FALSE)
expect_equal(length(batch), 3L)
solo <- encode_with_gemma3(prompts[2], model = model, tokenizer = fake_tok,
                           max_sequence_length = 16L, device = "cpu",
                           verbose = FALSE)
expect_equal(as.integer(batch[[2]]$prompt_embeds$shape),
             as.integer(solo$prompt_embeds$shape))
expect_true(as.numeric((batch[[2]]$prompt_embeds -
                        solo$prompt_embeds)$abs()$max()) < 1e-5)
expect_true(as.numeric((batch[[2]]$prompt_attention_mask -
                        solo$prompt_attention_mask)$abs()$max()) == 0)

# --- cache: paths returned, resume skips, round-trip matches -----------------------

cache <- file.path(tempdir(), "gemma3-batch-cache")
unlink(cache, recursive = TRUE)
paths <- gemma3_encode_batch(prompts, model = model, tokenizer = fake_tok,
                             batch_size = 2L, cache_dir = cache,
                             max_sequence_length = 16L, device = "cpu",
                             verbose = FALSE)
expect_equal(length(paths), 3L)
expect_true(all(file.exists(paths)))

loaded <- torch::torch_load(paths[2])
expect_true(as.numeric((loaded$prompt_embeds -
                        batch[[2]]$prompt_embeds)$abs()$max()) < 1e-6)

# Resume: files untouched on the second call (no re-encode)
mt <- file.mtime(paths)
paths2 <- gemma3_encode_batch(prompts, model = model, tokenizer = fake_tok,
                              batch_size = 2L, cache_dir = cache,
                              max_sequence_length = 16L, device = "cpu",
                              verbose = FALSE)
expect_equal(paths2, paths)
expect_equal(file.mtime(paths), mt)

# A new prompt encodes into the same cache without touching the others
paths3 <- gemma3_encode_batch(c(prompts, "one more scene"), model = model,
                              tokenizer = fake_tok, batch_size = 2L,
                              cache_dir = cache,
                              max_sequence_length = 16L, device = "cpu",
                              verbose = FALSE)
expect_equal(length(paths3), 4L)
expect_equal(paths3[1:3], paths)
expect_true(file.exists(paths3[4]))

unlink(cache, recursive = TRUE)
assignInNamespace("tokenize_gemma3", tokenize_real, ns = "diffuseR")
