# gemma4_quantize_nf4 + load_gemma4_nf4: round trip through the tiny
# LTX-2.5-layout checkpoint (tools/gen_fixtures_gemma4.py), compared with
# the dense load; the artifact keeps the connectors' projection and the
# tokenizer; a missing resident is a hard error.

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
src <- fixture("gemma4_tiny_ckpt.safetensors")
if (!file.exists(src)) exit_file("gemma4 fixtures missing")

# Tokenizer straight from the encoder file's tokenizer_json tensor
tok_json <- gemma4_tokenizer(fixture("gemma4_tiny_tokenizer.json"))
tok_file <- gemma4_tokenizer(src)
expect_identical(encode_gemma4(tok_file, "the cat sat", NULL)$input_ids,
                 encode_gemma4(tok_json, "the cat sat", NULL)$input_ids)

dense <- load_gemma4_text_encoder(src, dtype = "float32", pin = FALSE,
                                  verbose = FALSE)
fx <- safetensors::safe_load_file(fixture("gemma4_text.safetensors"),
                                  framework = "torch")
valid <- fx$attention_mask$to(dtype = torch::torch_bool())

art_seen <- local({
  art <- tempfile("gemma4-nf4-")
  on.exit(unlink(art, recursive = TRUE), add = TRUE)

  manifest <- gemma4_quantize_nf4(src, art, verbose = FALSE)
  expect_equal(manifest$model, "gemma4")
  # 4 sliding layers x 7 projections + 2 full (k == v, no v_proj) x 6
  expect_equal(manifest$n_cast, 40L)
  # Self-contained: tokenizer and the connectors' projection travel along
  tok_art <- gemma4_tokenizer(art)
  expect_identical(encode_gemma4(tok_art, "hello cat", NULL)$input_ids,
                   encode_gemma4(tok_json, "hello cat", NULL)$input_ids)
  keys <- unlist(lapply(manifest$shards, function(s) {
    names(safetensors::safe_load_file(file.path(art, s), framework = "torch"))
  }))
  expect_true("text_embedding_projection.video_aggregate_embed.weight" %in% keys)
  expect_false(any(grepl("^vision_model|tokenizer_json", keys)))
  # A second call reuses the artifact
  expect_identical(gemma4_quantize_nf4(src, art, verbose = FALSE)$shards,
                   manifest$shards)
  # ...but only a complete one: a missing tokenizer.json forces a rebuild
  expect_equal(manifest$tokenizer, "tokenizer.json")
  unlink(file.path(art, "tokenizer.json"))
  expect_message(gemma4_quantize_nf4(src, art, verbose = TRUE), "Quantized")
  expect_true(file.exists(file.path(art, "tokenizer.json")))

  # Shards stay under shard_bytes; only a single oversized source tensor
  # (NF4 packed + absmax pair, or one resident) may exceed it alone
  local({
    small <- tempfile("gemma4-nf4-small-")
    on.exit(unlink(small, recursive = TRUE), add = TRUE)
    bound <- 20000
    m <- gemma4_quantize_nf4(src, small, shard_bytes = bound, verbose = FALSE)
    expect_true(length(m$shards) > 1L)
    for (s in m$shards) {
      w <- safetensors::safe_load_file(file.path(small, s), framework = "torch")
      payload <- sum(vapply(w, function(t) t$numel() * t$element_size(), 0))
      n_src <- length(setdiff(names(w), grep("_absmax$", names(w), value = TRUE)))
      expect_true(payload <= bound * 1.01 || n_src == 1L,
                  info = sprintf("%s: %.0f bytes, %d tensors", s, payload, n_src))
    }
  })

  enc <- load_gemma4_text_encoder(art, device = "cpu", dtype = "float32",
                                  pin = FALSE, verbose = FALSE)
  expect_inherits(enc$layers[[1]]$self_attn$q_proj, "ltx23_nf4_linear")
  expect_inherits(enc$layers[[3]]$mlp$down_proj, "ltx23_nf4_linear")
  expect_null(enc$layers[[3]]$self_attn$v_proj)
  # Residents equal the source at the artifact's storage dtype (bf16 when
  # safetensors can write it), the layer_scalar buffers included
  stored <- function(x) {
    w <- safetensors::safe_load_file(file.path(art, manifest$shards[[1]]),
                                     framework = "torch")
    x$to(dtype = w[["embed_tokens.weight"]]$dtype)$to(dtype = torch::torch_float32())
  }
  expect_true(torch::torch_equal(enc$embed_tokens$weight,
                                 stored(dense$embed_tokens$weight)))
  expect_true(torch::torch_equal(enc$layers[[2]]$layer_scalar,
                                 stored(dense$layers[[2]]$layer_scalar)))

  # NF4 weight error per projection: 4-bit noise, not a layout bug
  # (a transposed or misassigned weight would be ~1.4)
  eye <- function(n) torch::torch_eye(n)
  for (lp in list(c(1, "self_attn", "q_proj"), c(3, "self_attn", "k_proj"),
                  c(3, "mlp", "down_proj"))) {
    l <- as.integer(lp[1])
    q <- enc$layers[[l]][[lp[2]]][[lp[3]]]
    w <- dense$layers[[l]][[lp[2]]][[lp[3]]]$weight
    deq <- torch::with_no_grad(q(eye(w$shape[2]))$t())
    rel_w <- as.numeric((deq - w)$norm() / w$norm())
    expect_true(rel_w < 0.15, info = sprintf("%s rel err %.3f", lp[3], rel_w))
  }

  # Load mechanics are exact: the NF4 model equals a dense twin carrying
  # its dequantized weights and stored residents. (End-to-end NF4 noise on
  # this random 6-layer model is large and says nothing about the 12B;
  # real-model quality is checked on real weights.)
  twin <- load_gemma4_text_encoder(src, dtype = "float32", pin = FALSE,
                                   verbose = FALSE)
  torch::with_no_grad({
    tp <- twin$named_parameters()
    ep <- enc$named_parameters()
    for (n in names(tp)) {
      if (grepl("^layers\\.[0-9]+\\.(self_attn|mlp)\\.[a-z]+_proj\\.weight$", n)) {
        segs <- strsplit(sub("\\.weight$", "", n), ".", fixed = TRUE)[[1]]
        lin <- enc$layers[[as.integer(segs[2]) + 1L]][[segs[3]]][[segs[4]]]
        tp[[n]]$copy_(lin(eye(tp[[n]]$shape[2]))$t())
      } else {
        tp[[n]]$copy_(ep[[n]])
      }
    }
    for (n in names(twin$named_buffers())) {
      twin$named_buffers()[[n]]$copy_(enc$named_buffers()[[n]])
    }
    h_nf4 <- enc(fx$input_ids, attention_mask = fx$attention_mask)$last_hidden_state
    h_twin <- twin(fx$input_ids, attention_mask = fx$attention_mask)$last_hidden_state
  })
  expect_true(as.numeric((h_nf4[valid] - h_twin[valid])$abs()$max()) < 1e-5)

  # Layer streaming (small cards): the staging attribute holds the
  # non-layer tensors, one set per layer on the module; on the CPU the
  # staged moves are host-to-host, so the output is identical
  enc_s <- load_gemma4_nf4(art, device = "cpu", dtype = "float32",
                           pin = FALSE, stream = TRUE, verbose = FALSE)
  expect_equal(length(enc_s$layer_stream), length(enc_s$layers))
  tensors <- c(enc_s$parameters, enc_s$buffers)
  expect_equal(length(attr(enc_s, "staging")),
               sum(!startsWith(names(tensors), "layers.")))
  h_s <- torch::with_no_grad(
    enc_s(fx$input_ids, attention_mask = fx$attention_mask)$last_hidden_state)
  expect_true(as.numeric((h_s - h_nf4)$abs()$max()) == 0)

  # Default bf16 compute through the encode step stays finite
  enc_bf <- load_gemma4_nf4(art, device = "cpu", pin = FALSE, verbose = FALSE)
  e <- encode_with_gemma4(c("the cat sat", "tell"), enc_bf, tok_art,
                          max_sequence_length = 6L, device = "cpu",
                          verbose = FALSE)
  expect_true(as.logical(e$prompt_embeds$to(dtype = torch::torch_float32())$
                           isfinite()$all()$item()))

  # An artifact missing a resident is a hard error
  local({
    art2 <- tempfile("gemma4-nf4-bad-")
    on.exit(unlink(art2, recursive = TRUE), add = TRUE)
    dir.create(art2)
    for (s in manifest$shards) {
      w <- safetensors::safe_load_file(file.path(art, s), framework = "torch")
      w[["layers.1.layer_scalar"]] <- NULL
      safetensors::safe_save_file(w, file.path(art2, s))
    }
    file.copy(file.path(art, "manifest.json"), art2)
    expect_error(load_gemma4_nf4(art2, device = "cpu", pin = FALSE,
                                 verbose = FALSE), "not filled")
  })
  art
})
expect_false(dir.exists(art_seen))
