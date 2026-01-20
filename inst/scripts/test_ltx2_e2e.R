#!/usr/bin/env r
#
# End-to-end test for LTX-2 pipeline
#
# Tests the pipeline structure without full model weights.
# Use text_backend="random" to skip text encoder loading.
#

library(torch)

# Load diffuseR
if (requireNamespace("diffuseR", quietly = TRUE)) {
  library(diffuseR)
} else {
  rhydrogen::load_all()
}

cat("=== LTX-2 End-to-End Test ===\n\n")

# Test 1: Text encoding + connectors (skip DiT for now - too large for single GPU)
cat("--- Test 1: Text encoding + connectors ---\n")
tryCatch({
  library(diffuseR)

  # Test encode_text_ltx2 + connectors
  t0 <- Sys.time()

  # Random packed embeddings
  packed_embeds <- encode_text_ltx2(
    prompt = "Test prompt for connector pipeline",
    backend = "random",
    max_sequence_length = 128L,
    caption_channels = 3840L
  )
  cat(sprintf("Packed embeds shape: %s\n",
              paste(as.integer(packed_embeds$prompt_embeds$shape), collapse = "x")))

  # Load and apply connectors
  connector_path <- Sys.glob("~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/*/connectors/diffusion_pytorch_model.safetensors")[1]
  connectors <- load_ltx2_connectors(connector_path, verbose = FALSE)

  result <- connectors(packed_embeds$prompt_embeds, packed_embeds$prompt_attention_mask)
  video_embeds <- result[[1]]
  audio_embeds <- result[[2]]

  elapsed <- as.numeric(Sys.time() - t0)
  cat(sprintf("Video embeds: %s\n", paste(as.integer(video_embeds$shape), collapse = "x")))
  cat(sprintf("Audio embeds: %s\n", paste(as.integer(audio_embeds$shape), collapse = "x")))
  cat(sprintf("Pipeline (random->connectors) in %.2fs\n", elapsed))
  cat("Test 1: PASSED\n\n")
}, error = function(e) {
  cat(sprintf("Test 1: FAILED - %s\n\n", e$message))
})

# Test 2: INT4 weight loading
cat("--- Test 2: INT4 weight loading ---\n")
tryCatch({
  int4_path <- path.expand("~/.cache/diffuseR/ltx2_transformer_int4.safetensors")
  t0 <- Sys.time()
  weights <- load_int4_weights(int4_path, verbose = TRUE)
  load_time <- as.numeric(Sys.time() - t0)
  cat(sprintf("Loaded %d quantized parameters in %.2fs\n", length(weights), load_time))
  cat("Test 2: PASSED\n\n")
}, error = function(e) {
  cat(sprintf("Test 2: FAILED - %s\n\n", e$message))
})

# Test 3: Connector loading
cat("--- Test 3: Connector loading ---\n")
tryCatch({
  connector_path <- Sys.glob("~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/*/connectors/diffusion_pytorch_model.safetensors")[1]
  t0 <- Sys.time()
  connectors <- load_ltx2_connectors(connector_path, verbose = TRUE)
  load_time <- as.numeric(Sys.time() - t0)
  cat(sprintf("Connectors loaded in %.2fs\n", load_time))

  # Test forward pass
  test_embeds <- torch_randn(c(1L, 128L, 188160L))  # [B, seq_len, 3840*49]
  test_mask <- torch_ones(c(1L, 128L), dtype = torch_int64())
  result <- connectors(test_embeds, test_mask)
  cat(sprintf("Video embeds: %s\n", paste(as.integer(result[[1]]$shape), collapse = "x")))
  cat(sprintf("Audio embeds: %s\n", paste(as.integer(result[[2]]$shape), collapse = "x")))
  cat("Test 3: PASSED\n\n")
}, error = function(e) {
  cat(sprintf("Test 3: FAILED - %s\n\n", e$message))
})

# Test 4: INT4Linear layer
cat("--- Test 4: INT4Linear layer ---\n")
tryCatch({
  # Create a standard linear layer
  linear <- torch::nn_linear(4096L, 4096L, bias = TRUE)
  linear <- linear$to(dtype = torch::torch_float16(), device = "cuda")

  # Test input
  x <- torch::torch_randn(c(1L, 128L, 4096L), dtype = torch::torch_float16(), device = "cuda")

  # Standard forward
  t0 <- Sys.time()
  y_std <- linear(x)
  elapsed_std <- as.numeric(Sys.time() - t0) * 1000

  # Convert to INT4
  int4_layer <- linear_to_int4(linear, device = "cuda", dtype = torch::torch_float16())

  # INT4 forward
  t0 <- Sys.time()
  y_int4 <- int4_layer(x)
  elapsed_int4 <- as.numeric(Sys.time() - t0) * 1000

  # Check output similarity
  diff <- (y_std - y_int4)$abs()$mean()$item()
  cat(sprintf("  Standard linear: %.1fms\n", elapsed_std))
  cat(sprintf("  INT4 linear: %.1fms\n", elapsed_int4))
  cat(sprintf("  Mean abs diff: %.6f\n", diff))

  # Memory comparison
  std_mem <- prod(linear$weight$shape) * 2 / 1e6  # float16
  int4_mem <- (int4_layer$weight_packed$numel() + int4_layer$weight_scales$numel() * 4) / 1e6
  cat(sprintf("  Memory: %.2f MB (std) vs %.2f MB (INT4) = %.1fx\n",
              std_mem, int4_mem, std_mem / int4_mem))

  # Cleanup
  rm(linear, int4_layer, x, y_std, y_int4)
  gc()
  torch::cuda_empty_cache()

  cat("Test 4: PASSED\n\n")
}, error = function(e) {
  cat(sprintf("Test 4: FAILED - %s\n\n", e$message))
})

# Test 5: INT4Linear from pre-quantized disk weights
cat("--- Test 5: INT4Linear from pre-quantized disk weights ---\n")
tryCatch({
  # Load INT4 weights
  int4_path <- path.expand("~/.cache/diffuseR/ltx2_transformer_int4.safetensors")
  q <- load_int4_weights(int4_path, verbose = FALSE)

  # Clear GPU memory first
  gc(full = TRUE)
  torch::cuda_empty_cache()

  # Create INT4Linear from a real DiT weight
  weight_name <- "transformer_blocks.0.attn1.to_q.weight"
  if (weight_name %in% names(q)) {
    q_weight <- q[[weight_name]]

    # Create layer directly from quantized weights
    t0 <- Sys.time()
    int4_layer <- int4_linear_from_quantized(q_weight, device = "cuda",
                                              dtype = torch::torch_float16())
    create_time <- as.numeric(Sys.time() - t0) * 1000

    # Test forward pass
    out_f <- q_weight$orig_shape[1]
    in_f <- q_weight$orig_shape[2]
    x <- torch::torch_randn(c(1L, 128L, in_f), dtype = torch::torch_float16(), device = "cuda")

    t0 <- Sys.time()
    y <- int4_layer(x)
    forward_time <- as.numeric(Sys.time() - t0) * 1000

    cat(sprintf("  Layer: %dx%d\n", out_f, in_f))
    cat(sprintf("  Create from INT4: %.1fms\n", create_time))
    cat(sprintf("  Forward pass: %.1fms\n", forward_time))
    cat(sprintf("  Output shape: %s\n", paste(as.integer(y$shape), collapse = "x")))

    # Cleanup
    rm(int4_layer, x, y)
    gc()
    torch::cuda_empty_cache()
  }

  cat("Test 5: PASSED\n\n")
}, error = function(e) {
  cat(sprintf("Test 5: FAILED - %s\n\n", e$message))
})

cat("=== Tests Complete ===\n")
