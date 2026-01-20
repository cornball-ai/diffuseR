#!/usr/bin/env r
#
# Test INT4 Memory Usage
#
# Verifies that INT4 weights fit on GPU with room for activations.
#

library(torch)
library(diffuseR)

cat("=== INT4 Memory Test ===\n\n")

# Helper to get GPU memory via nvidia-smi
get_gpu_memory <- function() {
  out <- system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", intern = TRUE)
  as.numeric(out[1]) / 1024  # Convert MB to GB
}

# Clear GPU
gc(full = TRUE)
torch::cuda_empty_cache()

# Initial memory
mem_start <- get_gpu_memory()
cat(sprintf("Initial GPU memory: %.2f GB\n", mem_start))

# Load INT4 weights from disk
cat("\nLoading INT4 weights...\n")
t0 <- Sys.time()
int4_path <- path.expand("~/.cache/diffuseR/ltx2_transformer_int4.safetensors")
q <- load_int4_weights(int4_path, verbose = TRUE)

# Move all INT4 weights to GPU
cat("\nMoving INT4 weights to GPU...\n")
t0 <- Sys.time()
for (name in names(q)) {
  q[[name]]$packed <- q[[name]]$packed$to(device = "cuda")
  q[[name]]$scales <- q[[name]]$scales$to(device = "cuda")
}
elapsed <- as.numeric(Sys.time() - t0)
cat(sprintf("Moved to GPU in %.2fs\n", elapsed))

mem_after_load <- get_gpu_memory()
cat(sprintf("GPU memory after INT4 load: %.2f GB\n", mem_after_load))
cat(sprintf("INT4 weights on GPU: %.2f GB\n", mem_after_load - mem_start))

# Simulate activations for a typical forward pass
cat("\nSimulating activations...\n")

# Typical video generation: 720p, 121 frames, batch 1
# Latent size: 768/32 x 512/32 x (121-1)/8+1 = 24 x 16 x 16
batch_size <- 1L
seq_len <- 24L * 16L * 16L  # 6144 patches
hidden_dim <- 4096L

# Create activation tensors (hidden states, attention, etc.)
hidden_states <- torch::torch_randn(c(batch_size, seq_len, hidden_dim),
                                     dtype = torch::torch_float16(), device = "cuda")
# Attention needs Q, K, V plus output
attn_q <- torch::torch_randn(c(batch_size, 32L, seq_len, 128L),
                              dtype = torch::torch_float16(), device = "cuda")
attn_k <- torch::torch_randn(c(batch_size, 32L, seq_len, 128L),
                              dtype = torch::torch_float16(), device = "cuda")
attn_v <- torch::torch_randn(c(batch_size, 32L, seq_len, 128L),
                              dtype = torch::torch_float16(), device = "cuda")

mem_with_activations <- get_gpu_memory()
cat(sprintf("GPU memory with activations: %.2f GB\n", mem_with_activations))
cat(sprintf("Activation memory: %.2f GB\n", mem_with_activations - mem_after_load))

# Test a forward pass through one layer
cat("\nTesting INT4 forward pass...\n")
weight_name <- "transformer_blocks.0.attn1.to_q.weight"
q_weight <- q[[weight_name]]

# Create INT4 layer and do forward
int4_layer <- int4_linear_from_quantized(q_weight, device = "cuda",
                                          dtype = torch::torch_float16())

x_in <- hidden_states[, 1:128, ]  # Small batch for testing
t0 <- Sys.time()
y_out <- int4_layer(x_in)
elapsed <- as.numeric(Sys.time() - t0) * 1000
cat(sprintf("Forward pass: %.1fms\n", elapsed))

mem_during_forward <- get_gpu_memory()
cat(sprintf("GPU memory during forward: %.2f GB\n", mem_during_forward))

# Cleanup test tensors
rm(hidden_states, attn_q, attn_k, attn_v, x_in, y_out, int4_layer)
gc()
torch::cuda_empty_cache()

mem_after_cleanup <- get_gpu_memory()
cat(sprintf("\nGPU memory after cleanup: %.2f GB\n", mem_after_cleanup))

# Summary
cat("\n=== Summary ===\n")
cat(sprintf("INT4 weights: %.2f GB\n", mem_after_load - mem_start))
cat(sprintf("Peak activations: %.2f GB\n", mem_with_activations - mem_after_load))
cat(sprintf("Peak total: %.2f GB\n", mem_with_activations))
cat(sprintf("Available headroom: %.2f GB (on 16GB GPU)\n", 16 - mem_with_activations))

if (mem_with_activations < 14) {
  cat("\nVERDICT: INT4 model should fit on 16GB GPU with room to spare\n")
} else if (mem_with_activations < 15.5) {
  cat("\nVERDICT: INT4 model fits on 16GB GPU (tight)\n")
} else {
  cat("\nVERDICT: INT4 model may not fit - need optimization\n")
}

cat("\n=== Test Complete ===\n")
