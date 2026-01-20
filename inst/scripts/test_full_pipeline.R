#!/usr/bin/env r
#
# Full LTX-2 Pipeline Test
#
# Tests: Text encoding -> DiT denoising -> (VAE decode)
#

library(torch)
library(diffuseR)

cat("=== LTX-2 Full Pipeline Test ===\n\n")

# Helper to get GPU memory via nvidia-smi
get_gpu_memory <- function() {
  out <- system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", intern = TRUE)
  as.numeric(out[1]) / 1024  # Convert MB to GB
}

# Clear GPU
gc(full = TRUE)
torch::cuda_empty_cache()

mem_start <- get_gpu_memory()
cat(sprintf("Initial GPU memory: %.2f GB\n\n", mem_start))

# Test parameters
test_prompt <- "A cat sitting on a table, looking at the camera"
width <- 512L  # Small resolution for testing
height <- 320L
num_frames <- 17L  # Small number of frames
num_steps <- 4L    # Few steps for quick test

cat(sprintf("Test settings:\n"))
cat(sprintf("  Prompt: %s\n", test_prompt))
cat(sprintf("  Resolution: %dx%d, %d frames\n", width, height, num_frames))
cat(sprintf("  Steps: %d\n\n", num_steps))

# Run pipeline (without VAE for now)
cat("Running pipeline...\n")
t0 <- Sys.time()

tryCatch({
  result <- txt2vid_ltx2(
    prompt = test_prompt,
    width = width,
    height = height,
    num_frames = num_frames,
    num_inference_steps = num_steps,
    guidance_scale = 4.0,
    text_backend = "random",  # Skip real text encoder for speed
    verbose = TRUE
  )

  elapsed <- as.numeric(Sys.time() - t0)
  mem_peak <- get_gpu_memory()

  cat(sprintf("\n=== Results ===\n"))
  cat(sprintf("Elapsed time: %.1f seconds\n", elapsed))
  cat(sprintf("Peak GPU memory: %.2f GB\n", mem_peak))
  cat(sprintf("Output shape: %s\n", paste(dim(result$video), collapse = "x")))
  cat(sprintf("Latent range: [%.3f, %.3f]\n",
              min(result$video), max(result$video)))
  cat("Pipeline: PASSED\n")

}, error = function(e) {
  cat(sprintf("\nPipeline: FAILED\n"))
  cat(sprintf("Error: %s\n", e$message))
  print(traceback())
})

# Cleanup
gc(full = TRUE)
torch::cuda_empty_cache()

mem_end <- get_gpu_memory()
cat(sprintf("\nFinal GPU memory: %.2f GB\n", mem_end))
cat("\n=== Test Complete ===\n")
