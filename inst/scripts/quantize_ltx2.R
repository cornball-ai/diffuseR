#!/usr/bin/env r
#
# Quantize LTX-2 Transformer and VAE to INT4 format
#
# Usage:
#   r inst/scripts/quantize_ltx2.R
#   r inst/scripts/quantize_ltx2.R --transformer-only
#   r inst/scripts/quantize_ltx2.R --vae-only
#   r inst/scripts/quantize_ltx2.R --force  # Re-quantize even if cached
#
# Output:
#   ~/.cache/diffuseR/ltx2_transformer_int4-NNNNN-of-NNNNN.safetensors
#   ~/.cache/diffuseR/ltx2_vae_int4.safetensors
#
# Requirements:
#   - LTX-2 model downloaded via: huggingface-cli download Lightricks/LTX-2
#   - ~40GB disk space for processing (final output ~11GB)
#

library(torch)

# Parse args
args <- commandArgs(trailingOnly = TRUE)
do_transformer <- !("--vae-only" %in% args)
do_vae <- !("--transformer-only" %in% args)
force <- "--force" %in% args

# Load diffuseR (installed or via rhydrogen)
if (requireNamespace("diffuseR", quietly = TRUE)) {
  library(diffuseR)
} else {
  # Development mode - load from source
  if (file.exists("DESCRIPTION")) {
    rhydrogen::load_all()
  } else {
    stop("diffuseR not installed. Run from package root or install first.")
  }
}

cat("=== LTX-2 INT4 Quantization ===\n\n")

# Quantize transformer
if (do_transformer) {
  cat("--- Transformer ---\n")
  t0 <- Sys.time()
  transformer_path <- quantize_ltx2_transformer(force = force, verbose = TRUE)
  elapsed <- as.numeric(Sys.time() - t0, units = "mins")
  cat(sprintf("Transformer quantization complete in %.1f minutes\n", elapsed))
  cat(sprintf("Output: %s\n\n", transformer_path))
}

# Quantize VAE
if (do_vae) {
  cat("--- VAE ---\n")
  t0 <- Sys.time()
  vae_path <- quantize_ltx2_vae(force = force, verbose = TRUE)
  elapsed <- as.numeric(Sys.time() - t0, units = "secs")
  cat(sprintf("VAE quantization complete in %.1f seconds\n", elapsed))
  cat(sprintf("Output: %s\n\n", vae_path))
}

# Summary
cat("=== Summary ===\n")
cache_dir <- path.expand("~/.cache/diffuseR")
int4_files <- list.files(cache_dir, pattern = "int4.*\\.safetensors$", full.names = TRUE)
if (length(int4_files) > 0) {
  total_gb <- sum(file.info(int4_files)$size) / 1e9
  cat(sprintf("Cached INT4 weights: %.2f GB total\n", total_gb))
  for (f in int4_files) {
    cat(sprintf("  %s: %.2f GB\n", basename(f), file.info(f)$size / 1e9))
  }
}

cat("\nTo load these weights in R:\n")
cat("  q <- load_int4_weights('~/.cache/diffuseR/ltx2_transformer_int4.safetensors')\n")
cat("  w <- dequantize_int4(q[['transformer_blocks.0.attn1.to_q.weight']], device = 'cuda')\n")
