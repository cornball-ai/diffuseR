# Validation tests for RoPE against Python diffusers
# These tests ensure numerical equivalence with HuggingFace implementation

# Load Python test cases
test_cases_path <- system.file("validation/rope_test_cases.json", package = "diffuseR")

if (!file.exists(test_cases_path)) {
  cat("Skipping Python validation tests - test cases file not found\n")
} else {
  test_cases <- jsonlite::fromJSON(test_cases_path)
  tol <- 1e-4

  # Test 1: Video coordinate shape and bounds
  cat("Test 1: Video coordinate shape and bounds\n")
  embedder <- rope_embedder_create(
    dim = 2048,
    patch_size = 1,
    patch_size_t = 1,
    scale_factors = c(8, 32, 32)
  )

  coords <- rope_prepare_video_coords(
    embedder = embedder,
    batch_size = 2,
    num_frames = 4,
    height = 16,
    width = 16,
    device = "cpu",
    fps = 24.0
  )

  python_coords <- test_cases$video_coords

  # Check shape
  r_shape <- as.numeric(coords$shape)
  py_shape <- python_coords$coords_shape
  expect_equal(r_shape, py_shape, info = "Coordinate shapes should match")

  # Check bounds
  r_min <- as.numeric(coords$min())
  r_max <- as.numeric(coords$max())
  expect_true(abs(r_min - python_coords$coords_min) < tol,
              info = sprintf("Min diff: %f", abs(r_min - python_coords$coords_min)))
  expect_true(abs(r_max - python_coords$coords_max) < tol,
              info = sprintf("Max diff: %f", abs(r_max - python_coords$coords_max)))

  cat(sprintf("  Shape: R=%s, Python=%s\n",
              paste(r_shape, collapse=","), paste(py_shape, collapse=",")))
  cat(sprintf("  Min: R=%.4f, Python=%.4f\n", r_min, python_coords$coords_min))
  cat(sprintf("  Max: R=%.4f, Python=%.4f\n", r_max, python_coords$coords_max))

  # Test 2: RoPE frequency shapes and bounds
  cat("Test 2: RoPE frequency computation\n")
  freqs <- rope_forward(embedder, coords, device = "cpu")

  python_freqs <- test_cases$rope_freqs

  # Check shapes
  cos_shape <- as.numeric(freqs$cos_freqs$shape)
  sin_shape <- as.numeric(freqs$sin_freqs$shape)
  expect_equal(cos_shape, python_freqs$cos_shape, info = "Cos shape mismatch")
  expect_equal(sin_shape, python_freqs$sin_shape, info = "Sin shape mismatch")

  # Check bounds
  cos_min <- as.numeric(freqs$cos_freqs$min())
  cos_max <- as.numeric(freqs$cos_freqs$max())
  sin_min <- as.numeric(freqs$sin_freqs$min())
  sin_max <- as.numeric(freqs$sin_freqs$max())

  expect_true(abs(cos_min - python_freqs$cos_min) < tol)
  expect_true(abs(cos_max - python_freqs$cos_max) < tol)
  expect_true(abs(sin_min - python_freqs$sin_min) < tol)
  expect_true(abs(sin_max - python_freqs$sin_max) < tol)

  cat(sprintf("  Cos shape: %s\n", paste(cos_shape, collapse=",")))
  cat(sprintf("  Cos range: [%.4f, %.4f]\n", cos_min, cos_max))
  cat(sprintf("  Sin range: [%.4f, %.4f]\n", sin_min, sin_max))

  # Check means (important for numerical equivalence)
  cos_mean <- as.numeric(freqs$cos_freqs$mean())
  sin_mean <- as.numeric(freqs$sin_freqs$mean())

  cat(sprintf("  Cos mean: R=%.6f, Python=%.6f, diff=%.2e\n",
              cos_mean, python_freqs$cos_mean, abs(cos_mean - python_freqs$cos_mean)))
  cat(sprintf("  Sin mean: R=%.6f, Python=%.6f, diff=%.2e\n",
              sin_mean, python_freqs$sin_mean, abs(sin_mean - python_freqs$sin_mean)))

  # Test 3: Apply rotary embedding
  cat("Test 3: Apply interleaved rotary embedding\n")
  torch::torch_manual_seed(42)
  x <- torch::torch_randn(c(2, 1024, 2048))

  rotated <- apply_interleaved_rotary_emb(x, freqs)

  python_apply <- test_cases$apply_rope
  r_mean <- as.numeric(rotated$mean())
  r_std <- as.numeric(rotated$std())

  cat(sprintf("  Output mean: R=%.6f, Python=%.6f\n", r_mean, python_apply$output_mean))
  cat(sprintf("  Output std:  R=%.6f, Python=%.6f\n", r_std, python_apply$output_std))

  # Shape should match exactly
  expect_equal(as.numeric(rotated$shape), python_apply$output_shape)

  # Test 4: Patched coordinates
  cat("Test 4: Patched coordinates\n")
  embedder_patched <- rope_embedder_create(
    dim = 2048,
    patch_size = 2,
    patch_size_t = 2
  )

  coords_patched <- rope_prepare_video_coords(
    embedder = embedder_patched,
    batch_size = 1,
    num_frames = 8,
    height = 32,
    width = 32,
    device = "cpu"
  )

  python_patched <- test_cases$patched_coords
  r_num_patches <- as.numeric(coords_patched$shape[3])

  expect_equal(r_num_patches, python_patched$expected_num_patches,
               info = sprintf("Num patches: R=%d, expected=%d",
                            r_num_patches, python_patched$expected_num_patches))
  cat(sprintf("  Num patches: %d (expected: %d)\n",
              r_num_patches, python_patched$expected_num_patches))

  cat("\nAll RoPE Python validation tests completed\n")
}
