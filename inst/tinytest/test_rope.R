# Tests for RoPE (Rotary Position Embeddings)

# Test embedder creation
expect_silent({
  embedder <- rope_embedder_create(
    dim = 2048,
    patch_size = 1,
    patch_size_t = 1
  )
})

expect_equal(embedder$dim, 2048)
expect_equal(embedder$rope_type, "interleaved")
expect_equal(embedder$theta, 10000.0)

# Test video coordinate preparation
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

expect_true(inherits(coords, "torch_tensor"))

# Check shape: (batch_size, 3, num_patches, 2)
# num_patches = num_frames * height * width = 4 * 16 * 16 = 1024
expect_equal(as.numeric(coords$shape[1]), 2)  # batch_size
expect_equal(as.numeric(coords$shape[2]), 3)  # 3 dimensions (f, h, w)
expect_equal(as.numeric(coords$shape[3]), 4 * 16 * 16)  # num_patches
expect_equal(as.numeric(coords$shape[4]), 2)  # start, end

# Test RoPE forward (frequency computation)
freqs <- rope_forward(embedder, coords, device = "cpu")

expect_true(is.list(freqs))
expect_true(inherits(freqs$cos_freqs, "torch_tensor"))
expect_true(inherits(freqs$sin_freqs, "torch_tensor"))

# cos and sin should have same shape
expect_equal(as.numeric(freqs$cos_freqs$shape), as.numeric(freqs$sin_freqs$shape))

# cos values should be in [-1, 1]
expect_true(as.numeric(freqs$cos_freqs$min()) >= -1.0)
expect_true(as.numeric(freqs$cos_freqs$max()) <= 1.0)

# sin values should be in [-1, 1]
expect_true(as.numeric(freqs$sin_freqs$min()) >= -1.0)
expect_true(as.numeric(freqs$sin_freqs$max()) <= 1.0)

# Test apply_interleaved_rotary_emb
batch_size <- 2
seq_len <- 1024
channels <- 2048

x <- torch::torch_randn(c(batch_size, seq_len, channels))

# Need to ensure freqs match x's sequence length
embedder_small <- rope_embedder_create(dim = channels)
coords_small <- rope_prepare_video_coords(
  embedder = embedder_small,
  batch_size = batch_size,
  num_frames = 4,
  height = 16,
  width = 16,
  device = "cpu"
)
freqs_small <- rope_forward(embedder_small, coords_small, device = "cpu")

# Apply rotation
rotated <- apply_interleaved_rotary_emb(x, freqs_small)

expect_true(inherits(rotated, "torch_tensor"))
expect_equal(as.numeric(rotated$shape), c(batch_size, seq_len, channels))

# Rotated should differ from original
expect_false(torch::torch_allclose(x, rotated))

# Test that applying rotation twice with negated freqs returns original
# (RoPE is a rotation, so -rotation undoes it)
freqs_neg <- list(
  cos_freqs = freqs_small$cos_freqs,
  sin_freqs = -freqs_small$sin_freqs
)
double_rotated <- apply_interleaved_rotary_emb(rotated, freqs_neg)

# Should be close to original (numerical precision)
diff <- (x - double_rotated)$abs()$max()$item()
expect_true(diff < 1e-5, info = sprintf("Double rotation diff: %f", diff))

# Test with different patch sizes
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

# num_patches = (8/2) * (32/2) * (32/2) = 4 * 16 * 16 = 1024
expect_equal(as.numeric(coords_patched$shape[3]), 4 * 16 * 16)

# Test GPU if available
if (torch::cuda_is_available()) {
  coords_gpu <- rope_prepare_video_coords(
    embedder = embedder,
    batch_size = 1,
    num_frames = 4,
    height = 16,
    width = 16,
    device = "cuda"
  )

  freqs_gpu <- rope_forward(embedder, coords_gpu, device = "cuda")

  expect_equal(as.character(freqs_gpu$cos_freqs$device), "cuda:0")

  x_gpu <- torch::torch_randn(c(1, 1024, 2048), device = "cuda")
  rotated_gpu <- apply_interleaved_rotary_emb(x_gpu, freqs_gpu)
  expect_equal(as.character(rotated_gpu$device), "cuda:0")
}

cat("All RoPE tests passed\n")
