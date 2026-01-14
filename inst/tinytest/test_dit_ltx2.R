# Tests for LTX2 DiT Transformer modules

# Test 1: Timesteps module
cat("Test 1: Timesteps module\n")
ts <- diffuseR:::timesteps_module(num_channels = 256L, flip_sin_to_cos = TRUE, downscale_freq_shift = 0)
timesteps <- torch::torch_tensor(c(0.5, 0.8))
emb <- ts(timesteps)
expect_equal(as.numeric(emb$shape[2]), 256L, info = "Timestep embedding should have 256 channels")

# Test 2: TimestepEmbedding MLP
cat("Test 2: TimestepEmbedding MLP\n")
te <- diffuseR:::timestep_embedding_module(in_channels = 256L, time_embed_dim = 512L)
x <- torch::torch_randn(c(2, 256))
y <- te(x)
expect_equal(as.numeric(y$shape[2]), 512L, info = "Output should have 512 channels")

# Test 3: PixArtAlphaCombinedTimestepSizeEmbeddings
cat("Test 3: PixArtAlphaCombinedTimestepSizeEmbeddings\n")
emb_module <- diffuseR:::pixart_alpha_combined_timestep_size_embeddings(
  embedding_dim = 512L,
  size_emb_dim = 170L,
  use_additional_conditions = FALSE
)
timesteps <- torch::torch_tensor(c(500.0, 750.0))
cond <- emb_module(timesteps, batch_size = 2L, hidden_dtype = torch::torch_float32())
expect_true(!is.null(cond), info = "Should return conditioning tensor")
cat(sprintf("  Conditioning shape: [%s]\n", paste(as.numeric(cond$shape), collapse = ", ")))

# Test 4: PixArtAlphaTextProjection
cat("Test 4: PixArtAlphaTextProjection\n")
text_proj <- diffuseR:::pixart_alpha_text_projection(
  in_features = 3840L,
  hidden_size = 4096L
)
caption <- torch::torch_randn(c(2, 77, 3840))
proj <- text_proj(caption)
expect_equal(as.numeric(proj$shape[3]), 4096L, info = "Projected caption should have 4096 channels")

# Test 5: RMSNorm
cat("Test 5: RMSNorm\n")
norm <- diffuseR:::rms_norm(dim = 512L, eps = 1e-6)
x <- torch::torch_randn(c(2, 10, 512))
y <- norm(x)
expect_equal(as.numeric(y$shape), as.numeric(x$shape), info = "RMS norm should preserve shape")

# Test 6: FeedForward
cat("Test 6: FeedForward\n")
ff <- diffuseR:::feed_forward(dim = 512L, mult = 4L, activation_fn = "gelu-approximate")
x <- torch::torch_randn(c(2, 10, 512))
y <- ff(x)
expect_equal(as.numeric(y$shape), as.numeric(x$shape), info = "FFN should preserve shape")

# Test 7: LTX2AdaLayerNormSingle
cat("Test 7: LTX2AdaLayerNormSingle\n")
ada_ln <- diffuseR:::ltx2_ada_layer_norm_single(embedding_dim = 512L, num_mod_params = 6L)
timesteps <- torch::torch_tensor(c(500.0, 750.0))
result <- ada_ln(timesteps, batch_size = 2L, hidden_dtype = torch::torch_float32())
expect_equal(length(result), 2L, info = "Should return mod_params and embedded_timestep")
cat(sprintf("  Mod params shape: [%s]\n", paste(as.numeric(result[[1]]$shape), collapse = ", ")))

# Test 8: LTX2Attention
cat("Test 8: LTX2Attention\n")
attn <- diffuseR:::ltx2_attention(
  query_dim = 512L,
  heads = 8L,
  kv_heads = 8L,
  dim_head = 64L,
  rope_type = "interleaved"
)
x <- torch::torch_randn(c(2, 100, 512))
y <- attn(x)
expect_equal(as.numeric(y$shape), as.numeric(x$shape), info = "Self-attention should preserve shape")

# Test 9: LTX2Attention with RoPE
cat("Test 9: LTX2Attention with RoPE\n")
# Create mock RoPE frequencies
cos_freqs <- torch::torch_ones(c(2, 100, 512))
sin_freqs <- torch::torch_zeros(c(2, 100, 512))
rope_freqs <- list(cos_freqs, sin_freqs)

y_rope <- attn(x, query_rotary_emb = rope_freqs)
expect_equal(as.numeric(y_rope$shape), as.numeric(x$shape), info = "Attention with RoPE should preserve shape")

# Test 10: LTX2Attention cross-attention
cat("Test 10: LTX2Attention cross-attention\n")
cross_attn <- diffuseR:::ltx2_attention(
  query_dim = 512L,
  heads = 8L,
  kv_heads = 8L,
  dim_head = 64L,
  cross_attention_dim = 2048L
)
q <- torch::torch_randn(c(2, 100, 512))
kv <- torch::torch_randn(c(2, 77, 2048))
y_cross <- cross_attn(q, encoder_hidden_states = kv)
expect_equal(as.numeric(y_cross$shape), as.numeric(q$shape), info = "Cross-attention should output query shape")

# Test 11: LTX2AudioVideoRotaryPosEmbed (video)
cat("Test 11: LTX2AudioVideoRotaryPosEmbed (video)\n")
rope <- diffuseR:::ltx2_audio_video_rotary_pos_embed(
  dim = 4096L,
  patch_size = 1L,
  patch_size_t = 1L,
  base_num_frames = 20L,
  base_height = 2048L,
  base_width = 2048L,
  scale_factors = c(8L, 32L, 32L),
  modality = "video",
  rope_type = "interleaved"
)

video_coords <- rope$prepare_video_coords(
  batch_size = 2L,
  num_frames = 4L,
  height = 16L,
  width = 16L,
  device = "cpu",
  fps = 24.0
)
expect_equal(video_coords$shape[2], 3L, info = "Video coords should have 3 dims (T, H, W)")

freqs <- rope(video_coords)
expect_equal(length(freqs), 2L, info = "Should return cos and sin freqs")
cat(sprintf("  Video RoPE cos shape: [%s]\n", paste(as.numeric(freqs[[1]]$shape), collapse = ", ")))

# Test 12: LTX2AudioVideoRotaryPosEmbed (audio)
cat("Test 12: LTX2AudioVideoRotaryPosEmbed (audio)\n")
audio_rope <- diffuseR:::ltx2_audio_video_rotary_pos_embed(
  dim = 2048L,
  patch_size_t = 1L,
  base_num_frames = 20L,
  scale_factors = c(4L),
  modality = "audio",
  rope_type = "interleaved"
)

audio_coords <- audio_rope$prepare_audio_coords(
  batch_size = 2L,
  num_frames = 100L,
  device = "cpu"
)
expect_equal(audio_coords$shape[2], 1L, info = "Audio coords should have 1 dim (T)")

audio_freqs <- audio_rope(audio_coords)
expect_equal(length(audio_freqs), 2L, info = "Should return cos and sin freqs")
cat(sprintf("  Audio RoPE cos shape: [%s]\n", paste(as.numeric(audio_freqs[[1]]$shape), collapse = ", ")))

# Test 13: LTX2VideoTransformerBlock
cat("Test 13: LTX2VideoTransformerBlock\n")
block <- diffuseR:::ltx2_video_transformer_block(
  dim = 512L,
  num_attention_heads = 8L,
  attention_head_dim = 64L,
  cross_attention_dim = 2048L,
  audio_dim = 256L,
  audio_num_attention_heads = 4L,
  audio_attention_head_dim = 64L,
  audio_cross_attention_dim = 1024L
)

# Prepare inputs
hidden_states <- torch::torch_randn(c(2, 100, 512))
audio_hidden_states <- torch::torch_randn(c(2, 50, 256))
encoder_hidden_states <- torch::torch_randn(c(2, 77, 2048))
audio_encoder_hidden_states <- torch::torch_randn(c(2, 77, 1024))
temb <- torch::torch_randn(c(2, 1, 3072))  # 6 * 512
temb_audio <- torch::torch_randn(c(2, 1, 1536))  # 6 * 256
temb_ca_scale_shift <- torch::torch_randn(c(2, 1, 2048))  # 4 * 512
temb_ca_audio_scale_shift <- torch::torch_randn(c(2, 1, 1024))  # 4 * 256
temb_ca_gate <- torch::torch_randn(c(2, 1, 512))  # 1 * 512
temb_ca_audio_gate <- torch::torch_randn(c(2, 1, 256))  # 1 * 256

# Mock RoPE (ones for cos, zeros for sin = identity)
video_rope_freqs <- list(torch::torch_ones(c(2, 100, 512)), torch::torch_zeros(c(2, 100, 512)))
audio_rope_freqs <- list(torch::torch_ones(c(2, 50, 256)), torch::torch_zeros(c(2, 50, 256)))
ca_video_rope <- list(torch::torch_ones(c(2, 100, 256)), torch::torch_zeros(c(2, 100, 256)))
ca_audio_rope <- list(torch::torch_ones(c(2, 50, 256)), torch::torch_zeros(c(2, 50, 256)))

result <- block(
  hidden_states = hidden_states,
  audio_hidden_states = audio_hidden_states,
  encoder_hidden_states = encoder_hidden_states,
  audio_encoder_hidden_states = audio_encoder_hidden_states,
  temb = temb,
  temb_audio = temb_audio,
  temb_ca_scale_shift = temb_ca_scale_shift,
  temb_ca_audio_scale_shift = temb_ca_audio_scale_shift,
  temb_ca_gate = temb_ca_gate,
  temb_ca_audio_gate = temb_ca_audio_gate,
  video_rotary_emb = video_rope_freqs,
  audio_rotary_emb = audio_rope_freqs,
  ca_video_rotary_emb = ca_video_rope,
  ca_audio_rotary_emb = ca_audio_rope
)

expect_equal(length(result), 2L, info = "Block should return video and audio hidden states")
expect_equal(as.numeric(result[[1]]$shape), as.numeric(hidden_states$shape), info = "Video shape preserved")
expect_equal(as.numeric(result[[2]]$shape), as.numeric(audio_hidden_states$shape), info = "Audio shape preserved")
cat("  Block forward pass successful\n")

# Test 14: Full LTX2VideoTransformer3DModel (small config)
cat("Test 14: LTX2VideoTransformer3DModel (small config)\n")
model <- ltx2_video_transformer_3d_model(
  in_channels = 32L,
  out_channels = 32L,
  num_attention_heads = 4L,
  attention_head_dim = 32L,
  cross_attention_dim = 128L,   # Must equal inner_dim = 4*32 = 128 (after caption projection)
  audio_in_channels = 16L,
  audio_out_channels = 16L,
  audio_num_attention_heads = 2L,
  audio_attention_head_dim = 32L,
  audio_cross_attention_dim = 64L,  # Must equal audio_inner_dim = 2*32 = 64
  num_layers = 2L,
  caption_channels = 512L,
  vae_scale_factors = c(8L, 32L, 32L)
)
expect_true(!is.null(model), info = "Model should instantiate")
cat("  Model instantiated with 2 layers\n")

# Test 15: Full model forward pass
cat("Test 15: Full model forward pass\n")
torch::with_no_grad({
  batch_size <- 1L
  num_frames <- 4L
  height <- 8L
  width <- 8L
  num_patches <- num_frames * height * width

  hidden_states <- torch::torch_randn(c(batch_size, num_patches, 32L))
  audio_hidden_states <- torch::torch_randn(c(batch_size, 50L, 16L))
  encoder_hidden_states <- torch::torch_randn(c(batch_size, 77L, 512L))
  audio_encoder_hidden_states <- torch::torch_randn(c(batch_size, 77L, 512L))  # Must match caption_channels
  timestep <- torch::torch_tensor(c(500.0))$unsqueeze(2)

  output <- model(
    hidden_states = hidden_states,
    audio_hidden_states = audio_hidden_states,
    encoder_hidden_states = encoder_hidden_states,
    audio_encoder_hidden_states = audio_encoder_hidden_states,
    timestep = timestep,
    num_frames = num_frames,
    height = height,
    width = width,
    fps = 24.0,
    audio_num_frames = 50L
  )
})

expect_equal(length(output), 2L, info = "Model should return sample and audio_sample")
expect_equal(as.numeric(output$sample$shape), c(1L, num_patches, 32L), info = "Video output shape correct")
expect_equal(as.numeric(output$audio_sample$shape), c(1L, 50L, 16L), info = "Audio output shape correct")
cat(sprintf("  Video output shape: [%s]\n", paste(as.numeric(output$sample$shape), collapse = ", ")))
cat(sprintf("  Audio output shape: [%s]\n", paste(as.numeric(output$audio_sample$shape), collapse = ", ")))

cat("\nAll LTX2 DiT transformer tests completed\n")
