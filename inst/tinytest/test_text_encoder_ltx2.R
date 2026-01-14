# Tests for LTX2 Text Encoder and Connectors

# Test 1: 1D RoPE for connectors
cat("Test 1: LTX2RotaryPosEmbed1d\n")
rope <- diffuseR:::ltx2_rotary_pos_embed_1d(
  dim = 512L,
  base_seq_len = 4096L,
  theta = 10000.0,
  rope_type = "interleaved"
)
freqs <- rope(batch_size = 2L, seq_len = 128L, device = "cpu")
expect_equal(length(freqs), 2L, info = "Should return cos and sin")
expect_equal(as.numeric(freqs[[1]]$shape), c(2, 128, 512), info = "Cos shape correct")
expect_equal(as.numeric(freqs[[2]]$shape), c(2, 128, 512), info = "Sin shape correct")

# Test 2: 1D Transformer Block
cat("Test 2: LTX2TransformerBlock1d\n")
block <- diffuseR:::ltx2_transformer_block_1d(
  dim = 512L,
  num_attention_heads = 8L,
  attention_head_dim = 64L
)
x <- torch::torch_randn(c(2, 100, 512))
y <- block(x)
expect_equal(as.numeric(y$shape), as.numeric(x$shape), info = "Block preserves shape")

# Test 3: 1D Transformer Block with RoPE
cat("Test 3: LTX2TransformerBlock1d with RoPE\n")
rope_freqs <- rope(batch_size = 2L, seq_len = 100L, device = "cpu")
y_rope <- block(x, rotary_emb = rope_freqs)
expect_equal(as.numeric(y_rope$shape), as.numeric(x$shape), info = "Block with RoPE preserves shape")

# Test 4: Connector Transformer 1D (without learnable registers)
cat("Test 4: LTX2ConnectorTransformer1d (no registers)\n")
connector <- diffuseR:::ltx2_connector_transformer_1d(
  num_attention_heads = 8L,
  attention_head_dim = 64L,
  num_layers = 2L,
  num_learnable_registers = NULL
)
x <- torch::torch_randn(c(2, 128, 512))
attn_mask <- torch::torch_zeros(c(2, 128))
result <- connector(x, attn_mask)
expect_equal(length(result), 2L, info = "Returns hidden_states and attention_mask")
expect_equal(as.numeric(result[[1]]$shape), c(2, 128, 512), info = "Output shape correct")

# Test 5: Connector Transformer 1D with attention mask (no registers)
cat("Test 5: LTX2ConnectorTransformer1d with mask (no registers)\n")
connector_mask <- diffuseR:::ltx2_connector_transformer_1d(
  num_attention_heads = 8L,
  attention_head_dim = 64L,
  num_layers = 2L,
  num_learnable_registers = NULL  # No registers - simpler for testing
)
x <- torch::torch_randn(c(2, 128, 512))
# Additive attention mask (0 = valid, negative large = masked)
attn_mask <- torch::torch_zeros(c(2, 128))
attn_mask[1, 65:128] <- -10000.0  # Mask second half of first batch
attn_mask[2, 100:128] <- -10000.0  # Mask last 28 of second batch
result <- connector_mask(x, attn_mask)
expect_equal(as.numeric(result[[1]]$shape), c(2, 128, 512), info = "Output shape with mask correct")

# Test 6: Full Text Connectors (small config)
cat("Test 6: LTX2TextConnectors (small config)\n")
connectors <- ltx2_text_connectors(
  caption_channels = 256L,
  text_proj_in_factor = 1L,
  video_connector_num_attention_heads = 4L,
  video_connector_attention_head_dim = 64L,
  video_connector_num_layers = 1L,
  video_connector_num_learnable_registers = NULL,
  audio_connector_num_attention_heads = 4L,
  audio_connector_attention_head_dim = 64L,
  audio_connector_num_layers = 1L,
  audio_connector_num_learnable_registers = NULL
)
expect_true(!is.null(connectors), info = "Connectors instantiate")
cat("  Connectors instantiated\n")

# Test 7: Text Connectors forward pass
cat("Test 7: Text Connectors forward\n")
text_embeds <- torch::torch_randn(c(2, 128, 256))
attn_mask <- torch::torch_ones(c(2, 128))  # All valid
result <- connectors(text_embeds, attn_mask, additive_mask = FALSE)
expect_equal(length(result), 3L, info = "Returns video, audio, and attention mask")
cat(sprintf("  Video embedding shape: [%s]\n", paste(as.numeric(result[[1]]$shape), collapse = ", ")))
cat(sprintf("  Audio embedding shape: [%s]\n", paste(as.numeric(result[[2]]$shape), collapse = ", ")))

# Test 8: encode_text_ltx2 with random backend
cat("Test 8: encode_text_ltx2 (random backend)\n")
result <- encode_text_ltx2(
  prompt = c("A cat sitting on a mat", "A dog running in a field"),
  backend = "random",
  max_sequence_length = 128L,
  caption_channels = 256L
)
expect_true(!is.null(result$prompt_embeds), info = "Returns prompt_embeds")
expect_true(!is.null(result$prompt_attention_mask), info = "Returns attention_mask")
expect_equal(as.numeric(result$prompt_embeds$shape), c(2, 128, 256), info = "Embeddings shape correct")
expect_equal(as.numeric(result$prompt_attention_mask$shape), c(2, 128), info = "Mask shape correct")

# Test 9: pack_text_embeds
cat("Test 9: pack_text_embeds\n")
# Simulate Gemma output: [batch, seq_len, hidden_dim, num_layers]
hidden_states <- torch::torch_randn(c(2, 64, 128, 4))  # 4 layers
sequence_lengths <- c(50L, 60L)  # Valid lengths
packed <- pack_text_embeds(
  hidden_states,
  sequence_lengths,
  padding_side = "left"
)
expect_equal(as.numeric(packed$shape), c(2, 64, 512), info = "Packed shape correct (128 * 4 = 512)")

# Test 10: Full integration - connectors with encoded text
cat("Test 10: Full integration test\n")
torch::with_no_grad({
  # 1. Get text embeddings (random for testing)
  text_result <- encode_text_ltx2(
    prompt = "A beautiful sunset over the ocean",
    backend = "random",
    max_sequence_length = 128L,
    caption_channels = 256L
  )

  # 2. Create connectors
  connectors <- ltx2_text_connectors(
    caption_channels = 256L,
    text_proj_in_factor = 1L,
    video_connector_num_attention_heads = 4L,
    video_connector_attention_head_dim = 64L,
    video_connector_num_layers = 1L,
    video_connector_num_learnable_registers = NULL,
    audio_connector_num_attention_heads = 4L,
    audio_connector_attention_head_dim = 64L,
    audio_connector_num_layers = 1L,
    audio_connector_num_learnable_registers = NULL
  )

  # 3. Process through connectors
  connector_result <- connectors(
    text_result$prompt_embeds,
    text_result$prompt_attention_mask,
    additive_mask = FALSE
  )

  video_embeds <- connector_result[[1]]
  audio_embeds <- connector_result[[2]]
})

expect_true(!is.null(video_embeds), info = "Video embeddings produced")
expect_true(!is.null(audio_embeds), info = "Audio embeddings produced")
cat(sprintf("  Final video embeddings: [%s]\n", paste(as.numeric(video_embeds$shape), collapse = ", ")))
cat(sprintf("  Final audio embeddings: [%s]\n", paste(as.numeric(audio_embeds$shape), collapse = ", ")))

cat("\nAll LTX2 Text Encoder tests completed\n")
