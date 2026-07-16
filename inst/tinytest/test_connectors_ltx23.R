# Parity tests for the LTX-2.3 text connectors against diffusers
# reference fixtures (tools/gen_fixtures_connectors.py).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

fixture_path <- system.file("tinytest", "fixtures", "connectors_ltx23.safetensors",
  package = "diffuseR")
if (fixture_path == "") fixture_path <- "fixtures/connectors_ltx23.safetensors"
if (!file.exists(fixture_path)) exit_file("connector fixtures missing")

fx <- safetensors::safe_load_file(fixture_path, framework = "torch")

max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(
    a$to(dtype = torch::torch_float32()) - b$to(dtype = torch::torch_float32())
  )))
}

conn <- ltx23_text_connectors(
  caption_channels = 8L,
  text_proj_in_factor = 3L,
  video_connector_num_attention_heads = 2L,
  video_connector_attention_head_dim = 8L,
  video_connector_num_layers = 2L,
  video_connector_num_learnable_registers = 4L,
  video_gated_attn = TRUE,
  audio_connector_num_attention_heads = 2L,
  audio_connector_attention_head_dim = 4L,
  audio_connector_num_layers = 2L,
  audio_connector_num_learnable_registers = 4L,
  audio_gated_attn = TRUE,
  rope_type = "split",
  video_hidden_dim = 16L,
  audio_hidden_dim = 8L,
  proj_bias = TRUE
)
conn$eval()

weights <- fx[grep("^conn\\.", names(fx))]
names(weights) <- sub("^conn\\.", "", names(weights))
dests <- c(conn$named_parameters(), conn$named_buffers())
expect_equal(sort(names(weights)), sort(names(dests)))
torch::with_no_grad({
  for (name in names(weights)) dests[[name]]$copy_(weights[[name]])
})

torch::with_no_grad({
  res <- conn(fx$c_states, fx$c_mask)
})
expect_equal(as.integer(res$video_text_embedding$shape), as.integer(fx$c_video_emb$shape))
expect_true(max_abs_diff(res$video_text_embedding, fx$c_video_emb) < 1e-4)
expect_true(max_abs_diff(res$audio_text_embedding, fx$c_audio_emb) < 1e-4)
expect_true(max_abs_diff(
  res$attention_mask$to(dtype = torch::torch_float32()), fx$c_out_mask
) < 1e-6)

# Flattened 3D input gives identical output
torch::with_no_grad({
  res3 <- conn(fx$c_states$flatten(start_dim = 3L), fx$c_mask)
})
# 1e-4 to match the sibling assertions above: float32 matmul
# accumulation order differs across libtorch builds (CRAN torch lands
# between 1e-5 and 1e-4 against fixtures generated on a newer libtorch)
expect_true(max_abs_diff(res3$video_text_embedding, fx$c_video_emb3) < 1e-4)

# Key mapper
expect_equal(
  ltx23_map_connector_key("model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.q_norm.weight"),
  "video_connector.transformer_blocks.0.attn1.norm_q.weight"
)
expect_equal(
  ltx23_map_connector_key("text_embedding_projection.video_aggregate_embed.weight"),
  "video_text_proj_in.weight"
)
expect_equal(
  ltx23_map_connector_key("text_embedding_projection.audio_aggregate_embed.bias"),
  "audio_text_proj_in.bias"
)
expect_equal(
  ltx23_map_connector_key("model.diffusion_model.audio_embeddings_connector.learnable_registers"),
  "audio_connector.learnable_registers"
)
