# SD 2.1 .pt -> diffusers converter: CLIPTextConfig derivation from the
# text_encoder parameters, and the missing-component error. The full
# conversion (bit-identical to the source) is validated out of band
# against the cached .pt weights.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
library(diffuseR)

# --- config derivation from mock text_encoder params (small dims) ------------------

z <- function(...) torch::torch_zeros(c(...))
p <- list()
p[["text_encoder.text_model.embeddings.token_embedding.weight"]] <- z(100L, 128L)
p[["text_encoder.text_model.embeddings.position_embedding.weight"]] <- z(77L, 128L)
for (i in 0:3) {
  p[[sprintf("text_encoder.text_model.encoder.layers.%d.mlp.fc1.weight", i)]] <-
    z(512L, 128L)
}
cfg <- diffuseR:::.sd21_write_clip_config(p, tempfile(fileext = ".json"))
expect_equal(cfg$vocab_size, 100L)
expect_equal(cfg$hidden_size, 128L)
expect_equal(cfg$max_position_embeddings, 77L)
expect_equal(cfg$num_hidden_layers, 4L)          # distinct layer indices
expect_equal(cfg$num_attention_heads, 2L)        # hidden / 64
expect_equal(cfg$intermediate_size, 512L)

# --- missing TorchScript components -> actionable error ---------------------------

expect_error(convert_sd21_pt_to_diffusers(pt_dir = tempfile()),
  pattern = "Missing")
