# Hidden-state collection semantics of the Gemma3 text encoder.
#
# HF transformers returns num_layers + 1 hidden states: the state BEFORE
# each decoder layer (the first being the embedding output) plus the
# post-final-norm output. The un-normed last-layer output is never
# included. The LTX connectors' input projection expects exactly
# hidden_size * (num_layers + 1) features, so an extra or missing state
# fails at the first matmul with real weights.

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}

library(diffuseR)

config <- gemma3_config_ltx2()
config$vocab_size <- 64L
config$hidden_size <- 16L
config$intermediate_size <- 32L
config$num_hidden_layers <- 3L
config$num_attention_heads <- 2L
config$num_key_value_heads <- 1L
config$head_dim <- 8L
config$query_pre_attn_scalar <- 8L
config$sliding_window <- 4L

model <- gemma3_text_model(config)
model$eval()

ids <- torch::torch_randint(1L, 64L, size = c(1L, 7L), dtype = torch::torch_long())
mask <- torch::torch_ones(1L, 7L)
torch::with_no_grad({
  out <- model(ids, attention_mask = mask, output_hidden_states = TRUE)
})

# num_layers + 1 states
expect_equal(length(out$hidden_states), config$num_hidden_layers + 1L)

# Last collected state is the post-norm output (identical to last_hidden_state)
last_collected <- out$hidden_states[[length(out$hidden_states)]]
expect_true(as.numeric((last_collected - out$last_hidden_state)$abs()$max()) == 0)

# The un-normed last-layer output must NOT be in the stack: the
# second-to-last entry is the input to the final layer, so running the
# final layer + norm over it must reproduce last_hidden_state.
pre_last <- out$hidden_states[[config$num_hidden_layers]]
expect_false(as.numeric((pre_last - out$last_hidden_state)$abs()$max()) == 0)

# Without the flag, no states are collected
torch::with_no_grad({
  out2 <- model(ids, attention_mask = mask, output_hidden_states = FALSE)
})
expect_null(out2$hidden_states)
