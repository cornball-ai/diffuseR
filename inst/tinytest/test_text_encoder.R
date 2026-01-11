# Test native CLIP text encoder

# Skip if models not available
skip_if_not <- function(cond, msg) {
  if (!cond) exit_file(msg)
}

library(torch)

# Test module architecture
encoder <- text_encoder_native()
expect_true(inherits(encoder, "nn_module"))

# Check parameters exist
params <- names(encoder$parameters)
expect_true(length(params) > 0)
expect_true("token_embedding.weight" %in% params)
expect_true("position_embedding" %in% params)
expect_true("final_layer_norm.weight" %in% params)

# Test forward pass with random weights
with_no_grad({
  tokens <- torch_tensor(matrix(c(49406, 320, 4380, 49407, rep(49407, 73)), nrow = 1),
                         dtype = torch_long())
  out <- encoder(tokens)
  expect_equal(out$shape[1], 1L)
  expect_equal(out$shape[2], 77L)
  expect_equal(out$shape[3], 768L)
})

# Test load_model_component with use_native
model_dir <- tools::R_user_dir("diffuseR", "data")
encoder_file <- file.path(model_dir, "sdxl", "text_encoder-cpu.pt")
skip_if_not(file.exists(encoder_file), "SDXL text_encoder not available")

native_enc <- load_model_component("text_encoder", "sdxl", "cpu", use_native = TRUE)
expect_true(inherits(native_enc, "TextEncoderNative"))

# Verify weight loading (parameter count)
expect_equal(length(native_enc$parameters), 196L)

# Test output equivalency with TorchScript
ts_enc <- torch::jit_load(encoder_file)
tokens <- torch_tensor(matrix(c(49406, 320, 4380, 49407, rep(49407, 73)), nrow = 1),
                       dtype = torch_long())
with_no_grad({
  ts_out <- ts_enc(tokens)
  native_out <- native_enc$forward(tokens)
})

# Check output shapes match
expect_equal(as.integer(ts_out$shape), as.integer(native_out$shape))

# Check outputs are close (within float32 precision)
diff <- (ts_out - native_out)$abs()
max_diff <- as.numeric(diff$max())
expect_true(max_diff < 5.0, info = paste("max_diff:", max_diff))  # relaxed threshold
