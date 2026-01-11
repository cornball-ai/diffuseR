# Test native VAE decoder

# Skip if no GPU or models not available
skip_if_not <- function(cond, msg) {
  if (!cond) exit_file(msg)
}

# Test module architecture
decoder <- vae_decoder_native()
expect_true(inherits(decoder, "nn_module"))

# Check parameters exist
params <- names(decoder$parameters)
expect_true(length(params) > 0)
expect_true("conv_in.weight" %in% params)
expect_true("conv_out.weight" %in% params)

# Test forward pass on CPU with random weights
library(torch)
with_no_grad({
  test_input <- torch_randn(c(1, 4, 8, 8))
  out <- decoder(test_input)
  expect_equal(out$shape[1], 1L)
  expect_equal(out$shape[2], 3L)  # RGB output
  expect_equal(out$shape[3], 64L)  # 8x upscaling
  expect_equal(out$shape[4], 64L)
})

# Test load_model_component with use_native
model_dir <- tools::R_user_dir("diffuseR", "data")
decoder_file <- file.path(model_dir, "sdxl", "decoder-cpu.pt")
skip_if_not(file.exists(decoder_file), "SDXL decoder not available")

native_dec <- load_model_component("decoder", "sdxl", "cpu", use_native = TRUE)
expect_true(inherits(native_dec, "VAEDecoderNative"))

# Verify weight loading (parameter count)
expect_equal(length(native_dec$parameters), 138L)
