# Test native UNet

# Skip if models not available
skip_if_not <- function(cond, msg) {
  if (!cond) exit_file(msg)
}

library(diffuseR)
library(torch)

# Test module architecture
unet <- unet_native(
  in_channels = 4L,
  out_channels = 4L,
  block_out_channels = c(320L, 640L, 1280L, 1280L),
  layers_per_block = 2L,
  cross_attention_dim = 1024L,
  attention_head_dim = 64L
)
expect_true(inherits(unet, "nn_module"))

# Check parameters exist
params <- names(unet$parameters)
expect_true(length(params) > 0)
expect_true("conv_in.weight" %in% params)
expect_true("conv_out.weight" %in% params)
expect_true("time_embedding_linear_1.weight" %in% params)

# Test forward pass with random weights
with_no_grad({
  sample <- torch_randn(c(1L, 4L, 32L, 32L))
  timestep <- torch_tensor(c(500L))
  encoder_hidden_states <- torch_randn(c(1L, 77L, 1024L))
  out <- unet(sample, timestep, encoder_hidden_states)
  expect_equal(out$shape[1], 1L)
  expect_equal(out$shape[2], 4L)
  expect_equal(out$shape[3], 32L)
  expect_equal(out$shape[4], 32L)
})

# Test load_model_component with use_native
model_dir <- tools::R_user_dir("diffuseR", "data")
unet_file <- file.path(model_dir, "sd21", "unet-cpu.pt")
skip_if_not(file.exists(unet_file), "SD21 unet not available")

native_unet <- load_model_component("unet", "sd21", "cpu", use_native = TRUE)
expect_true(inherits(native_unet, "UNetNative"))

# Verify weight loading (parameter count)
expect_equal(length(native_unet$parameters), 686L)

# Test output equivalency with TorchScript (deterministic inputs)
ts_unet <- torch::jit_load(unet_file)

# Use deterministic inputs for consistent comparison
sample <- torch_ones(c(1L, 4L, 32L, 32L)) * 0.1
timestep <- torch_tensor(c(500L))
encoder_hidden_states <- torch_ones(c(1L, 77L, 1024L)) * 0.1

with_no_grad({
  ts_out <- ts_unet(sample, timestep, encoder_hidden_states)
  native_out <- native_unet$forward(sample, timestep, encoder_hidden_states)
})

# Check output shapes match
expect_equal(as.integer(ts_out$shape), as.integer(native_out$shape))

# Check outputs are close (within float32 precision for deep attention models)
diff <- (ts_out - native_out)$abs()
max_diff <- as.numeric(diff$max())
mean_diff <- as.numeric(diff$mean())
expect_true(max_diff < 0.1, info = paste("max_diff:", max_diff))  # deterministic inputs
expect_true(mean_diff < 0.02, info = paste("mean_diff:", mean_diff))
