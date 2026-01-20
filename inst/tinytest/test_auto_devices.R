# Test auto_devices function

# Test auto_devices returns correct structure for SDXL
devices <- auto_devices("sdxl", strategy = "unet_gpu")
expect_true(is.list(devices))
expect_true("unet" %in% names(devices))
expect_true("decoder" %in% names(devices))
expect_true("text_encoder" %in% names(devices))
expect_true("text_encoder2" %in% names(devices))
expect_true("encoder" %in% names(devices))

# Test unet_gpu strategy
expect_equal(devices$unet, "cuda")
expect_equal(devices$decoder, "cpu")
expect_equal(devices$text_encoder, "cpu")

# Test SD21 (no text_encoder2)
devices_sd21 <- auto_devices("sd21", strategy = "cpu_only")
expect_true("unet" %in% names(devices_sd21))
expect_false("text_encoder2" %in% names(devices_sd21))
expect_equal(devices_sd21$unet, "cpu")

# Test full_gpu strategy (may be overridden on Blackwell)
devices_full <- auto_devices("sdxl", strategy = "full_gpu")
expect_equal(devices_full$unet, "cuda")
# On Blackwell, full_gpu gets overridden to unet_gpu
if (requireNamespace("gpu.ctl", quietly = TRUE) && gpu.ctl::gpu_is_blackwell()) {
  expect_equal(devices_full$decoder, "cpu")
} else {
  expect_equal(devices_full$decoder, "cuda")
}

# Test cpu_only strategy
devices_cpu <- auto_devices("sdxl", strategy = "cpu_only")
expect_equal(devices_cpu$unet, "cpu")
expect_equal(devices_cpu$decoder, "cpu")

# Test invalid model
expect_error(auto_devices("invalid_model"))

# Test fallback function
fallback <- diffuseR:::.build_fallback_devices("sdxl", "unet_gpu")
expect_equal(fallback$unet, "cuda")
expect_equal(fallback$decoder, "cpu")
