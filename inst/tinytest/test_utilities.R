# Test utility functions

# --- filename_from_prompt ---

# Test basic filename generation without datetime
result <- filename_from_prompt("test prompt", datetime = FALSE)
expect_equal(result, "prompt_test_prompt.png")

# Test special characters are replaced with underscores
result <- filename_from_prompt("hello! world?", datetime = FALSE)
expect_equal(result, "prompt_hello__world_.png")

# Test prompt truncation at 50 characters
long_prompt <- paste(rep("a", 100), collapse = "")
result <- filename_from_prompt(long_prompt, datetime = FALSE)
expect_true(nchar(result) <= 62)  # "prompt_" (7) + 50 + ".png" (4)

# Test with datetime (just check format, not exact value)
result <- filename_from_prompt("test", datetime = TRUE)
expect_true(grepl("^\\d{8}_\\d{6}_test\\.png$", result))

# --- get_required_components (internal function) ---

# Test sd21 components
components <- diffuseR:::get_required_components("sd21")
expect_true(is.character(components))
expect_true("unet" %in% components)
expect_true("decoder" %in% components)
expect_true("text_encoder" %in% components)

# Test sdxl components
components <- diffuseR:::get_required_components("sdxl")
expect_true("text_encoder2" %in% components)

# Test invalid model
expect_error(diffuseR:::get_required_components("invalid_model"))

# --- standardize_devices (internal function) ---

# Test single device string expands to all components
required <- c("unet", "decoder", "text_encoder")
result <- diffuseR:::standardize_devices("cpu", required)
expect_true(is.list(result))
expect_equal(result$unet, "cpu")
expect_equal(result$decoder, "cpu")
expect_equal(result$text_encoder, "cpu")

# Test list passthrough
devices <- list(unet = "cuda", decoder = "cpu", text_encoder = "cpu")
result <- diffuseR:::standardize_devices(devices, required)
expect_equal(result$unet, "cuda")

# Test text_encoder2 inherits from text_encoder
devices <- list(unet = "cuda", decoder = "cpu", text_encoder = "cpu", encoder = "cpu")
required_xl <- c("unet", "decoder", "text_encoder", "text_encoder2", "encoder")
result <- diffuseR:::standardize_devices(devices, required_xl)
expect_equal(result$text_encoder2, "cpu")

# Test encoder inherits from decoder
devices <- list(unet = "cuda", decoder = "cpu", text_encoder = "cpu")
required_enc <- c("unet", "decoder", "text_encoder", "encoder")
result <- diffuseR:::standardize_devices(devices, required_enc)
expect_equal(result$encoder, "cpu")
