# Tests for LTX2 weight loading functions

# Test 1: load_ltx2_vae function exists
cat("Test 1: load_ltx2_vae function exists\n")
expect_true(is.function(load_ltx2_vae), info = "load_ltx2_vae should be a function")

# Test 2: load_ltx2_transformer function exists
cat("Test 2: load_ltx2_transformer function exists\n")
expect_true(is.function(load_ltx2_transformer), info = "load_ltx2_transformer should be a function")

# Test 3: load_ltx2_connectors function exists
cat("Test 3: load_ltx2_connectors function exists\n")
expect_true(is.function(load_ltx2_connectors), info = "load_ltx2_connectors should be a function")

# Test 4: VAE module can be created with defaults
cat("Test 4: VAE module creation\n")
vae <- ltx2_video_vae()
expect_true(inherits(vae, "nn_module"), info = "VAE should be nn_module")

# Test 5: VAE has expected structure
cat("Test 5: VAE structure\n")
expect_true("encoder" %in% names(vae$children), info = "VAE should have encoder")
expect_true("decoder" %in% names(vae$children), info = "VAE should have decoder")

# Test 6: VAE parameter count is reasonable
cat("Test 6: VAE parameter count\n")
vae_params <- names(vae$parameters)
expect_true(length(vae_params) > 100,
            info = sprintf("VAE should have many parameters (got %d)", length(vae_params)))

# Test 7: Transformer module can be created
cat("Test 7: Transformer module creation\n")
# Create with smaller config for testing (less memory)
transformer <- ltx2_video_transformer_3d_model(
  num_layers = 2L,  # Minimal layers for testing
  num_attention_heads = 4L,
  attention_head_dim = 32L,
  audio_num_attention_heads = 4L,
  audio_attention_head_dim = 32L
)
expect_true(inherits(transformer, "nn_module"), info = "Transformer should be nn_module")

# Test 8: Transformer has expected structure
cat("Test 8: Transformer structure\n")
expect_true("transformer_blocks" %in% names(transformer$children),
            info = "Transformer should have transformer_blocks")
expect_true("proj_in" %in% names(transformer$children),
            info = "Transformer should have proj_in")
expect_true("proj_out" %in% names(transformer$children),
            info = "Transformer should have proj_out")

# Test 9: Connectors module can be created
cat("Test 9: Connectors module creation\n")
connectors <- ltx2_text_connectors(
  video_connector_num_attention_heads = 4L,
  video_connector_attention_head_dim = 32L,
  audio_connector_num_attention_heads = 4L,
  audio_connector_attention_head_dim = 32L
)
expect_true(inherits(connectors, "nn_module"), info = "Connectors should be nn_module")

# Test 10: Connectors has expected structure
cat("Test 10: Connectors structure\n")
expect_true("video_connector" %in% names(connectors$children),
            info = "Connectors should have video_connector")
expect_true("audio_connector" %in% names(connectors$children),
            info = "Connectors should have audio_connector")
expect_true("text_proj_in" %in% names(connectors$children),
            info = "Connectors should have text_proj_in")

# Test 11: load_ltx2_vae errors on missing file
cat("Test 11: load_ltx2_vae error handling\n")
expect_error(load_ltx2_vae("/nonexistent/path.safetensors"),
             info = "Should error on missing file")

# Test 12: load_ltx2_transformer errors on missing directory
cat("Test 12: load_ltx2_transformer error handling\n")
expect_error(load_ltx2_transformer("/nonexistent/dir"),
             info = "Should error on missing directory")

# Test 13: load_ltx2_connectors errors on missing file
cat("Test 13: load_ltx2_connectors error handling\n")
expect_error(load_ltx2_connectors("/nonexistent/path.safetensors"),
             info = "Should error on missing file")

# Test 14: Internal weight loading function exists for VAE
cat("Test 14: Internal VAE weight loading function\n")
expect_true(exists("load_ltx2_vae_weights", envir = asNamespace("diffuseR")),
            info = "load_ltx2_vae_weights should exist")

# Test 15: Internal weight loading function exists for transformer
cat("Test 15: Internal transformer weight loading function\n")
expect_true(exists("load_ltx2_transformer_weights", envir = asNamespace("diffuseR")),
            info = "load_ltx2_transformer_weights should exist")

# Test 16: Internal weight loading function exists for connectors
cat("Test 16: Internal connector weight loading function\n")
expect_true(exists("load_ltx2_connector_weights", envir = asNamespace("diffuseR")),
            info = "load_ltx2_connector_weights should exist")

cat("\nLTX2 weight loading tests completed\n")
