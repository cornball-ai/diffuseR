# Tests for LTX-2 Video Generation Pipeline
# Note: Full integration tests require model weights

# Test 1: Pipeline function exists and has expected signature
cat("Test 1: txt2vid_ltx2 function signature\n")
expect_true(is.function(txt2vid_ltx2), info = "txt2vid_ltx2 should be a function")

# Check key parameters exist
params <- names(formals(txt2vid_ltx2))
expect_true("prompt" %in% params, info = "Should have prompt param")
expect_true("width" %in% params, info = "Should have width param")
expect_true("height" %in% params, info = "Should have height param")
expect_true("num_frames" %in% params, info = "Should have num_frames param")
expect_true("memory_profile" %in% params, info = "Should have memory_profile param")
expect_true("text_backend" %in% params, info = "Should have text_backend param")

# Test 2: Default parameters are sensible
cat("Test 2: Default parameters\n")
defaults <- formals(txt2vid_ltx2)
expect_equal(defaults$width, 768L, info = "Default width should be 768")
expect_equal(defaults$height, 512L, info = "Default height should be 512")
expect_equal(defaults$num_frames, 121L, info = "Default frames should be 121")
expect_equal(defaults$num_inference_steps, 8L, info = "Default steps should be 8 (distilled)")
expect_equal(defaults$guidance_scale, 1.0, info = "Default CFG should be 1.0 (distilled mode)")

# Test 3: Memory profile resolution
cat("Test 3: Memory profile parameter\n")
profile_str <- ltx2_memory_profile(vram_gb = 8)
expect_equal(profile_str$name, "low", info = "8GB should resolve to low profile")

# Test 4: Latent dimension calculation
cat("Test 4: Latent dimensions\n")
# LTX-2 uses 32x spatial and 8x temporal compression
width <- 768L
height <- 512L
num_frames <- 121L

latent_width <- width %/% 32L
latent_height <- height %/% 32L
latent_frames <- (num_frames - 1L) %/% 8L + 1L

expect_equal(latent_width, 24L, info = "Latent width correct")
expect_equal(latent_height, 16L, info = "Latent height correct")
expect_equal(latent_frames, 16L, info = "Latent frames correct (121 -> 16)")

cat("\nLTX-2 pipeline tests completed\n")
