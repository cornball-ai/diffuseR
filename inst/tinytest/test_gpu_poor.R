# Tests for GPU-Poor Memory Management

# Test 1: Memory profile detection
cat("Test 1: ltx2_memory_profile with explicit VRAM\n")
profile_high <- ltx2_memory_profile(vram_gb = 20)
expect_equal(profile_high$name, "high", info = "20GB should be high profile")

profile_medium <- ltx2_memory_profile(vram_gb = 12)
expect_equal(profile_medium$name, "medium", info = "12GB should be medium profile")

profile_low <- ltx2_memory_profile(vram_gb = 8)
expect_equal(profile_low$name, "low", info = "8GB should be low profile")

profile_very_low <- ltx2_memory_profile(vram_gb = 6)
expect_equal(profile_very_low$name, "very_low", info = "6GB should be very_low profile")

profile_cpu <- ltx2_memory_profile(vram_gb = 2)
expect_equal(profile_cpu$name, "cpu_only", info = "2GB should be cpu_only profile")

# Test 2: Profile contains required fields
cat("Test 2: Profile structure\n")
profile <- ltx2_memory_profile(vram_gb = 8)
expect_true("dit_device" %in% names(profile), info = "Profile should have dit_device")
expect_true("vae_device" %in% names(profile), info = "Profile should have vae_device")
expect_true("vae_tiling" %in% names(profile), info = "Profile should have vae_tiling")
expect_true("vae_tile_size" %in% names(profile), info = "Profile should have vae_tile_size")
expect_true("max_resolution" %in% names(profile), info = "Profile should have max_resolution")
expect_true("max_frames" %in% names(profile), info = "Profile should have max_frames")
expect_true("cfg_mode" %in% names(profile), info = "Profile should have cfg_mode")

# Test 3: Low profile enables tiling and offloading
cat("Test 3: Low profile settings\n")
low <- ltx2_memory_profile(vram_gb = 8)
expect_true(low$dit_offload, info = "Low profile should enable DiT offload")
expect_true(low$vae_tiling, info = "Low profile should enable VAE tiling")
expect_equal(low$cfg_mode, "sequential", info = "Low profile should use sequential CFG")

# Test 4: High profile disables aggressive optimizations
cat("Test 4: High profile settings\n")
high <- ltx2_memory_profile(vram_gb = 20)
expect_false(high$dit_offload, info = "High profile should not offload DiT")
expect_false(high$vae_tiling, info = "High profile should not tile VAE")
expect_equal(high$cfg_mode, "batched", info = "High profile should use batched CFG")

# Test 5: Resolution validation
cat("Test 5: validate_resolution\n")
profile <- ltx2_memory_profile(vram_gb = 8)
result <- validate_resolution(1080, 1920, 121, profile)
expect_true(result$adjusted, info = "Should adjust 1080p for low profile")
expect_true(result$height <= profile$max_resolution[1], info = "Height should be capped")
expect_true(result$width <= profile$max_resolution[2], info = "Width should be capped")
expect_true(result$num_frames <= profile$max_frames, info = "Frames should be capped")

# Test 6: Resolution within limits
cat("Test 6: validate_resolution within limits\n")
result_ok <- validate_resolution(480, 640, 30, profile)
expect_false(result_ok$adjusted, info = "Should not adjust if within limits")

# Test 7: VRAM report (just test it runs)
cat("Test 7: vram_report\n")
result <- vram_report("test")
expect_true(is.list(result), info = "vram_report should return a list")
expect_true("used" %in% names(result), info = "Result should have used")
expect_true("free" %in% names(result), info = "Result should have free")

# Test 8: clear_vram (just test it runs)
cat("Test 8: clear_vram\n")
clear_vram()
expect_true(TRUE, info = "clear_vram should run without error")

# Test 9: is_blackwell_gpu returns logical
cat("Test 9: is_blackwell_gpu\n")
is_bb <- is_blackwell_gpu()
expect_true(is.logical(is_bb), info = "is_blackwell_gpu should return logical")

# Test 10: .detect_vram returns numeric
cat("Test 10: .detect_vram\n")
vram <- diffuseR:::.detect_vram()
expect_true(is.numeric(vram), info = ".detect_vram should return numeric")
expect_true(vram >= 0, info = "VRAM should be non-negative")

cat("\nAll GPU-poor tests completed\n")
