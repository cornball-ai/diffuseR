# Tests for SDXL GPU-Poor Memory Profiles

# Test 1: sdxl_memory_profile exists
cat("Test 1: sdxl_memory_profile function\n")
expect_true(is.function(sdxl_memory_profile), info = "sdxl_memory_profile should be a function")

# Test 2: Profile detection by VRAM
cat("Test 2: Profile detection by VRAM\n")
profile_full <- sdxl_memory_profile(vram_gb = 20)
expect_equal(profile_full$name, "full_gpu", info = "20GB should be full_gpu")

profile_balanced <- sdxl_memory_profile(vram_gb = 12)
expect_equal(profile_balanced$name, "balanced", info = "12GB should be balanced")

profile_unet <- sdxl_memory_profile(vram_gb = 8)
expect_equal(profile_unet$name, "unet_gpu", info = "8GB should be unet_gpu")

profile_cpu <- sdxl_memory_profile(vram_gb = 4)
expect_equal(profile_cpu$name, "cpu_only", info = "4GB should be cpu_only")

# Test 3: Profile structure
cat("Test 3: Profile structure\n")
profile <- sdxl_memory_profile(vram_gb = 10)
expect_true("devices" %in% names(profile), info = "Should have devices")
expect_true("dtype" %in% names(profile), info = "Should have dtype")
expect_true("cfg_mode" %in% names(profile), info = "Should have cfg_mode")
expect_true("cleanup" %in% names(profile), info = "Should have cleanup")
expect_true("max_resolution" %in% names(profile), info = "Should have max_resolution")
expect_true("step_cleanup_interval" %in% names(profile), info = "Should have step_cleanup_interval")

# Test 4: Device configuration
cat("Test 4: Device configurations\n")
full <- sdxl_memory_profile(vram_gb = 20)
expect_equal(full$devices$unet, "cuda", info = "full_gpu: unet on cuda")
expect_equal(full$devices$decoder, "cuda", info = "full_gpu: decoder on cuda")
expect_equal(full$devices$text_encoder, "cuda", info = "full_gpu: text_encoder on cuda")

balanced <- sdxl_memory_profile(vram_gb = 12)
expect_equal(balanced$devices$unet, "cuda", info = "balanced: unet on cuda")
expect_equal(balanced$devices$decoder, "cuda", info = "balanced: decoder on cuda")
expect_equal(balanced$devices$text_encoder, "cpu", info = "balanced: text_encoder on cpu")

unet <- sdxl_memory_profile(vram_gb = 8)
expect_equal(unet$devices$unet, "cuda", info = "unet_gpu: unet on cuda")
expect_equal(unet$devices$decoder, "cpu", info = "unet_gpu: decoder on cpu")
expect_equal(unet$devices$text_encoder, "cpu", info = "unet_gpu: text_encoder on cpu")

cpu <- sdxl_memory_profile(vram_gb = 4)
expect_equal(cpu$devices$unet, "cpu", info = "cpu_only: unet on cpu")
expect_equal(cpu$devices$decoder, "cpu", info = "cpu_only: decoder on cpu")

# Test 5: CFG mode by profile
cat("Test 5: CFG mode by profile\n")
expect_equal(sdxl_memory_profile(vram_gb = 20)$cfg_mode, "batched", info = "full_gpu uses batched CFG")
expect_equal(sdxl_memory_profile(vram_gb = 8)$cfg_mode, "sequential", info = "unet_gpu uses sequential CFG")

# Test 6: Cleanup settings
cat("Test 6: Cleanup settings\n")
expect_equal(sdxl_memory_profile(vram_gb = 20)$cleanup, "none", info = "full_gpu: no cleanup")
expect_equal(sdxl_memory_profile(vram_gb = 12)$cleanup, "phase", info = "balanced: phase cleanup")
expect_equal(sdxl_memory_profile(vram_gb = 8)$cleanup, "phase", info = "unet_gpu: phase cleanup")

# Test 7: Step cleanup interval
cat("Test 7: Step cleanup interval\n")
expect_equal(sdxl_memory_profile(vram_gb = 20)$step_cleanup_interval, 0L, info = "full_gpu: no step cleanup")
expect_equal(sdxl_memory_profile(vram_gb = 8)$step_cleanup_interval, 10L, info = "unet_gpu: cleanup every 10 steps")

# Test 8: txt2img_sdxl accepts memory_profile parameter
cat("Test 8: txt2img_sdxl memory_profile parameter\n")
params <- names(formals(txt2img_sdxl))
expect_true("memory_profile" %in% params, info = "txt2img_sdxl should have memory_profile param")
expect_true("verbose" %in% params, info = "txt2img_sdxl should have verbose param")

cat("\nSDXL GPU-poor tests completed\n")
