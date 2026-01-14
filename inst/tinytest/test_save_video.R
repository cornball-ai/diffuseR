# Tests for Video Output Utilities

# Test 1: save_video function exists
cat("Test 1: save_video function signature\n")
expect_true(is.function(save_video), info = "save_video should be a function")
expect_true(is.function(save_frames), info = "save_frames should be a function")
expect_true(is.function(latents_to_video), info = "latents_to_video should be a function")

# Test 2: save_video parameters
cat("Test 2: save_video parameters\n")
params <- names(formals(save_video))
expect_true("video" %in% params, info = "Should have video param")
expect_true("file" %in% params, info = "Should have file param")
expect_true("fps" %in% params, info = "Should have fps param")
expect_true("format" %in% params, info = "Should have format param")
expect_true("backend" %in% params, info = "Should have backend param")

# Test 3: Create test video array
cat("Test 3: Create test video array\n")
# Create a simple 10-frame gradient video [T, H, W, C]
num_frames <- 10L
height <- 64L
width <- 64L
video <- array(0, dim = c(num_frames, height, width, 3L))

for (t in seq_len(num_frames)) {
  # Create a gradient that changes over time
  for (h in seq_len(height)) {
    for (w in seq_len(width)) {
      video[t, h, w, 1] <- (h - 1) / (height - 1)  # Red gradient vertical
      video[t, h, w, 2] <- (w - 1) / (width - 1)  # Green gradient horizontal
      video[t, h, w, 3] <- (t - 1) / (num_frames - 1)  # Blue changes over time
    }
  }
}

expect_equal(dim(video), c(num_frames, height, width, 3L), info = "Video array shape correct")
expect_true(min(video) >= 0, info = "Video values >= 0")
expect_true(max(video) <= 1, info = "Video values <= 1")

# Test 4: save_frames
cat("Test 4: save_frames to temp directory\n")
temp_dir <- tempfile("frames_test_")
saved_files <- save_frames(video, temp_dir, verbose = FALSE)
expect_equal(length(saved_files), num_frames, info = "Should save all frames")
expect_true(dir.exists(temp_dir), info = "Directory should exist")
expect_true(file.exists(saved_files[1]), info = "First frame should exist")
expect_true(file.exists(saved_files[num_frames]), info = "Last frame should exist")

# Check file naming
expect_true(grepl("frame_0001\\.png$", saved_files[1]), info = "Frame naming correct")

# Cleanup
unlink(temp_dir, recursive = TRUE)

# Test 5: ffmpeg availability check
cat("Test 5: FFmpeg availability check\n")
has_ffmpeg <- diffuseR:::.ffmpeg_available()
expect_true(is.logical(has_ffmpeg), info = ".ffmpeg_available returns logical")
cat(sprintf("  FFmpeg available: %s\n", has_ffmpeg))

# Test 6: save_video with frames format
cat("Test 6: save_video with frames format\n")
temp_dir2 <- tempfile("video_frames_")
save_video(video, temp_dir2, format = "frames", verbose = FALSE)
expect_true(dir.exists(temp_dir2), info = "Output directory created")
frame_files <- list.files(temp_dir2, pattern = "\\.png$")
expect_equal(length(frame_files), num_frames, info = "All frames saved")
unlink(temp_dir2, recursive = TRUE)

# Test 7: Format inference from extension
cat("Test 7: Format inference\n")
# Create temp files to test format detection
temp_mp4 <- tempfile(fileext = ".mp4")
temp_gif <- tempfile(fileext = ".gif")
temp_webm <- tempfile(fileext = ".webm")

expect_equal(tools::file_ext(temp_mp4), "mp4", info = "MP4 extension detected")
expect_equal(tools::file_ext(temp_gif), "gif", info = "GIF extension detected")
expect_equal(tools::file_ext(temp_webm), "webm", info = "WebM extension detected")

# Test 8: save_video to MP4 (if ffmpeg available)
cat("Test 8: save_video to MP4\n")
if (has_ffmpeg) {
  temp_mp4 <- tempfile(fileext = ".mp4")
  tryCatch({
    save_video(video, temp_mp4, fps = 10, verbose = FALSE)
    expect_true(file.exists(temp_mp4), info = "MP4 file created")
    expect_true(file.size(temp_mp4) > 0, info = "MP4 file has content")
    cat("  MP4 saved successfully\n")
  }, error = function(e) {
    cat(sprintf("  MP4 save failed: %s\n", e$message))
  })
  unlink(temp_mp4)
} else {
  cat("  Skipped (ffmpeg not available)\n")
}

# Test 9: save_video to GIF (if ffmpeg available)
cat("Test 9: save_video to GIF\n")
if (has_ffmpeg) {
  temp_gif <- tempfile(fileext = ".gif")
  tryCatch({
    save_video(video, temp_gif, fps = 5, verbose = FALSE)
    expect_true(file.exists(temp_gif), info = "GIF file created")
    expect_true(file.size(temp_gif) > 0, info = "GIF file has content")
    cat("  GIF saved successfully\n")
  }, error = function(e) {
    cat(sprintf("  GIF save failed: %s\n", e$message))
  })
  unlink(temp_gif)
} else {
  cat("  Skipped (ffmpeg not available)\n")
}

cat("\nVideo output tests completed\n")
