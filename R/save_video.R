#' Save Video to File
#'
#' Saves a video array to a file in various formats.
#'
#' @param video Array of video frames with shape [T, H, W, C] where C is 3 (RGB).
#'   Values should be in [0, 1] range.
#' @param file Character. Output file path. Extension determines format.
#' @param fps Numeric. Frames per second (default 24).
#' @param format Character. Output format: "mp4", "gif", "webm", or "frames".
#'   If NULL, inferred from file extension.
#' @param backend Character. Backend to use: "ffmpeg", "av", or "auto".
#' @param quality Integer. Quality level 1-100 (for lossy formats).
#' @param verbose Logical. Print progress messages.
#'
#' @return Invisibly returns the output file path.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Save as MP4
#' save_video(video_array, "output.mp4", fps = 24)
#'
#' # Save as GIF
#' save_video(video_array, "output.gif", fps = 10)
#'
#' # Save as individual frames
#' save_video(video_array, "frames/", format = "frames")
#' }
save_video <- function(
  video,
  file,
  fps = 24,
  format = NULL,
  backend = "auto",
  quality = 85,
  verbose = TRUE
) {
  # Validate input
  if (!is.array(video)) {
    stop("video must be an array")
  }

  dims <- dim(video)
  if (length(dims) != 4 || dims[4] != 3) {
    stop("video must have shape [T, H, W, 3] (RGB)")
  }

  num_frames <- dims[1]
  height <- dims[2]
  width <- dims[3]

  if (verbose) {
    message(sprintf("Video: %d frames, %dx%d", num_frames, width, height))
  }

  # Infer format from extension if not specified
  if (is.null(format)) {
    ext <- tolower(tools::file_ext(file))
    format <- switch(ext,
      "mp4" = "mp4",
      "gif" = "gif",
      "webm" = "webm",
      "png" = "frames",
      "frames"# default to frames for directories
    )
    if (ext == "" && dir.exists(dirname(file))) {
      format <- "frames"
    }
  }

  # Clamp values to [0, 1]
  video <- pmax(pmin(video, 1), 0)

  # Dispatch to backend
  if (format == "frames") {
    save_frames(video, file, verbose = verbose)
  } else if (backend == "auto") {
    # Try ffmpeg first, fall back to av
    if (.ffmpeg_available()) {
      save_video_ffmpeg(video, file, fps = fps, format = format,
        quality = quality, verbose = verbose)
    } else if (requireNamespace("av", quietly = TRUE)) {
      save_video_av(video, file, fps = fps, verbose = verbose)
    } else {
      stop("No video backend available. Install ffmpeg or the 'av' R package.")
    }
  } else if (backend == "ffmpeg") {
    save_video_ffmpeg(video, file, fps = fps, format = format,
      quality = quality, verbose = verbose)
  } else if (backend == "av") {
    save_video_av(video, file, fps = fps, verbose = verbose)
  } else {
    stop("Unknown backend: ", backend)
  }

  invisible(file)
}

#' Save Video Frames as Individual Images
#'
#' @param video Array of video frames [T, H, W, C].
#' @param dir Directory to save frames in.
#' @param prefix Character. Filename prefix (default "frame_").
#' @param format Character. Image format: "png" or "jpg".
#' @param verbose Logical.
#'
#' @return Invisibly returns vector of saved file paths.
#'
#' @export
save_frames <- function(
  video,
  dir,
  prefix = "frame_",
  format = "png",
  verbose = TRUE
) {
  # Create directory if needed
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }

  num_frames <- dim(video)[1]
  files <- character(num_frames)

  for (i in seq_len(num_frames)) {
    frame <- video[i,,,]
    filename <- file.path(dir, sprintf("%s%04d.%s", prefix, i, format))

    # Convert to 8-bit integer array
    frame_int <- as.integer(frame * 255)
    dim(frame_int) <- dim(frame)

    # Save using png or jpeg
    if (format == "png") {
      png::writePNG(frame_int / 255, filename)
    } else {
      jpeg::writeJPEG(frame_int / 255, filename)
    }

    files[i] <- filename
  }

  if (verbose) {
    message(sprintf("Saved %d frames to %s", num_frames, dir))
  }

  invisible(files)
}

#' Save Video using FFmpeg
#'
#' @param video Array of video frames [T, H, W, C].
#' @param file Output file path.
#' @param fps Frames per second.
#' @param format Output format.
#' @param quality Quality level 1-100.
#' @param verbose Logical.
#'
#' @keywords internal
save_video_ffmpeg <- function(
  video,
  file,
  fps = 24,
  format = "mp4",
  quality = 85,
  verbose = TRUE
) {
  if (!.ffmpeg_available()) {
    stop("ffmpeg not found in PATH")
  }

  # Create temp directory for frames
  temp_dir <- tempfile("frames_")
  dir.create(temp_dir)
  on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE)

  # Save frames as PNG
  if (verbose) message("Writing frames...")
  num_frames <- dim(video)[1]

  for (i in seq_len(num_frames)) {
    frame <- video[i,,,]
    filename <- file.path(temp_dir, sprintf("frame_%04d.png", i))
    png::writePNG(frame, filename)
  }

  # Build ffmpeg command
  input_pattern <- file.path(temp_dir, "frame_%04d.png")

  # Quality mapping (CRF for H.264: 0-51, lower is better)
  crf <- as.integer(51 - (quality / 100) * 51)

  if (format == "mp4") {
    cmd <- sprintf(
      'ffmpeg -y -framerate %d -i "%s" -c:v libx264 -crf %d -pix_fmt yuv420p "%s"',
      fps, input_pattern, crf, file
    )
  } else if (format == "webm") {
    cmd <- sprintf(
      'ffmpeg -y -framerate %d -i "%s" -c:v libvpx-vp9 -crf %d -b:v 0 "%s"',
      fps, input_pattern, crf, file
    )
  } else if (format == "gif") {
    # GIF requires palette generation for quality
    palette <- file.path(temp_dir, "palette.png")
    cmd1 <- sprintf(
      'ffmpeg -y -framerate %d -i "%s" -vf "palettegen=stats_mode=diff" "%s"',
      fps, input_pattern, palette
    )
    cmd2 <- sprintf(
      'ffmpeg -y -framerate %d -i "%s" -i "%s" -lavfi "paletteuse=dither=bayer" "%s"',
      fps, input_pattern, palette, file
    )
    if (verbose) message("Generating palette...")
    system(cmd1, ignore.stdout = !verbose, ignore.stderr = !verbose)
    cmd <- cmd2
  } else {
    stop("Unsupported format for ffmpeg: ", format)
  }

  if (verbose) message(sprintf("Encoding %s...", format))
  result <- system(cmd, ignore.stdout = !verbose, ignore.stderr = !verbose)

  if (result != 0) {
    stop("ffmpeg encoding failed")
  }

  if (verbose) message(sprintf("Saved: %s", file))
}

#' Save Video using av Package
#'
#' @param video Array of video frames [T, H, W, C].
#' @param file Output file path.
#' @param fps Frames per second.
#' @param verbose Logical.
#'
#' @keywords internal
save_video_av <- function(
  video,
  file,
  fps = 24,
  verbose = TRUE
) {
  if (!requireNamespace("av", quietly = TRUE)) {
    stop("Package 'av' is required for this backend")
  }

  num_frames <- dim(video)[1]
  height <- dim(video)[2]
  width <- dim(video)[3]

  if (verbose) message("Encoding video with av...")

  # av::av_encode_video expects a function that returns frames
  # or a matrix/array. We need to convert our [T,H,W,C] to what av expects.

  # Create temp directory for frames
  temp_dir <- tempfile("frames_")
  dir.create(temp_dir)
  on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE)

  frame_files <- character(num_frames)
  for (i in seq_len(num_frames)) {
    frame <- video[i,,,]
    filename <- file.path(temp_dir, sprintf("frame_%04d.png", i))
    png::writePNG(frame, filename)
    frame_files[i] <- filename
  }

  # Use av to encode from image files
  av::av_encode_video(
    input = frame_files,
    output = file,
    framerate = fps,
    verbose = verbose
  )

  if (verbose) message(sprintf("Saved: %s", file))
}

#' Check if FFmpeg is Available
#'
#' @return Logical. TRUE if ffmpeg is in PATH.
#' @keywords internal
.ffmpeg_available <- function() {
  result <- tryCatch(
    system2("ffmpeg", "-version", stdout = FALSE, stderr = FALSE),
    error = function(e) 1
  )
  result == 0
}

#' Create Video from Latents (Helper)
#'
#' Convenience function to decode latents and save video in one step.
#'
#' @param latents Tensor of latents from generation.
#' @param vae VAE decoder module.
#' @param file Output file path.
#' @param fps Frames per second.
#' @param ... Additional arguments to save_video.
#'
#' @return Invisibly returns the output file path.
#'
#' @export
latents_to_video <- function(
  latents,
  vae,
  file,
  fps = 24,
  ...
) {
  torch::with_no_grad({
      # Decode latents
      video_tensor <- vae$decode(latents)

      # Convert to array [T, H, W, C]
      video_array <- video_tensor$squeeze(1L) $permute(c(2, 3, 4, 1)) $cpu() $numpy()

      # Clamp to [0, 1]
      video_array <- pmax(pmin(video_array, 1), 0)
    })

  # Save video
  save_video(video_array, file, fps = fps, ...)
}

