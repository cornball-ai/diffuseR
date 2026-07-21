#' LTX-2.3 Prefix Conditioning (image-to-video, video continuation)
#'
#' Fresh R port of the frame-conditioning mechanics from the diffusers
#' reference (Apache-2.0, pipelines/ltx2/pipeline_ltx2_image2video.py
#' and pipeline_ltx2_condition.py), restricted to prefix conditioning
#' at latent index 0 with strength 1: a single start image (i2v) or the
#' leading pixel frames of a previous clip (continuation). Conditioned
#' latent tokens are initialized from the VAE-encoded pixels, see a
#' per-token timestep of zero, and are frozen through the Euler loop.
#'
#' @name condition_ltx23
NULL

#' Preprocess an image (or frame stack) for VAE encoding
#'
#' Mirrors the diffusers VideoProcessor: bilinear resize so the shorter
#' relative side matches, center-crop to the exact target, and scale to
#' [-1, 1].
#'
#' @param x Path to a PNG/JPEG, or an array [H, W, 3] (values in
#'   [0, 1]), or a [F, H, W, 3] array of frames.
#' @param height,width Integers. Target size (multiples of 32).
#'
#' @return Float32 tensor [1, 3, F, height, width] in [-1, 1].
#'
#' @export
ltx23_preprocess_frames <- function(x, height, width) {
    if (is.character(x)) {
        x <- if (grepl("\\.png$", x, ignore.case = TRUE)) {
            png::readPNG(x)
        } else {
            jpeg::readJPEG(x)
        }
    }
    if (length(dim(x)) == 3L) {
        dim(x) <- c(1L, dim(x))
    }
    if (dim(x)[4] > 3L) {
        # Drop alpha
        x <- x[,,, 1:3, drop = FALSE]
    }
    frames <- torch::torch_tensor(x, dtype = torch::torch_float32())
    # [F, H, W, C] -> [F, C, H, W]
    frames <- frames$permute(c(1L, 4L, 2L, 3L))

    src_h <- frames$shape[3]
    src_w <- frames$shape[4]
    scale <- max(height / src_h, width / src_w)
    new_h <- as.integer(round(src_h * scale))
    new_w <- as.integer(round(src_w * scale))
    frames <- torch::nnf_interpolate(frames, size = c(new_h, new_w),
                                     mode = "bilinear", align_corners = FALSE)
    top <- (new_h - height) %/% 2L
    left <- (new_w - width) %/% 2L
    frames <- frames$narrow(3L, top + 1L, height)$narrow(4L, left + 1L, width)

    # [F, C, H, W] -> [1, C, F, H, W], [0, 1] -> [-1, 1]
    frames$permute(c(2L, 1L, 3L, 4L))$unsqueeze(1L)$mul(2)$sub(1)
}

#' Read the trailing frames of a video file
#'
#' Extracts the last \code{n} frames of an MP4 (via \code{av}) for use
#' as continuation conditioning.
#'
#' @param path Video file.
#' @param n Integer. Trailing frame count.
#'
#' @return Array [n, H, W, 3] in [0, 1].
#'
#' @export
ltx23_read_tail_frames <- function(path, n = 9L) {
    if (!requireNamespace("av", quietly = TRUE)) {
        stop("Reading video frames requires the 'av' package.")
    }
    info <- av::av_media_info(path)
    total <- info$video$frames
    if (is.null(total) || is.na(total)) {
        # Some containers omit the frame count; derive from duration
        total <- floor(info$duration * info$video$framerate)
    }
    out_dir <- tempfile("ltx23_tail_")
    dir.create(out_dir)
    on.exit(unlink(out_dir, recursive = TRUE), add = TRUE)
    av::av_video_images(path, destdir = out_dir, format = "png")
    files <- sort(list.files(out_dir, pattern = "\\.png$", full.names = TRUE))
    files <- utils::tail(files, n)
    frames <- lapply(files, png::readPNG)
    arr <- array(0, dim = c(length(frames), dim(frames[[1]])[1],
                            dim(frames[[1]])[2], 3L))
    for (i in seq_along(frames)) {
        f <- frames[[i]]
        arr[i,,,] <- f[,, 1:3]
    }
    arr
}

#' Encode pixel frames to normalized video latents
#'
#' VAE encode in "argmax" mode (the distribution mean), then normalize
#' with the checkpoint's per-channel statistics — the exact inverse of
#' the decode path.
#'
#' @param vae An \code{ltx23_video_vae}.
#' @param frames Tensor [1, 3, F, H, W] in [-1, 1] (see
#'   \code{\link{ltx23_preprocess_frames}}).
#'
#' @return Normalized latents [1, 128, F', H/32, W/32] (float32).
#'
#' @export
ltx23_encode_video_frames <- function(vae, frames) {
    enc_dtype <- vae$encoder$conv_in$conv$weight$dtype
    enc_device <- vae$encoder$conv_in$conv$weight$device
    moments <- torch::with_no_grad(
                                   vae$encode(frames$to(device = enc_device, dtype = enc_dtype))
    )
    latents <- moments$mean$to(dtype = torch::torch_float32())
    ltx23_normalize_latents(latents, vae$latents_mean$to(dtype = torch::torch_float32()),
                            vae$latents_std$to(dtype = torch::torch_float32()))
}

#' Slice the trailing latent frames of a generation for chaining
#'
#' Cuts the last \code{k} latent frames out of a result's video
#' latents, in the [1, 128, k, H', W'] layout that
#' \code{txt2vid_ltx2(condition_latents = )} consumes, so one chunk
#' can seed the next without leaving latent space: no decode, no
#' re-encode, no video round-trip.
#'
#' Semantics caveat: a latent frame sliced from inside a sequence
#' represents 8 pixel frames, while a fresh VAE encode of a k-frame
#' tail represents 1 + 8(k - 1) pixel frames with its first latent in
#' first-frame form. The frozen prefix the next generation sees is
#' therefore not identical to the pixel-path prefix; compare both on
#' real content before relying on latent-only joins.
#'
#' @param result A \code{\link{txt2vid_ltx2}} result list (uses its
#'   \code{latents} and \code{latent_shape}), or the packed latents
#'   tensor [1, S, 128] itself (then \code{latent_shape} is
#'   required).
#' @param k Integer. Trailing latent frames to keep (default 2 = the
#'   standard 9-pixel-frame conditioning prefix).
#' @param latent_shape Integer vector c(frames, height, width) of the
#'   latent geometry; only needed when \code{result} is a raw tensor.
#'
#' @return Normalized latents [1, 128, k, H', W'] (float32), ready
#'   for \code{txt2vid_ltx2(condition_latents = )}.
#'
#' @export
ltx23_tail_latents <- function(result, k = 2L, latent_shape = NULL) {
    if (is.list(result)) {
        latents <- result$latents
        latent_shape <- latent_shape %||% result$latent_shape
    } else {
        latents <- result
    }
    if (is.null(latent_shape)) {
        stop("latent_shape is required when result is a raw latents tensor")
    }
    lf <- latent_shape[1]
    if (k < 1L || k > lf) {
        stop("k must be between 1 and the latent frame count")
    }
    video <- ltx23_unpack_video_latents(latents, lf, latent_shape[2],
                                        latent_shape[3])
    video$narrow(3L, lf - k + 1L, k)
}

#' Build conditioned initial latents and the conditioning mask
#'
#' i2v (\code{cond_latents} has one latent frame): the encoded frame is
#' repeated across all latent frames and only latent frame 0 is marked
#' conditioned. Continuation (k latent frames): the prefix tokens are
#' replaced and marked. Unconditioned positions start as pure noise.
#'
#' @param cond_latents Normalized condition latents
#'   [1, 128, k, H', W'] from \code{\link{ltx23_encode_video_frames}}.
#' @param latent_frames,latent_height,latent_width Integers. Full
#'   latent geometry of the generation.
#' @param noise Tensor [1, 128, F', H', W'] of standard noise (caller
#'   provides so seeding stays in one place).
#' @param cond_noise_scale Numeric. Optional partial noising of the
#'   conditioned tokens (diffusers \code{noise_scale}, default 0).
#'
#' @return list(latents [1, S, 128] float32 packed,
#'   conditioning_mask [1, S] float32 packed).
#'
#' @export
ltx23_prepare_conditioned_latents <- function(cond_latents, latent_frames,
    latent_height, latent_width,
    noise, cond_noise_scale = 0) {
    k <- cond_latents$shape[3]
    stopifnot(k <= latent_frames, cond_latents$shape[4] == latent_height,
              cond_latents$shape[5] == latent_width)

    init <- if (k == 1L) {
        cond_latents$`repeat`(c(1L, 1L, latent_frames, 1L, 1L))
    } else {
        torch::torch_cat(list(
                              cond_latents,
                              noise$narrow(3L, k + 1L, latent_frames - k)
            ), dim = 3L)
    }

    mask <- torch::torch_zeros(1L, 1L, latent_frames, latent_height,
                               latent_width, dtype = torch::torch_float32(),
                               device = noise$device)
    mask$narrow(3L, 1L, k)$fill_(1)

    latents <- init$to(device = noise$device) * mask + noise * (1 - mask)
    if (cond_noise_scale > 0) {
        # Partially re-noise the conditioned tokens (reference
        # _create_noised_state with per-token scale)
        scale <- mask * cond_noise_scale
        latents <- noise * scale + latents * (1 - scale)
    }

    list(
         latents = ltx23_pack_video_latents(latents),
         conditioning_mask = ltx23_pack_video_latents(mask)$squeeze(3L)
    )
}
