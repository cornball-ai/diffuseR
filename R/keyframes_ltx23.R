# LTX keyframes: content pinned at any frame of a generation.
#
# Ported from diffusers `pipelines/ltx2/pipeline_ltx2_condition.py`
# (Apache-2.0; LTX2VideoCondition, apply_first_frame_conditioning,
# _prepare_keyframe_coords, prepare_latents):
#
#   frame 0  the keyframe's latent frames replace the generation's own
#            from latent 0 (image-to-video, a carried tail).
#   frame f  the keyframe's latent tokens are appended to the sequence
#            with RoPE positions starting at pixel frame f (no causal
#            first-frame fix; a single image spans [f, f + 1)); the
#            model attends to them and they are dropped before decoding.
#
# Strength s gives each token a conditioning mask s: it starts at
# noise * (1 - s) + clean * s, sees timestep (1 - s) * sigma, and after
# each step its x0 estimate is blended back toward the keyframe by s.
# Strength 1 pins the content exactly.

#' Define an LTX keyframe
#'
#' A keyframe conditions an \code{\link{txt2vid_ltx2}} generation on
#' content at a frame: a first frame, an end frame, a carried tail from
#' a previous chunk, or any latent frames in between. Pass a list of
#' them as \code{txt2vid_ltx2(keyframes = )}.
#'
#' @param x The content: normalized video latents [1, 128, k, H/32,
#'   W/32] (e.g. from \code{\link{ltx23_tail_latents}} or
#'   \code{\link{ltx23_encode_video_frames}}); an image path; an [H, W,
#'   3] image array; or an [F, H, W, 3] array of frames in [0, 1]
#'   (VAE-encoded at generation time, so F should be 8k + 1).
#' @param frame Integer. Pixel frame (0-based) the keyframe starts at.
#' @param strength Numeric in (0, 1]. 1 holds the content as given;
#'   lower values let the generation move away from it.
#'
#' @details At frame 0 the keyframe replaces the generation's first
#' latent frames and, at strength 1, decodes exactly as given there.
#' Anywhere else it is appended as extra tokens positioned at
#' \code{frame}: the generation attends to them and moves toward them,
#' so they act as a strong guide rather than a copy (at 512x512, a still
#' at frame 48 came back at about 25 dB PSNR). The last latent frame of
#' an 8k + 1 clip starts at pixel frame \code{num_frames - 8}.
#'
#' @return An \code{ltx23_keyframe} object.
#' @export
ltx23_keyframe <- function(x, frame = 0L, strength = 1) {
    frame <- as.integer(frame)
    if (length(frame) != 1L || is.na(frame) || frame < 0L) {
        stop("frame must be a single non-negative integer")
    }
    if (!is.numeric(strength) || length(strength) != 1L ||
        is.na(strength) || strength <= 0 || strength > 1) {
        stop("strength must be in (0, 1]")
    }
    ok <- inherits(x, "torch_tensor") ||
        (is.character(x) && length(x) == 1L) ||
        (is.numeric(x) && length(dim(x)) %in% c(3L, 4L))
    if (!ok) {
        stop("x must be latents, an image path, or an image/frames array")
    }
    structure(list(x = x, frame = frame, strength = as.numeric(strength)),
              class = "ltx23_keyframe")
}

#' @export
print.ltx23_keyframe <- function(x, ...) {
    what <- if (inherits(x$x, "torch_tensor")) {
        sprintf("latents [%s]", paste(x$x$shape, collapse = ", "))
    } else if (is.character(x$x)) {
        basename(x$x)
    } else {
        sprintf("array [%s]", paste(dim(x$x), collapse = ", "))
    }
    cat(sprintf("<ltx23_keyframe> %s at frame %d, strength %g\n", what,
                x$frame, x$strength))
    invisible(x)
}

# A keyframe's content as normalized latents [1, 128, k, lh, lw] on
# `device`, plus how many pixel frames it came from (NA for latents).
# `encode` is called with preprocessed [1, 3, F, H, W] frames only when
# needed.
.ltx23_keyframe_latents <- function(kf, encode, height, width, lh, lw,
                                    device) {
    x <- kf$x
    if (inherits(x, "torch_tensor")) {
        lat <- x$to(device = device, dtype = torch::torch_float32())
        # Set when the latents were encoded from pixels on the caller's
        # behalf (the stage-2 copy of a two-stage keyframe)
        pixel_frames <- kf$pixel_frames %||% NA_integer_
    } else {
        frames <- ltx23_preprocess_frames(x, height, width)
        pixel_frames <- frames$shape[3]
        lat <- encode(frames)$to(device = device,
                                 dtype = torch::torch_float32())
    }
    if (lat$ndim != 5L || lat$shape[1] != 1L || lat$shape[2] != 128L ||
        lat$shape[4] != lh || lat$shape[5] != lw) {
        stop(sprintf("keyframe latents must be [1, 128, k, %d, %d] for %dx%d generation; got [%s]",
                     lh, lw, width, height, paste(lat$shape, collapse = ", ")))
    }
    list(latents = lat, pixel_frames = pixel_frames)
}

# RoPE coordinates [1, 3, k*lh*lw, 2] (seconds, pixels) for k appended
# keyframe latent frames starting at pixel frame `frame`: the generation's
# own grid scaled to pixels with no causal fix, offset by `frame`, and
# narrowed to [frame, frame + 1) for a single image (diffusers
# _prepare_keyframe_coords).
.ltx23_keyframe_coords <- function(rope, k, lh, lw, frame, single, fps,
                                   device) {
    coords <- rope$prepare_video_coords(1L, k, lh, lw, device, fps = fps)
    st <- rope$scale_factors[1]
    start <- torch::torch_arange(0, k - 1L, dtype = torch::torch_float32(),
                                 device = device)$repeat_interleave(lh * lw) *
        st + frame
    end <- if (single) start + 1 else start + st
    coords[1, 1, , 1] <- start / fps
    coords[1, 1, , 2] <- end / fps
    coords
}

#' Build the conditioned starting state from keyframes
#'
#' @param keyframes List of \code{\link{ltx23_keyframe}} objects.
#' @param latent_frames,latent_height,latent_width Latent geometry.
#' @param noise Standard noise [1, 128, F', H', W'].
#' @param encode Function taking preprocessed frames to latents (only
#'   called for pixel keyframes).
#' @param height,width Pixel size of the generation.
#' @param rope The transformer's video RoPE (for appended coordinates).
#' @param fps Frame rate.
#' @param cond_noise_scale Partial re-noising of fully pinned frame-0
#'   tokens (the \code{txt2vid_ltx2} argument of that name).
#' @param base Optional normalized latents [1, 128, F', H', W'] to
#'   re-enter from (two-stage: the upsampled stage-1 result). NULL starts
#'   from pure noise.
#' @param noise_scale Entry noise level when \code{base} is given: each
#'   token is re-noised by \code{(1 - strength) * noise_scale}.
#' @return list(latents [1, S + A, 128], conditioning_mask [1, S + A]
#'   strengths (0 = free), clean [1, S + A, 128], extra_coords [1, 3, A,
#'   2] or NULL, n_main = S).
#' @keywords internal
.ltx23_keyframe_state <- function(keyframes, latent_frames, latent_height,
                                  latent_width, noise, encode, height, width,
                                  rope, fps, cond_noise_scale = 0,
                                  base = NULL, noise_scale = 1) {
    device <- noise$device
    f32 <- torch::torch_float32()
    clean <- torch::torch_zeros_like(noise)
    strength <- torch::torch_zeros(1L, 1L, latent_frames, latent_height,
                                   latent_width, dtype = f32, device = device)
    extra_lat <- list()
    extra_str <- list()
    extra_coords <- list()
    for (kf in keyframes) {
        enc <- .ltx23_keyframe_latents(kf, encode, height, width,
                                       latent_height, latent_width, device)
        k <- enc$latents$shape[3]
        if (kf$frame == 0L) {
            if (k > latent_frames) {
                stop(sprintf("keyframe at frame 0 has %d latent frames; the generation has %d",
                             k, latent_frames))
            }
            clean$narrow(3L, 1L, k)$copy_(enc$latents)
            strength$narrow(3L, 1L, k)$fill_(kf$strength)
        } else {
            extra_lat[[length(extra_lat) + 1L]] <- enc$latents
            extra_str[[length(extra_str) + 1L]] <- torch::torch_full(
                c(1L, k * latent_height * latent_width), kf$strength,
                dtype = f32, device = device)
            extra_coords[[length(extra_coords) + 1L]] <- .ltx23_keyframe_coords(
                rope, k, latent_height, latent_width, kf$frame,
                identical(enc$pixel_frames, 1L), fps, device)
        }
    }
    if (is.null(base)) {
        latents <- clean * strength + noise * strength$neg()$add(1)
    } else {
        # Re-entry (diffusers prepare_latents with supplied latents): the
        # keyframe content replaces base at its frame-0 positions, then
        # every token is re-noised by (1 - strength) * noise_scale
        init <- torch::torch_where(strength > 0, clean, base)
        scaled <- strength$neg()$add(1)$mul(noise_scale)
        latents <- noise * scaled + init * scaled$neg()$add(1)
    }
    if (cond_noise_scale > 0) {
        pinned <- (strength == 1)$to(dtype = f32)
        scale <- pinned * cond_noise_scale
        latents <- noise * scale + latents * (1 - scale)
    }
    latents <- ltx23_pack_video_latents(latents)
    clean_p <- ltx23_pack_video_latents(clean)
    mask <- ltx23_pack_video_latents(strength)$squeeze(3L)
    n_main <- latents$shape[2]
    extra <- NULL
    if (length(extra_lat)) {
        e_clean <- torch::torch_cat(lapply(extra_lat, ltx23_pack_video_latents),
                                    dim = 2L)
        e_str <- torch::torch_cat(extra_str, dim = 2L)
        e_s <- e_str$unsqueeze(3L)
        e_noise <- torch::torch_randn_like(e_clean)
        e_lat <- if (is.null(base)) {
            e_clean * e_s + e_noise * (1 - e_s)
        } else {
            e_scaled <- (1 - e_s) * noise_scale
            e_noise * e_scaled + e_clean * (1 - e_scaled)
        }
        latents <- torch::torch_cat(list(latents, e_lat), dim = 2L)
        clean_p <- torch::torch_cat(list(clean_p, e_clean), dim = 2L)
        mask <- torch::torch_cat(list(mask, e_str), dim = 2L)
        extra <- torch::torch_cat(extra_coords, dim = 3L)
    }
    list(latents = latents, conditioning_mask = mask, clean = clean_p,
         extra_coords = extra, n_main = n_main)
}
