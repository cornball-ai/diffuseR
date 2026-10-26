#' Edit an image with FLUX.2 klein
#'
#' Image + prompt to image with the same klein-4B weights as
#' \code{\link{txt2img_flux2}}: the reference image is VAE-encoded once,
#' its tokens ride after the noisy latents in every transformer forward
#' (position ids at T = 10, see \code{\link{flux2_prepare_image_ids}}),
#' and only the latents' predictions drive the scheduler. Reference:
#' diffusers Flux2KleinPipeline with \code{image =} (Apache-2.0).
#'
#' The image is checked (both sides at least 64 px, aspect ratio at most
#' 8), resized with Lanczos to 1024 x 1024 in area if it is larger, and
#' center-cropped to multiples of 16. One reference image is supported.
#'
#' @param prompt Character. The edit instruction or prompt.
#' @param image Path to a PNG/JPEG, or an [H, W, 3] array in [0, 1].
#' @param pipeline A \code{flux2_pipeline} loaded with
#'   \code{encoder = TRUE}; NULL loads one (passing \code{...} through).
#' @param width,height Integers divisible by 16, or NULL (default) for
#'   the processed reference image's size.
#' @param num_inference_steps Integer. Denoising steps (klein-4B: 4).
#' @param max_sequence_length Integer. Qwen3 token length (512).
#' @param seed Integer or NULL. See \code{\link{txt2img_flux2}}.
#' @param prompt_embeds Optional precomputed [B, S, 7680] embeddings.
#' @param save_file Logical. Write a PNG.
#' @param filename Output path (default derived from the prompt).
#' @param verbose Logical, or one of "silent", "progress", "steps".
#' @param ... Passed to \code{\link{flux2_load_pipeline}} when
#'   \code{pipeline} is NULL.
#'
#' @return Invisibly, \code{list(image, metadata)} where \code{image} is
#'   an [H, W, 3] array in [0, 1].
#'
#' @export
img2img_flux2 <- function(prompt, image, pipeline = NULL, width = NULL,
                          height = NULL, num_inference_steps = 4L,
                          max_sequence_length = 512L, seed = NULL,
                          prompt_embeds = NULL, save_file = TRUE,
                          filename = NULL, verbose = TRUE, ...) {
    level <- .verbosity(verbose)
    reference <- flux2_preprocess_image(image)
    width <- as.integer(width %||% reference$shape[4])
    height <- as.integer(height %||% reference$shape[3])
    if (level != "silent") {
        message(sprintf("FLUX.2 klein edit: %dx%d from a %dx%d reference, %d steps",
                        width, height, as.integer(reference$shape[4]),
                        as.integer(reference$shape[3]),
                        as.integer(num_inference_steps)))
    }
    if (is.null(pipeline)) {
        pipeline <- flux2_load_pipeline(..., encoder = TRUE,
                                        verbose = level == "steps")
    }
    .flux2_generate(prompt, pipeline, width, height, num_inference_steps,
                    max_sequence_length, seed, prompt_embeds, save_file,
                    filename, level, reference = reference)
}

#' Preprocess a FLUX.2 reference image
#'
#' Mirrors the diffusers klein pipeline: validate (both sides at least
#' 64 px, aspect ratio at most 8), Lanczos-resize to \code{max_area} if
#' larger (PIL's antialiased Lanczos-3, computed in floating point), floor
#' both sides to a multiple of 16, center-crop, and scale to [-1, 1].
#'
#' @param image Path to a PNG/JPEG, or an [H, W, C] array in [0, 1].
#' @param max_area Numeric. Pixel area cap.
#'
#' @return Float32 tensor [1, 3, H, W] in [-1, 1].
#'
#' @export
flux2_preprocess_image <- function(image, max_area = 1024 * 1024) {
    if (is.character(image)) {
        if (!file.exists(image)) {
            stop("image file not found: ", image, call. = FALSE)
        }
        image <- if (grepl("\\.png$", image, ignore.case = TRUE)) {
            png::readPNG(image)
        } else if (grepl("\\.jpe?g$", image, ignore.case = TRUE)) {
            jpeg::readJPEG(image)
        } else {
            stop("image must be a .png, .jpg or .jpeg file", call. = FALSE)
        }
    }
    if (!is.array(image) && !is.matrix(image)) {
        stop("image must be a file path or an [H, W, 3] array", call. = FALSE)
    }
    if (length(dim(image)) == 2L) {
        image <- array(rep(image, 3L), dim = c(dim(image), 3L))
    }
    if (length(dim(image)) != 3L) {
        stop("image must be an [H, W, 3] array", call. = FALSE)
    }
    if (dim(image)[3] == 4L) {
        image <- image[,, 1:3, drop = FALSE]
    }
    if (dim(image)[3] != 3L) {
        stop("image must have 3 colour channels", call. = FALSE)
    }

    h <- dim(image)[1]
    w <- dim(image)[2]
    if (h < 64L || w < 64L) {
        stop(sprintf("image too small: %dx%d; both sides must be at least 64 px",
                     w, h), call. = FALSE)
    }
    if (max(w / h, h / w) > 8) {
        stop(sprintf("aspect ratio too extreme: %dx%d; at most 8:1", w, h),
             call. = FALSE)
    }

    x <- torch::torch_tensor(image, dtype = torch::torch_float32())
    x <- x$permute(c(3L, 1L, 2L)) # [3, H, W]
    if (w * h > max_area) {
        scale <- sqrt(max_area / (w * h))
        new_w <- as.integer(floor(w * scale))
        new_h <- as.integer(floor(h * scale))
        x <- .lanczos_resize(x, new_h, new_w)
        h <- new_h
        w <- new_w
    }
    crop_h <- (h %/% 16L) * 16L
    crop_w <- (w %/% 16L) * 16L
    top <- (h - crop_h) %/% 2L
    left <- (w - crop_w) %/% 2L
    x <- x$narrow(2L, top + 1L, crop_h)$narrow(3L, left + 1L, crop_w)
    x$unsqueeze(1L)$mul(2)$sub(1)
}

# PIL's Lanczos-3 resampling weights for one axis (ImagingResample
# precompute_coeffs): the kernel support widens by the downscale factor,
# which is what antialiases, and each row is normalized to sum to 1.
# Returns an [out_size, in_size] matrix.
.lanczos_weights <- function(in_size, out_size) {
    scale <- in_size / out_size
    filterscale <- max(scale, 1)
    support <- 3 * filterscale
    sinc <- function(v) ifelse(v == 0, 1, sin(pi * v) / (pi * v))
    lanczos <- function(v) ifelse(abs(v) < 3, sinc(v) * sinc(v / 3), 0)
    m <- matrix(0, out_size, in_size)
    for (i in seq_len(out_size)) {
        center <- (i - 0.5) * scale
        xmin <- max(as.integer(center - support + 0.5), 0L)
        xmax <- min(as.integer(center + support + 0.5), in_size)
        if (xmax <= xmin) {
            next
        }
        src <- xmin:(xmax - 1L)
        k <- lanczos((src - center + 0.5) / filterscale)
        m[i, src + 1L] <- k / sum(k)
    }
    m
}

# Separable Lanczos resize of a [C, H, W] float tensor.
.lanczos_resize <- function(x, new_h, new_w) {
    f32 <- torch::torch_float32()
    wh <- torch::torch_tensor(.lanczos_weights(x$shape[2], new_h), dtype = f32)
    ww <- torch::torch_tensor(.lanczos_weights(x$shape[3], new_w), dtype = f32)
    x <- torch::torch_matmul(x, ww$t()) # [C, H, new_w]
    torch::torch_matmul(wh, x)$clamp(0, 1) # [C, new_h, new_w]
}
