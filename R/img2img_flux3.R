#' Edit an image with FLUX 3 Image Klein
#'
#' Image + caption to image with the same weights as
#' \code{\link{txt2img_flux3}}. The reference is kept whole (see
#' \code{\link{flux3_preprocess_image}}): scaled so its canvas fits
#' \code{max_area}, reflect-padded to the 16 px grid, VAE-encoded once,
#' and passed to every transformer forward as its own token mode
#' (position ids at L = -1). The output defaults to the canvas size and
#' is cropped back to the reference's box. Reference: diffusers
#' Flux3DistilledModularPipeline with \code{image =} (Apache-2.0).
#'
#' Write the caption as a full description of the edited result (the
#' reference with the edit folded in); a one-line instruction reframes
#' the shot. BFL's prompt upsampler (flux-3-image-klein-pup) writes such
#' captions from an instruction plus the reference image.
#'
#' @param prompt Character. The full caption of the edited image.
#' @param image Path to a PNG/JPEG, or an [H, W, 3] array in [0, 1].
#' @param pipeline A \code{flux3_pipeline} loaded with
#'   \code{encoder = TRUE}; NULL loads one (passing \code{...} through).
#' @param width,height Integers divisible by 16, or NULL (default) for
#'   the reference canvas, cropped back to the reference afterwards.
#' @param max_area Numeric. Reference canvas area cap (reference: 4
#'   megapixels). The reference adds \code{area / 256} tokens to every
#'   forward, so lower it on small cards.
#' @param num_inference_steps Integer. Denoising steps (16).
#' @param max_sequence_length Integer. Qwen3-VL token cap (8192).
#' @param seed Integer or NULL. See \code{\link{txt2img_flux3}}.
#' @param prompt_embeds Optional precomputed [1, S, 20480] embeddings.
#' @param save_file Logical. Write a PNG.
#' @param filename Output path (default derived from the prompt).
#' @param verbose Logical, or one of "silent", "progress", "steps".
#' @param ... Passed to \code{\link{flux3_load_pipeline}} when
#'   \code{pipeline} is NULL.
#'
#' @return Invisibly, \code{list(image, metadata)} where \code{image} is
#'   an [H, W, 3] array in [0, 1].
#'
#' @export
img2img_flux3 <- function(prompt, image, pipeline = NULL, width = NULL,
                          height = NULL, max_area = 4 * 1024 * 1024,
                          num_inference_steps = 16L,
                          max_sequence_length = 8192L, seed = NULL,
                          prompt_embeds = NULL, save_file = TRUE,
                          filename = NULL, verbose = TRUE, ...) {
    level <- .verbosity(verbose)
    pre <- flux3_preprocess_image(image, max_area = max_area)
    canvas_w <- as.integer(pre$image$shape[4])
    canvas_h <- as.integer(pre$image$shape[3])
    # Crop back only when the output is the canvas (the reference's
    # default); an explicit size is returned whole
    crop <- if (is.null(width) && is.null(height)) pre$crop else NULL
    width <- as.integer(width %||% canvas_w)
    height <- as.integer(height %||% canvas_h)
    if (level != "silent") {
        message(sprintf("FLUX 3 klein edit: %dx%d from a %dx%d reference canvas, %d steps",
                        width, height, canvas_w, canvas_h,
                        as.integer(num_inference_steps)))
    }
    if (is.null(pipeline)) {
        pipeline <- flux3_load_pipeline(..., encoder = TRUE,
                                        verbose = level == "steps")
    }
    .flux3_generate(prompt, pipeline, width, height, num_inference_steps,
                    max_sequence_length, seed, prompt_embeds, save_file,
                    filename, level, reference = pre$image, crop = crop)
}
