#' FLUX 3 position ids
#'
#' 4-axis (T, H, W, L) position ids for the FLUX 3 transformer, from the
#' diffusers reference (Apache-2.0,
#' src/diffusers/modular_pipelines/flux3/before_denoise.py). Image tokens
#' sit at T = 0 with the H/W axes scaled by the image RoPE scale (0.5);
#' the generated image has L = 0 and the i-th reference L = -i. Text
#' advances along L. Rotary frequencies come from
#' \code{\link{flux_pos_embed}} with \code{axes_dim = c(32, 32, 32, 32)},
#' \code{theta = 10000}.
#'
#' @name rope_flux3
NULL

#' FLUX 3 image token position ids
#'
#' @param height Integer. Token grid rows (pixels / 16).
#' @param width Integer. Token grid columns (pixels / 16).
#' @param l_coord Numeric. L coordinate: 0 for the generated image,
#'   \code{-i} for the i-th reference image.
#' @param rope_scale Numeric. Scale of the H/W axes (0.5).
#' @param device Character. Device.
#'
#' @return Float tensor [height * width, 4], row-major over (h, w).
#'
#' @export
flux3_prepare_image_ids <- function(height, width, l_coord = 0,
                                    rope_scale = 0.5, device = "cpu") {
    h <- rep(seq_len(height) - 1, each = width)
    w <- rep(seq_len(width) - 1, times = height)
    ids <- cbind(0, h * rope_scale, w * rope_scale, l_coord)
    torch::torch_tensor(ids, dtype = torch::torch_float32(), device = device)
}

#' FLUX 3 text token position ids
#'
#' @param len Integer. Text sequence length (padded).
#' @param device Character. Device.
#'
#' @return Float tensor [len, 4]: (0, 0, 0, 0..len-1).
#'
#' @export
flux3_prepare_text_ids <- function(len, device = "cpu") {
    ids <- cbind(0, 0, 0, seq_len(len) - 1)
    torch::torch_tensor(ids, dtype = torch::torch_float32(), device = device)
}

#' FLUX 3 distilled sampling schedule
#'
#' FlowMatch Euler with the static shift of the FLUX 3 recipe (6.93)
#' over \code{sigmas = linspace(1, 1 / n, n)}. Reference:
#' Flux3SetTimestepsStep.
#'
#' @param num_inference_steps Integer (16 for the distilled checkpoint).
#' @param shift Numeric. Static timestep shift.
#' @param device Character. Device.
#'
#' @return A FlowMatch schedule from \code{\link{flowmatch_set_timesteps}}.
#'
#' @export
flux3_schedule <- function(num_inference_steps = 16L, shift = 6.93,
                           device = "cpu") {
    n <- as.integer(num_inference_steps)
    schedule <- flowmatch_scheduler_create(shift = shift)
    flowmatch_set_timesteps(schedule, num_inference_steps = n,
                            device = device,
                            sigmas = seq(1, 1 / n, length.out = n))
}
