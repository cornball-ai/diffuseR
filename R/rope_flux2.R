#' FLUX.2 Position Ids and Empirical Shift
#'
#' Fresh R port of the FLUX.2 position-id builders and the empirical
#' timestep-shift formula from the diffusers reference (Apache-2.0,
#' src/diffusers/pipelines/flux2/pipeline_flux2_klein.py). FLUX.2 uses
#' 4-axis rotary position ids (T, H, W, L): text tokens carry only the
#' L axis (sequence position), image latents carry H and W, and the T
#' axis distinguishes reference images (unused for txt2img). Frequencies
#' come from \code{\link{flux_pos_embed}} with
#' \code{axes_dim = c(32, 32, 32, 32)} and \code{theta = 2000}.
#'
#' @name rope_flux2
NULL

#' Build FLUX.2 text position ids
#'
#' Columns (T, H, W, L) with only L varying: 0..len-1. Reference:
#' Flux2KleinPipeline._prepare_text_ids.
#'
#' @param len Integer. Text sequence length.
#' @param device Device for the resulting tensor.
#'
#' @return Float tensor [len, 4].
#'
#' @export
flux2_prepare_text_ids <- function(len, device = "cpu") {
    f32 <- torch::torch_float32()
    ids <- torch::torch_zeros(len, 4L, dtype = f32, device = device)
    ids[, 4] <- torch::torch_arange(start = 0, end = len - 1, dtype = f32,
                                    device = device)
    ids
}

#' Build FLUX.2 latent position ids
#'
#' Columns (T, H, W, L) with H and W carrying the packed-grid position
#' (row-major: H varies slowest), T = L = 0. Reference:
#' Flux2KleinPipeline._prepare_latent_ids.
#'
#' @param height Integer. Packed grid height (pixel height / 16).
#' @param width Integer. Packed grid width (pixel width / 16).
#' @param device Device for the resulting tensor.
#'
#' @return Float tensor [height * width, 4].
#'
#' @export
flux2_prepare_latent_ids <- function(height, width, device = "cpu") {
    f32 <- torch::torch_float32()
    ids <- torch::torch_zeros(height, width, 4L, dtype = f32,
                              device = device)
    rows <- torch::torch_arange(start = 0, end = height - 1, dtype = f32,
                                device = device)
    cols <- torch::torch_arange(start = 0, end = width - 1, dtype = f32,
                                device = device)
    ids[, , 2] <- ids[, , 2] + rows$unsqueeze(2L)
    ids[, , 3] <- ids[, , 3] + cols$unsqueeze(1L)
    ids$reshape(c(height * width, 4L))
}

#' Empirical timestep shift for FLUX.2
#'
#' BFL's piecewise-linear fit of the dynamic-shifting mu as a function of
#' image sequence length and step count; replaces FLUX.1's
#' calculate_shift. Reference: compute_empirical_mu (adapted from BFL
#' sampling.py).
#'
#' @param image_seq_len Integer. Packed image token count.
#' @param num_steps Integer. Inference steps.
#'
#' @return Numeric mu for \code{\link{flowmatch_set_timesteps}}.
#'
#' @export
flux2_empirical_mu <- function(image_seq_len, num_steps) {
    a1 <- 8.73809524e-05
    b1 <- 1.89833333
    a2 <- 0.00016927
    b2 <- 0.45666666

    if (image_seq_len > 4300) {
        return(a2 * image_seq_len + b2)
    }
    m_200 <- a2 * image_seq_len + b2
    m_10 <- a1 * image_seq_len + b1
    a <- (m_200 - m_10) / 190.0
    b <- m_200 - 200.0 * a
    a * num_steps + b
}
