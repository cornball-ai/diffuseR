#' FLUX.2 Klein FlowMatch sigma schedule (host-side)
#'
#' Dynamic exponential timestep shift with the BFL empirical mu, plus the
#' terminal-0 sigma. Matches diffusers' \code{flowmatch_set_timesteps}
#' with \code{use_dynamic_shifting = TRUE}, \code{sigmas = seq(1, 1/n,
#' length.out = n)}. Distilled klein is guidance-free.
#'
#' @param seq_img Integer. Packed image token count.
#' @param n_steps Integer. Denoising steps (klein: 4).
#'
#' @return Numeric vector of length \code{n_steps + 1}: the shifted
#'   sigmas followed by 0. \code{sigmas[i]} is both the DiT timestep at
#'   step i and the ODE position; \code{dt = sigmas[i+1] - sigmas[i]}.
#'
#' @export
yq_flux2_sigmas <- function(seq_img, n_steps = 4L) {
    mu <- .yq_flux2_empirical_mu(seq_img, n_steps)
    sig_in <- seq(1, 1 / n_steps, length.out = n_steps)
    shifted <- exp(mu) / (exp(mu) + (1 / sig_in - 1))     # exponential shift
    c(shifted, 0)
}

# BFL piecewise-linear fit of mu vs image sequence length and steps.
.yq_flux2_empirical_mu <- function(image_seq_len, num_steps) {
    a1 <- 8.73809524e-05; b1 <- 1.89833333
    a2 <- 0.00016927; b2 <- 0.45666666
    if (image_seq_len > 4300) return(a2 * image_seq_len + b2)
    m_200 <- a2 * image_seq_len + b2
    m_10 <- a1 * image_seq_len + b1
    a <- (m_200 - m_10) / 190
    b <- m_200 - 200 * a
    a * num_steps + b
}

#' FLUX.2 Klein denoising loop (anvl)
#'
#' Runs the FlowMatch Euler loop over the jitted DiT: at each step,
#' predicts velocity and takes \code{latents <- latents + dt * velocity}
#' with \code{dt = sigmas[i+1] - sigmas[i]}. The DiT is jitted once and
#' called \code{n_steps} times; latents stay on-device across steps.
#' Guidance-free (distilled klein). Produces final packed latents (VAE
#' decode is a separate, convolution-dependent stage).
#'
#' @param latents AnvlArray \code{[B, S_img, in_channels]} initial noise
#'   (packed).
#' @param text_embeds AnvlArray \code{[B, S_txt, joint_dim]} from
#'   \code{\link{yq_qwen3_encoder}}.
#' @param sigmas Numeric vector from \code{\link{yq_flux2_sigmas}}.
#' @param cos,sin RoPE tables from \code{\link{yq_flux2_rope}} (built
#'   over the concatenated text + image ids).
#' @param w_dit DiT weights pytree from
#'   \code{\link{yq_flux2_load_weights}}.
#' @param dit_fn Optional pre-jitted DiT closure; built from
#'   \code{\link{yq_flux2_transformer}} if NULL.
#' @param device Character. Device for the timestep projections.
#' @param precision Character. Matmul precision.
#'
#' @return AnvlArray \code{[B, S_img, out_channels]} final latents.
#'
#' @export
yq_flux2_sample <- function(latents, text_embeds, sigmas, cos, sin, w_dit,
                            dit_fn = NULL, device = "cpu",
                            precision = "highest") {
    if (is.null(dit_fn)) {
        dit_fn <- anvl::jit(yq_flux2_transformer(precision = precision))
    }
    n_steps <- length(sigmas) - 1L
    for (i in seq_len(n_steps)) {
        tp <- yq_flux2_time_proj(sigmas[i], device = device)
        v <- dit_fn(latents, text_embeds, tp, cos, sin, w_dit)
        dt <- sigmas[i + 1L] - sigmas[i]
        latents <- latents + v * anvl::nv_scalar(dt, "f32", device = device)
    }
    latents
}
