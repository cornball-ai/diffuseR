#' FLUX rotary position embedding tables (host-side)
#'
#' Base-R port of \code{diffuseR::flux_pos_embed}: per-axis frequencies
#' (float64 outer products), cos/sin, each doubled by interleave, then
#' concatenated across axes. Deterministic, no weights.
#'
#' @param ids Numeric matrix \code{[S, n_axes]} of position ids.
#' @param axes_dim Integer vector of per-axis rotary dims (must sum to
#'   the attention head dim; FLUX.2: \code{c(32, 32, 32, 32)}).
#' @param theta Numeric. Base frequency (FLUX.2: 2000).
#' @param device Character. Target device.
#'
#' @return List \code{list(cos, sin)}, each an AnvlArray
#'   \code{[S, sum(axes_dim)]}, f32.
#'
#' @export
yq_flux_pos_embed <- function(ids, axes_dim = c(32L, 32L, 32L, 32L),
                              theta = 2000, device = "cpu") {
    ids <- as.matrix(ids)
    n_axes <- ncol(ids)
    dbl <- function(m) m[, rep(seq_len(ncol(m)), each = 2L), drop = FALSE]
    cos_parts <- vector("list", n_axes)
    sin_parts <- vector("list", n_axes)
    for (i in seq_len(n_axes)) {
        d <- axes_dim[i]
        exponents <- seq(0, d - 2L, by = 2L)      # length d/2
        freqs <- 1 / theta^(exponents / d)
        ang <- outer(ids[, i], freqs)             # [S, d/2]
        cos_parts[[i]] <- dbl(cos(ang))
        sin_parts[[i]] <- dbl(sin(ang))
    }
    list(
        cos = anvl::nv_array(do.call(cbind, cos_parts), dtype = "f32",
                             device = device),
        sin = anvl::nv_array(do.call(cbind, sin_parts), dtype = "f32",
                             device = device)
    )
}

#' FLUX.2 Klein RoPE tables for a txt2img forward
#'
#' Builds the concatenated [text; image] position ids (text tokens carry
#' only the L/sequence axis; image latents carry H and W over the packed
#' grid) and returns the rotary tables. Reference:
#' \code{Flux2KleinPipeline._prepare_{text,latent}_ids}.
#'
#' @param text_len Integer. Text sequence length.
#' @param height Integer. Packed grid height (pixel height / 16).
#' @param width Integer. Packed grid width (pixel width / 16).
#' @param theta Numeric. Base frequency (2000).
#' @param device Character. Target device.
#'
#' @return List \code{list(cos, sin)}, each \code{[text_len + height*width, 128]}.
#'
#' @export
yq_flux2_rope <- function(text_len, height, width, theta = 2000,
                          device = "cpu") {
    txt <- matrix(0, nrow = text_len, ncol = 4L)
    txt[, 4L] <- 0:(text_len - 1L)
    grid <- matrix(0, nrow = height * width, ncol = 4L)
    # row-major packing: H varies slowest (matches torch reshape)
    grid[, 2L] <- rep(0:(height - 1L), each = width)
    grid[, 3L] <- rep(0:(width - 1L), times = height)
    ids <- rbind(txt, grid)
    yq_flux_pos_embed(ids, axes_dim = c(32L, 32L, 32L, 32L), theta = theta,
                      device = device)
}
