#' Preprocess a reference image for FLUX 3
#'
#' The single-reference ("fill") path of the reference pipeline
#' (diffusers Flux3ImageProcessor, Apache-2.0, following BFL's
#' \code{reference_canvas}): the image is kept whole. It is downscaled,
#' keeping its aspect ratio, until its canvas (its size rounded up to the
#' 16 px token grid, sides clamped to [64, \code{max_side}]) fits
#' \code{max_area}; backing the scale off by 0.5\% per try when the
#' rounding overshoots. The resize is PIL's antialiased Lanczos-3 in
#' floating point. The image is then centered on the canvas and the
#' border filled with its reflection (numpy \code{"reflect"}, or
#' \code{"symmetric"} when a side is at most 16 px).
#'
#' @param image Path to a PNG/JPEG, or an [H, W, C] array in [0, 1].
#' @param max_area Numeric. Canvas area cap (reference: 4 megapixels).
#'   Lower it to bound memory; the reference tokens are
#'   \code{canvas area / 256}.
#' @param max_side Integer. Canvas side cap.
#'
#' @return \code{list(image, crop)}: a float32 tensor [1, 3, H, W] in
#'   [-1, 1] at the canvas size, and the (left, top, right, bottom) box
#'   of the image inside the canvas (0-based, right/bottom exclusive),
#'   used to crop the generated image back.
#'
#' @export
flux3_preprocess_image <- function(image, max_area = 4 * 1024 * 1024,
                                   max_side = 4096L) {
    image <- .read_rgb_image(image)
    h <- dim(image)[1]
    w <- dim(image)[2]

    scale <- min(1, sqrt(max_area / (w * h)), max_side / max(w, h))
    repeat {
        nh <- max(1, round(h * scale))
        nw <- max(1, round(w * scale))
        canvas <- .flux3_canvas(nh, nw, max_side)
        if (prod(canvas) <= max_area) {
            break
        }
        scale <- scale * 0.995
    }

    x <- torch::torch_tensor(image, dtype = torch::torch_float32())
    x <- x$permute(c(3L, 1L, 2L)) # [3, H, W]
    if (nh != h || nw != w) {
        x <- .lanczos_resize(x, nh, nw)
    }

    ch <- canvas[1]
    cw <- canvas[2]
    top <- (ch - nh) %/% 2
    left <- (cw - nw) %/% 2
    mode <- if (min(nh, nw) > 16) "reflect" else "symmetric"
    rows <- .flux3_pad_index(nh, top, ch - nh - top, mode)
    cols <- .flux3_pad_index(nw, left, cw - nw - left, mode)
    x <- x[, rows, cols, drop = FALSE]

    list(image = x$unsqueeze(1L)$mul(2)$sub(1),
         crop = c(left = left, top = top, right = left + nw, bottom = top + nh))
}

# Canvas (height, width): each side rounded up to a multiple of 16 and
# clamped to [min_side, max_side]. Reference: _get_canvas_height_width.
.flux3_canvas <- function(h, w, max_side = 4096L, min_side = 64L) {
    vapply(c(h, w), function(side) {
        min(max_side, max(min_side, ceiling(side / 16) * 16))
    }, numeric(1))
}

# 1-based source indices for numpy.pad(mode = "reflect" | "symmetric")
# along one axis of length n, padded by `before` and `after`. Both modes
# are periodic extensions, so any pad width works.
.flux3_pad_index <- function(n, before, after, mode) {
    j <- seq(-before, n + after - 1)
    if (n == 1) {
        return(rep(1L, length(j)))
    }
    if (mode == "reflect") {
        period <- 2 * (n - 1)
        i <- j %% period
        i <- ifelse(i >= n, period - i, i)
    } else {
        period <- 2 * n
        i <- j %% period
        i <- ifelse(i >= n, period - 1 - i, i)
    }
    as.integer(i) + 1L
}
