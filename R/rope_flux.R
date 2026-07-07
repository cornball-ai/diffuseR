#' FLUX Rotary Positional Embeddings
#'
#' Fresh R port of the FLUX rotary positional embedding scheme from the
#' diffusers reference implementation (Apache-2.0,
#' src/diffusers/models/transformers/transformer_flux.py FluxPosEmbed and
#' src/diffusers/models/embeddings.py get_1d_rotary_pos_embed /
#' apply_rotary_emb). FLUX uses the interleaved adjacent-pair convention
#' (use_real_unbind_dim = -1) with per-axis frequencies computed in
#' float64 and applied in float32. Text tokens carry all-zero ids, so
#' they receive the identity rotation.
#'
#' @name rope_flux
NULL

#' Build FLUX latent image position ids
#'
#' Position ids over the packed latent grid (latent height/2 x width/2).
#' Channel 1 is always zero, channel 2 holds the row index, channel 3 the
#' column index. Reference: FluxPipeline._prepare_latent_image_ids.
#'
#' @param height Integer. Packed grid height (latent height / 2).
#' @param width Integer. Packed grid width (latent width / 2).
#' @param device Device for the resulting tensor.
#'
#' @return Float tensor of shape [height * width, 3].
#'
#' @export
flux_prepare_latent_image_ids <- function(height, width, device = "cpu") {
    f32 <- torch::torch_float32()
    ids <- torch::torch_zeros(height, width, 3L, dtype = f32, device = device)
    # torch_arange has an inclusive end; end = n - 1 matches Python arange
    rows <- torch::torch_arange(start = 0, end = height - 1, dtype = f32,
                                device = device)
    cols <- torch::torch_arange(start = 0, end = width - 1, dtype = f32,
                                device = device)
    ids[,, 2] <- ids[,, 2] + rows$unsqueeze(2L)
    ids[,, 3] <- ids[,, 3] + cols$unsqueeze(1L)
    ids$reshape(c(height * width, 3L))
}

#' Compute FLUX rotary frequencies from position ids
#'
#' Per-axis 1D rotary frequencies (interleaved-real convention), computed
#' in float64 on CPU and concatenated over the axes. Reference:
#' FluxPosEmbed with get_1d_rotary_pos_embed(repeat_interleave_real=TRUE,
#' use_real=TRUE, freqs_dtype=float64).
#'
#' @param ids Tensor of shape [S, 3]: concatenated text ids (all zero)
#'   and image ids from \code{flux_prepare_latent_image_ids}.
#' @param axes_dim Integer vector of per-axis rotary dims; must sum to
#'   the attention head dim. FLUX uses c(16, 56, 56).
#' @param theta Numeric. RoPE base frequency.
#'
#' @return List of two tensors (cos, sin), each [S, sum(axes_dim)],
#'   float32, on the device of \code{ids}.
#'
#' @export
flux_pos_embed <- function(ids, axes_dim = c(16L, 56L, 56L), theta = 10000) {
    n_axes <- ids$shape[2]
    device <- ids$device
    f64 <- torch::torch_float64()
    # Frequencies in float64 on CPU: Blackwell fp64 throughput is 1/64,
    # and the tensors are tiny
    pos <- ids$to(dtype = f64)$cpu()

    cos_out <- vector("list", n_axes)
    sin_out <- vector("list", n_axes)
    for (i in seq_len(n_axes)) {
        d <- axes_dim[i]
        # freqs = 1 / theta^(seq(0, d - 2, by = 2) / d), length d / 2
        exponents <- torch::torch_arange(start = 0, end = d - 2, step = 2,
            dtype = f64)
        freqs <- 1.0 / torch::torch_pow(theta, exponents / d)
        # Outer product [S, d/2]
        freqs <- pos[, i]$unsqueeze(2L) * freqs$unsqueeze(1L)
        cos_out[[i]] <- freqs$cos()$repeat_interleave(2L, dim = 2L)
        sin_out[[i]] <- freqs$sin()$repeat_interleave(2L, dim = 2L)
    }

    f32 <- torch::torch_float32()
    list(
         torch::torch_cat(cos_out, dim = -1L)$to(dtype = f32, device = device),
         torch::torch_cat(sin_out, dim = -1L)$to(dtype = f32, device = device)
    )
}

#' Apply FLUX rotary embeddings to a per-head tensor
#'
#' Rotates adjacent element pairs of the last dimension:
#' \code{out = x * cos + rotate_half(x) * sin} with pairs interleaved
#' (elements 1,2 form the first complex pair). Math in float32, result
#' cast back to the input dtype. Reference: apply_rotary_emb with
#' use_real_unbind_dim = -1.
#'
#' @param x Tensor of shape [B, H, S, D] (per-head layout).
#' @param freqs List of two tensors (cos, sin), each [S, D], from
#'   \code{flux_pos_embed}.
#'
#' @return Tensor with the same shape and dtype as \code{x}.
#'
#' @export
flux_apply_rotary_emb <- function(x, freqs) {
    cos <- freqs[[1]]$unsqueeze(1L)$unsqueeze(1L) # [1, 1, S, D]
    sin <- freqs[[2]]$unsqueeze(1L)$unsqueeze(1L)

    pairs <- x$unflatten(4L, c(-1L, 2L)) # [B, H, S, D/2, 2]
    x_real <- pairs[,,,, 1]
    x_imag <- pairs[,,,, 2]
    x_rotated <- torch::torch_stack(list(-x_imag, x_real), dim = -1L)$flatten(start_dim = 4L)

    out <- x$to(dtype = torch::torch_float32()) * cos +
    x_rotated$to(dtype = torch::torch_float32()) * sin
    out$to(dtype = x$dtype)
}
