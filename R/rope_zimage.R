#' Z-Image Rotary Positional Embeddings and Patchify Helpers
#'
#' Fresh R port of the Z-Image position scheme from the diffusers
#' reference (Apache-2.0,
#' src/diffusers/models/transformers/transformer_z_image.py RopeEmbedder,
#' create_coordinate_grid, _patchify_image, _pad_with_ids, unpatchify).
#' Z-Image uses 3-axis interleaved RoPE with theta 256; frequencies are
#' built in float64 but the angles are cast to float32 before cos/sin
#' (torch.polar on a .float() tensor), which differs measurably from the
#' FLUX convention at large positions. Every sub-sequence is padded to a
#' multiple of 32 (SEQ_MULTI_OF); caption positions are a 1-based ramp on
#' axis 1 built over the padded length, image positions sit on axes 2/3
#' with axis 1 offset just past the caption.
#'
#' @name rope_zimage
NULL

# Sub-sequences are padded to a multiple of this many tokens.
.zimage_seq_multi_of <- 32L

#' Padding length to the next multiple of 32
#'
#' @param n Integer token count.
#' @return Integer pad length in [0, 31].
#' @keywords internal
zimage_pad_len <- function(n) {
    (-n) %% .zimage_seq_multi_of
}

#' Build Z-Image caption position ids
#'
#' Caption tokens ramp 1..cap_padded_len on the first axis (axes 2 and 3
#' zero). The reference builds the grid over the already-padded length,
#' so pad tokens continue the ramp rather than sitting at the origin
#' (the (0,0,0) pad ids it also emits are truncated away in
#' _prepare_sequence and never reach RoPE).
#'
#' @param cap_padded_len Integer. Caption length after padding to a
#'   multiple of 32.
#' @param device Device for the resulting tensor.
#'
#' @return Float tensor of shape [cap_padded_len, 3].
#'
#' @export
zimage_cap_pos_ids <- function(cap_padded_len, device = "cpu") {
    f32 <- torch::torch_float32()
    ids <- torch::torch_zeros(cap_padded_len, 3L, dtype = f32, device = device)
    ids[, 1] <- torch::torch_arange(start = 1, end = cap_padded_len,
                                    dtype = f32, device = device)
    ids
}

#' Build Z-Image latent image position ids
#'
#' Image tokens use axis 1 for the frame index offset past the caption
#' (start0 = cap_padded_len + 1), axis 2 for the token row and axis 3 for
#' the token column. Trailing pad tokens (token count not a multiple of
#' 32) sit at (0, 0, 0). Reference: patchify_and_embed / _pad_with_ids.
#'
#' @param h_tokens Integer. Token grid height (latent height / patch).
#' @param w_tokens Integer. Token grid width (latent width / patch).
#' @param start0 Integer. First-axis start, cap_padded_len + 1.
#' @param f_tokens Integer. Token grid frames; 1 for txt2img.
#' @param device Device for the resulting tensor.
#'
#' @return Float tensor of shape [padded token count, 3].
#'
#' @export
zimage_img_pos_ids <- function(h_tokens, w_tokens, start0, f_tokens = 1L,
                               device = "cpu") {
    f32 <- torch::torch_float32()
    ids <- torch::torch_zeros(f_tokens, h_tokens, w_tokens, 3L, dtype = f32,
                              device = device)
    frames <- torch::torch_arange(start = start0,
                                  end = start0 + f_tokens - 1,
                                  dtype = f32, device = device)
    rows <- torch::torch_arange(start = 0, end = h_tokens - 1, dtype = f32,
                                device = device)
    cols <- torch::torch_arange(start = 0, end = w_tokens - 1, dtype = f32,
                                device = device)
    ids[,,, 1] <- ids[,,, 1] + frames$reshape(c(-1L, 1L, 1L))
    ids[,,, 2] <- ids[,,, 2] + rows$reshape(c(1L, -1L, 1L))
    ids[,,, 3] <- ids[,,, 3] + cols$reshape(c(1L, 1L, -1L))
    ids <- ids$reshape(c(f_tokens * h_tokens * w_tokens, 3L))

    pad <- zimage_pad_len(ids$shape[1])
    if (pad > 0L) {
        ids <- torch::torch_cat(list(
                                     ids,
                                     torch::torch_zeros(pad, 3L, dtype = f32, device = device)
        ))
    }
    ids
}

#' Compute Z-Image rotary frequencies from position ids
#'
#' Per-axis 1D rotary frequencies in the interleaved-real convention.
#' Frequencies and angles are built in float64, then the angles are cast
#' to float32 before cos/sin — matching the reference torch.polar call on
#' a .float() tensor. Output format matches \code{flux_pos_embed} so
#' \code{flux_apply_rotary_emb} applies unchanged.
#'
#' @param ids Tensor of shape [S, 3] from \code{zimage_cap_pos_ids} /
#'   \code{zimage_img_pos_ids}.
#' @param axes_dim Integer vector of per-axis rotary dims; must sum to
#'   the attention head dim. Z-Image uses c(32, 48, 48).
#' @param theta Numeric. RoPE base frequency. Z-Image uses 256.
#'
#' @return List of two tensors (cos, sin), each [S, sum(axes_dim)],
#'   float32, on the device of \code{ids}.
#'
#' @export
zimage_pos_embed <- function(ids, axes_dim = c(32L, 48L, 48L), theta = 256) {
    n_axes <- ids$shape[2]
    device <- ids$device
    f64 <- torch::torch_float64()
    f32 <- torch::torch_float32()
    pos <- ids$to(dtype = f64)$cpu()

    cos_out <- vector("list", n_axes)
    sin_out <- vector("list", n_axes)
    for (i in seq_len(n_axes)) {
        d <- axes_dim[i]
        exponents <- torch::torch_arange(start = 0, end = d - 2, step = 2,
            dtype = f64)
        freqs <- 1.0 / torch::torch_pow(theta, exponents / d)
        # Angle in float64, cast to float32 BEFORE cos/sin (reference:
        # torch.outer(...).float() then torch.polar)
        angles <- (pos[, i]$unsqueeze(2L) * freqs$unsqueeze(1L))$to(dtype = f32)
        cos_out[[i]] <- angles$cos()$repeat_interleave(2L, dim = 2L)
        sin_out[[i]] <- angles$sin()$repeat_interleave(2L, dim = 2L)
    }

    list(
         torch::torch_cat(cos_out, dim = -1L)$to(device = device),
         torch::torch_cat(sin_out, dim = -1L)$to(device = device)
    )
}

#' Patchify a latent image to Z-Image tokens
#'
#' (C, F, H, W) -> [F/pF * H/p * W/p, pF * p * p * C], matching
#' _patchify_image. No padding is applied here.
#'
#' @param image Tensor of shape [C, F, H, W].
#' @param patch_size Integer spatial patch size. Default 2.
#' @param f_patch_size Integer temporal patch size. Default 1.
#'
#' @return Tensor of shape [num_tokens, patch_dim].
#'
#' @export
zimage_patchify <- function(image, patch_size = 2L, f_patch_size = 1L) {
    p <- patch_size
    pf <- f_patch_size
    shape <- image$shape
    c_in <- shape[1]
    f_tokens <- shape[2] %/% pf
    h_tokens <- shape[3] %/% p
    w_tokens <- shape[4] %/% p
    image <- image$view(c(c_in, f_tokens, pf, h_tokens, p, w_tokens, p))
    # Python permute(1, 3, 5, 2, 4, 6, 0) -> R 1-indexed
    image <- image$permute(c(2L, 4L, 6L, 3L, 5L, 7L, 1L))
    image$reshape(c(f_tokens * h_tokens * w_tokens, pf * p * p * c_in))
}

#' Unpatchify Z-Image tokens back to a latent image
#'
#' Takes the first F/pF * H/p * W/p tokens (the image span of the
#' unified sequence) and reassembles [C, F, H, W], matching unpatchify.
#'
#' @param tokens Tensor of shape [S, pF * p * p * C] with the image
#'   tokens first.
#' @param size Integer vector c(F, H, W) of the target latent size.
#' @param patch_size Integer spatial patch size. Default 2.
#' @param f_patch_size Integer temporal patch size. Default 1.
#' @param out_channels Integer number of latent channels. Default 16.
#'
#' @return Tensor of shape [C, F, H, W].
#'
#' @export
zimage_unpatchify <- function(tokens, size, patch_size = 2L,
                              f_patch_size = 1L, out_channels = 16L) {
    p <- patch_size
    pf <- f_patch_size
    f_tokens <- size[1] %/% pf
    h_tokens <- size[2] %/% p
    w_tokens <- size[3] %/% p
    ori_len <- f_tokens * h_tokens * w_tokens
    x <- tokens[1:ori_len, ]
    x <- x$view(c(f_tokens, h_tokens, w_tokens, pf, p, p, out_channels))
    # Python permute(6, 0, 3, 1, 4, 2, 5) -> R 1-indexed
    x <- x$permute(c(7L, 1L, 4L, 2L, 5L, 3L, 6L))
    x$reshape(c(out_channels, size[1], size[2], size[3]))
}
