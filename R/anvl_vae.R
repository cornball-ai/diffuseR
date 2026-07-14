#' FLUX.2 VAE decoder (anvl port of AutoencoderKLFlux2)
#'
#' anvl re-implementation of \code{diffuseR::flux2_vae_decoder}: the
#' standard AutoencoderKL decoder (conv_in, mid block with self
#' attention, four up blocks of three ResNet blocks with nearest-2x
#' upsampling on the first three, GroupNorm output head) plus the FLUX.2
#' \code{post_quant_conv} and BatchNorm latent de-normalization. Decodes
#' \code{[B, 32, H, W]} latents to \code{[B, 3, 8H, 8W]} pixels in
#' [-1, 1].
#'
#' @name vae
NULL

.YQ_VAE_EPS <- 1e-6

# ResNet block: (GroupNorm -> SiLU -> conv3x3) x2 + skip (conv1x1 if the
# channel count changes). Weights list: norm1_w/b, conv1_w/b, norm2_w/b,
# conv2_w/b, and optionally shortcut_w/b.
.yq_vae_resnet <- function(x, w) {
    h <- yunque::group_norm(x, w$norm1_w, w$norm1_b, 32L, .YQ_VAE_EPS)
    h <- yq_conv3x3(yunque::silu(h), w$conv1_w, w$conv1_b)
    h <- yunque::group_norm(h, w$norm2_w, w$norm2_b, 32L, .YQ_VAE_EPS)
    h <- yq_conv3x3(yunque::silu(h), w$conv2_w, w$conv2_b)
    if (!is.null(w$shortcut_w)) {
        x <- yq_conv1x1(x, w$shortcut_w, w$shortcut_b)
    }
    h + x
}

# 3x3 conv, stride 1, pad 1, with bias.
yq_conv3x3 <- function(x, weight, bias) {
    y <- anvl::nv_conv2d(x, weight, stride = 1L, padding = 1L)
    yq_add_conv_bias(y, bias)
}

# 1x1 conv, stride 1, no pad, with bias.
yq_conv1x1 <- function(x, weight, bias) {
    y <- anvl::nv_conv2d(x, weight, stride = 1L, padding = 0L)
    yq_add_conv_bias(y, bias)
}

# Add a per-output-channel bias [C_out] to a conv result [B, C_out, H, W].
yq_add_conv_bias <- function(y, bias) {
    s <- anvl::shape(y)
    y + anvl::nv_broadcast_to(anvl::nv_reshape(bias, c(1L, s[2L], 1L, 1L)), s)
}

# VAE self-attention over spatial positions (single head).
.yq_vae_attention <- function(x, w) {
    s <- anvl::shape(x)
    b <- s[1L]; c <- s[2L]; h <- s[3L]; wd <- s[4L]
    residual <- x
    xn <- yunque::group_norm(x, w$gn_w, w$gn_b, 32L, .YQ_VAE_EPS)
    # [B, C, H, W] -> [B, H*W, C]
    seq <- anvl::nv_transpose(anvl::nv_reshape(xn, c(b, c, h * wd)), c(1L, 3L, 2L))
    q <- yunque::linear(seq, w$q_w, w$q_b)
    k <- yunque::linear(seq, w$k_w, w$k_b)
    v <- yunque::linear(seq, w$v_w, w$v_b)
    add_head <- function(t) anvl::nv_reshape(t, c(b, 1L, h * wd, c))
    attn <- yunque::sdpa(add_head(q), add_head(k), add_head(v))   # scale 1/sqrt(C)
    attn <- anvl::nv_reshape(attn, c(b, h * wd, c))
    out <- yunque::linear(attn, w$out_w, w$out_b)
    # [B, H*W, C] -> [B, C, H, W]
    out <- anvl::nv_reshape(anvl::nv_transpose(out, c(1L, 3L, 2L)), c(b, c, h, wd))
    out + residual
}

# Up block: `n` ResNet blocks then optional nearest-2x + conv3x3.
.yq_vae_up_block <- function(x, w) {
    for (r in w$resnets) x <- .yq_vae_resnet(x, r)
    if (!is.null(w$up_conv_w)) {
        x <- yq_conv3x3(yunque::upsample_nearest2d(x), w$up_conv_w, w$up_conv_b)
    }
    x
}

#' FLUX.2 VAE decoder forward (anvl)
#'
#' @param z AnvlArray \code{[B, 32, H, W]} latents (already
#'   BN-denormalized; see \code{\link{yq_flux2_vae_prepare}}).
#' @param w VAE weights pytree from \code{\link{yq_flux2_load_vae}}.
#'
#' @return AnvlArray \code{[B, 3, 8H, 8W]} pixels in [-1, 1].
#'
#' @export
yq_flux2_vae_decode <- function(z, w) {
    x <- yq_conv1x1(z, w$post_quant_w, w$post_quant_b)
    x <- yq_conv3x3(x, w$conv_in_w, w$conv_in_b)
    x <- .yq_vae_resnet(x, w$mid$resnet1)
    x <- .yq_vae_attention(x, w$mid$attn)
    x <- .yq_vae_resnet(x, w$mid$resnet2)
    for (blk in w$up_blocks) x <- .yq_vae_up_block(x, blk)
    x <- yunque::group_norm(x, w$norm_out_w, w$norm_out_b, 32L, .YQ_VAE_EPS)
    yq_conv3x3(yunque::silu(x), w$conv_out_w, w$conv_out_b)
}

#' Prepare DiT latents for the FLUX.2 VAE decoder (host-side)
#'
#' Turns the DiT's final packed latents into the decoder's input:
#' unpacks tokens to the grid (\code{[B, S, 128] -> [B, 128, h2, w2]}),
#' applies the inverse BatchNorm normalization on the 128 packed
#' channels (\code{z * sqrt(var + eps) + mean}), then un-patchifies the
#' 2x2 blocks (\code{128 -> 32}, doubling each spatial dim). Parameter-
#' free reshaping and per-channel affine, computed in base R.
#'
#' @param latents R array \code{[B, S, 128]} (DiT output).
#' @param h2,w2 Integers. Packed grid dims (\code{S == h2 * w2}).
#' @param bn_mean,bn_var R vectors \code{[128]} of BatchNorm stats.
#' @param eps Numeric. BatchNorm epsilon (1e-4).
#' @param device Character.
#'
#' @return AnvlArray \code{[B, 32, 2 h2, 2 w2]}, f32.
#'
#' @export
yq_flux2_vae_prepare <- function(latents, h2, w2, bn_mean, bn_var,
                                 eps = 1e-4, device = "cpu") {
    d <- dim(latents)
    b <- d[1]; ch <- d[3]
    # unpack: [B, S, C] -> [B, C, h2, w2]  (permute then torch reshape)
    grid <- .yq_torch_reshape(aperm(latents, c(1L, 3L, 2L)),
                              c(b, ch, h2, w2))
    # BN denorm on the 128 packed channels
    std <- sqrt(bn_var + eps)
    grid <- sweep(sweep(grid, 2L, std, `*`), 2L, bn_mean, `+`)
    # unpatchify: [B, 4C, h2, w2] -> [B, C, 2 h2, 2 w2]
    a <- .yq_torch_reshape(grid, c(b, ch %/% 4L, 2L, 2L, h2, w2))
    a <- aperm(a, c(1L, 2L, 5L, 3L, 6L, 4L))
    a <- .yq_torch_reshape(a, c(b, ch %/% 4L, h2 * 2L, w2 * 2L))
    anvl::nv_array(a, dtype = "f32", device = device)
}

# Reshape an R array with torch (row-major) semantics.
.yq_torch_reshape <- function(a, dims) {
    flat <- aperm(a, rev(seq_along(dim(a))))
    out <- array(as.vector(flat), dim = rev(dims))
    aperm(out, rev(seq_along(dims)))
}
