#' Stable Diffusion XL VAE decoder (anvl port of AutoencoderKL decode)
#'
#' anvl re-implementation of the SDXL \code{AutoencoderKL} decode path.
#' SDXL's VAE is the same 4-channel \code{AutoencoderKL} as SD 2.1's
#' (only the \code{scaling_factor} differs: 0.13025 vs 0.18215), so the
#' decode body is identical: the scalar \code{scaling_factor} latent
#' rescale (host-side, see \code{\link{yq_sdxl_vae_prepare}}), the 1x1
#' \code{post_quant_conv}, then the standard AutoencoderKL decoder
#' (\code{conv_in}; a mid block of ResNet, self attention, ResNet; four up
#' blocks of three ResNet blocks with nearest-2x upsampling on the first
#' three; a GroupNorm-32 -> SiLU -> \code{conv_out} output head). The block
#' and conv helpers (\code{.yq_vae_resnet}, \code{.yq_vae_attention},
#' \code{.yq_vae_up_block}, \code{yq_conv3x3}, \code{yq_conv1x1}) are shared
#' with \code{R/anvl_vae.R}. Decodes \code{[B, 4, H, W]} latents to
#' \code{[B, 3, 8H, 8W]} pixels in [-1, 1].
#'
#' @name anvl_sdxl_vae
NULL

# SDXL VAE scaling factor (diffusers AutoencoderKL config; note the base
# sdxl-vae ships fp16-fixed but the scale is the same). Distinct from the
# 0.18215 SD 1.x/2.x factor.
.YQ_SDXL_VAE_SCALING <- 0.13025

#' Stable Diffusion XL VAE decoder forward (anvl)
#'
#' Reuses the shared AutoencoderKL block helpers from \code{R/anvl_vae.R};
#' \code{anvl::jit()} the wrapper \code{function(z) yq_sdxl_vae_decode(z, w)}.
#' \code{z} must already be rescaled by the VAE scaling factor (see
#' \code{\link{yq_sdxl_vae_prepare}}).
#'
#' @param z AnvlArray \code{[B, 4, H, W]} latents, already scaling-factor
#'   rescaled.
#' @param w VAE weights pytree from \code{\link{yq_sdxl_vae_load_weights}}.
#'
#' @return AnvlArray \code{[B, 3, 8H, 8W]} pixels in [-1, 1].
#'
#' @export
yq_sdxl_vae_decode <- function(z, w) {
    x <- yq_conv1x1(z, w$post_quant_w, w$post_quant_b)
    x <- yq_conv3x3(x, w$conv_in_w, w$conv_in_b)
    x <- .yq_vae_resnet(x, w$mid$resnet1)
    x <- .yq_vae_attention(x, w$mid$attn)
    x <- .yq_vae_resnet(x, w$mid$resnet2)
    for (blk in w$up_blocks) x <- .yq_vae_up_block(x, blk)
    x <- yunque::group_norm(x, w$norm_out_w, w$norm_out_b, 32L, .YQ_VAE_EPS)
    yq_conv3x3(yunque::silu(x), w$conv_out_w, w$conv_out_b)
}

#' Prepare SDXL latents for the VAE decoder (scaling factor)
#'
#' The diffusers SDXL pipeline divides latents by the VAE
#' \code{scaling_factor} (0.13025) before decoding
#' (\code{z = z / scaling_factor}); this applies that rescale, returning
#' the input to \code{\link{yq_sdxl_vae_decode}}. A scalar-broadcast
#' divide, so latents stay on device (no host round-trip).
#'
#' @param z AnvlArray \code{[B, 4, H, W]} latents from the sampling loop.
#' @param scaling_factor Numeric. VAE scaling factor (SDXL: 0.13025).
#'
#' @return AnvlArray \code{[B, 4, H, W]}, rescaled.
#'
#' @export
yq_sdxl_vae_prepare <- function(z, scaling_factor = .YQ_SDXL_VAE_SCALING) {
    z / anvl::nv_scalar(scaling_factor, "f32", device = anvl::device(z))
}

#' Load SDXL VAE decoder weights into an anvl pytree
#'
#' Reads the decode half of the diffusers SDXL \code{AutoencoderKL}
#' checkpoint (F16 upcast to f32): \code{post_quant_conv}, the decoder
#' body, and the output head. Conv weights load raw \code{[out, in, kH,
#' kW]} (what \code{\link[anvl]{nv_conv2d}} expects); the attention
#' linears transpose to \code{[in, out]} for \code{\link[yunque]{yq_linear}}.
#' Encoder and \code{quant_conv} keys are skipped (txt2img needs only the
#' decode path). With \code{strict = TRUE} every \code{decoder.*} /
#' \code{post_quant_conv.*} key must be consumed exactly once (and no
#' requested key may be absent), so a wrong architecture or a missed
#' weight fails loudly. The SDXL VAE key tree is byte-for-byte the SD 2.1
#' one, so this loader mirrors \code{diffuseR}'s SD 2.1 VAE loader.
#'
#' @param path Path to \code{vae/diffusion_pytorch_model.safetensors}.
#' @param device Character. Target device.
#' @param strict Logical. Enforce the exact decode-key census.
#'
#' @return Weights pytree for \code{\link{yq_sdxl_vae_decode}}.
#'
#' @export
yq_sdxl_vae_load_weights <- function(path, device = "cpu", strict = TRUE) {
    st <- yunque::st_open(path)
    on.exit(close(st$con))
    seen <- new.env(parent = emptyenv())
    raw <- function(key) {
        assign(key, TRUE, envir = seen)
        anvl::nv_array(yunque::st_read(st, key), dtype = "f32", device = device)
    }
    lin <- function(key) {
        assign(key, TRUE, envir = seen)
        anvl::nv_array(yunque::st_read(st, key, transpose = TRUE),
                       dtype = "f32", device = device)
    }
    has <- function(key) !is.null(st$header[[key]])

    resnet <- function(p) {
        r <- list(
            norm1_w = raw(paste0(p, "norm1.weight")),
            norm1_b = raw(paste0(p, "norm1.bias")),
            conv1_w = raw(paste0(p, "conv1.weight")),
            conv1_b = raw(paste0(p, "conv1.bias")),
            norm2_w = raw(paste0(p, "norm2.weight")),
            norm2_b = raw(paste0(p, "norm2.bias")),
            conv2_w = raw(paste0(p, "conv2.weight")),
            conv2_b = raw(paste0(p, "conv2.bias"))
        )
        if (has(paste0(p, "conv_shortcut.weight"))) {
            r$shortcut_w <- raw(paste0(p, "conv_shortcut.weight"))
            r$shortcut_b <- raw(paste0(p, "conv_shortcut.bias"))
        }
        r
    }

    dp <- "decoder."
    w <- list(
        post_quant_w = raw("post_quant_conv.weight"),
        post_quant_b = raw("post_quant_conv.bias"),
        conv_in_w = raw(paste0(dp, "conv_in.weight")),
        conv_in_b = raw(paste0(dp, "conv_in.bias")),
        norm_out_w = raw(paste0(dp, "conv_norm_out.weight")),
        norm_out_b = raw(paste0(dp, "conv_norm_out.bias")),
        conv_out_w = raw(paste0(dp, "conv_out.weight")),
        conv_out_b = raw(paste0(dp, "conv_out.bias"))
    )

    mp <- paste0(dp, "mid_block.")
    ap <- paste0(mp, "attentions.0.")
    w$mid <- list(
        resnet1 = resnet(paste0(mp, "resnets.0.")),
        resnet2 = resnet(paste0(mp, "resnets.1.")),
        attn = list(
            gn_w = raw(paste0(ap, "group_norm.weight")),
            gn_b = raw(paste0(ap, "group_norm.bias")),
            q_w = lin(paste0(ap, "to_q.weight")), q_b = raw(paste0(ap, "to_q.bias")),
            k_w = lin(paste0(ap, "to_k.weight")), k_b = raw(paste0(ap, "to_k.bias")),
            v_w = lin(paste0(ap, "to_v.weight")), v_b = raw(paste0(ap, "to_v.bias")),
            out_w = lin(paste0(ap, "to_out.0.weight")),
            out_b = raw(paste0(ap, "to_out.0.bias"))
        )
    )

    w$up_blocks <- lapply(0:3, function(i) {
        bp <- sprintf("%sup_blocks.%d.", dp, i)
        blk <- list(resnets = lapply(0:2, function(j)
            resnet(sprintf("%sresnets.%d.", bp, j))))
        if (has(paste0(bp, "upsamplers.0.conv.weight"))) {
            blk$up_conv_w <- raw(paste0(bp, "upsamplers.0.conv.weight"))
            blk$up_conv_b <- raw(paste0(bp, "upsamplers.0.conv.bias"))
        }
        blk
    })

    if (strict) {
        all_keys <- setdiff(names(st$header), "__metadata__")
        decode_keys <- all_keys[startsWith(all_keys, "decoder.") |
                                startsWith(all_keys, "post_quant_conv.")]
        used <- ls(seen)
        unused <- setdiff(decode_keys, used)
        extra <- setdiff(used, all_keys)
        if (length(extra)) {
            stop("SDXL VAE anvl load: ", length(extra),
                 " requested keys not in file, e.g. ",
                 paste(utils::head(extra, 5), collapse = ", "))
        }
        if (length(unused)) {
            stop("SDXL VAE anvl load: ", length(unused),
                 " decode keys unused, e.g. ",
                 paste(utils::head(unused, 5), collapse = ", "))
        }
    }

    w
}
