#' Z-Image / FLUX.1 16-channel VAE decoder (anvl port of AutoencoderKL decode)
#'
#' anvl re-implementation of \code{diffuseR::vae_decoder_native} at
#' \code{latent_channels = 16}: the FLUX.1 / Z-Image \code{AutoencoderKL}
#' decode path. Structurally identical to the SD 2.1 and FLUX.2 VAE
#' decoders (\code{conv_in}; a mid block of ResNet, self attention,
#' ResNet; four up blocks of three ResNet blocks with nearest-2x
#' upsampling on the first three; a GroupNorm-32 -> SiLU -> \code{conv_out}
#' output head), so the block helpers (\code{.yq_vae_resnet},
#' \code{.yq_vae_attention}, \code{.yq_vae_up_block}) and conv helpers are
#' shared with \code{R/anvl_vae.R}.
#'
#' Two things distinguish the FLUX.1 / Z-Image VAE from the SD and FLUX.2
#' ports:
#' \itemize{
#'   \item \strong{16 latent channels} (SD was 4, FLUX.2 was 32).
#'   \item \strong{No \code{post_quant_conv}} \emph{and} \strong{no
#'     BatchNorm}. The FLUX family VAE sets \code{use_post_quant_conv =
#'     FALSE}, so the decoder runs \code{conv_in} straight on the latent.
#'     The latent de-normalization is a scalar shift+scale
#'     (\code{z / scaling_factor + shift_factor}), not SD's single 0.18215
#'     divide and not FLUX.2's per-channel BatchNorm; see
#'     \code{\link{yq_zimage_vae_prepare}}.
#' }
#'
#' Decodes \code{[B, 16, H, W]} latents to \code{[B, 3, 8H, 8W]} pixels in
#' [-1, 1].
#'
#' @name anvl_zimage_vae
NULL

# FLUX.1 / Z-Image VAE latent normalization (diffusers AutoencoderKL
# config, black-forest-labs/FLUX.1-schnell vae/config.json).
.YQ_ZIMAGE_VAE_SCALING <- 0.3611
.YQ_ZIMAGE_VAE_SHIFT <- 0.1159

#' Z-Image / FLUX.1 VAE decoder forward (anvl)
#'
#' Reuses the shared AutoencoderKL block helpers from \code{R/anvl_vae.R};
#' \code{anvl::jit()} the wrapper \code{function(z) yq_zimage_vae_decode(z, w)}.
#' \code{z} must already be de-normalized (see
#' \code{\link{yq_zimage_vae_prepare}}). Unlike the FLUX.2 decoder there is
#' no \code{post_quant_conv}: \code{conv_in} runs directly on the latent.
#'
#' @param z AnvlArray \code{[B, 16, H, W]} latents, already
#'   scaling/shift de-normalized.
#' @param w VAE weights pytree from \code{\link{yq_zimage_vae_load_weights}}.
#'
#' @return AnvlArray \code{[B, 3, 8H, 8W]} pixels in [-1, 1].
#'
#' @export
yq_zimage_vae_decode <- function(z, w) {
    x <- yq_conv3x3(z, w$conv_in_w, w$conv_in_b)
    x <- .yq_vae_resnet(x, w$mid$resnet1)
    x <- .yq_vae_attention(x, w$mid$attn)
    x <- .yq_vae_resnet(x, w$mid$resnet2)
    for (blk in w$up_blocks) x <- .yq_vae_up_block(x, blk)
    x <- yunque::group_norm(x, w$norm_out_w, w$norm_out_b, 32L, .YQ_VAE_EPS)
    yq_conv3x3(yunque::silu(x), w$conv_out_w, w$conv_out_b)
}

#' Prepare Z-Image / FLUX.1 latents for the VAE decoder (shift + scale)
#'
#' The diffusers FLUX pipeline de-normalizes latents before decoding with
#' a scalar affine (\code{z = z / scaling_factor + shift_factor}); this
#' applies that, returning the input to \code{\link{yq_zimage_vae_decode}}.
#' Both are 0-d scalar broadcasts, so latents stay on device (no host
#' round-trip). This is the FLUX.1 16-channel VAE convention, shared
#' verbatim by Z-Image-Turbo (\code{scaling_factor = 0.3611},
#' \code{shift_factor = 0.1159}) — distinct from SD 2.1's single 0.18215
#' divide and from FLUX.2's per-channel BatchNorm de-normalization.
#'
#' @param z AnvlArray \code{[B, 16, H, W]} latents from the sampling loop.
#' @param scaling_factor Numeric. VAE scaling factor (FLUX.1: 0.3611).
#' @param shift_factor Numeric. VAE shift factor (FLUX.1: 0.1159).
#'
#' @return AnvlArray \code{[B, 16, H, W]}, de-normalized.
#'
#' @export
yq_zimage_vae_prepare <- function(z, scaling_factor = .YQ_ZIMAGE_VAE_SCALING,
                                  shift_factor = .YQ_ZIMAGE_VAE_SHIFT) {
    dev <- anvl::device(z)
    z / anvl::nv_scalar(scaling_factor, "f32", device = dev) +
        anvl::nv_scalar(shift_factor, "f32", device = dev)
}

#' Load Z-Image / FLUX.1 VAE decoder weights into an anvl pytree
#'
#' Reads the decode half of a \code{diffuseR::vae_decoder_native}
#' (\code{latent_channels = 16}) state_dict (F16/F32 upcast to f32):
#' \code{conv_in}, the mid block, the four up blocks, and the output head.
#' There is no \code{post_quant_conv} and no encoder in this checkpoint
#' (the FLUX VAE omits both). Conv weights load raw \code{[out, in, kH, kW]}
#' (what \code{\link[anvl]{nv_conv2d}} expects); the attention linears
#' transpose to \code{[in, out]} for \code{\link[yunque]{yq_linear}}.
#' Optional ResNet \code{conv_shortcut} and up-block \code{upsamplers} are
#' detected by key existence. With \code{strict = TRUE} every key in the
#' file must be consumed exactly once (and no requested key may be
#' absent), so a wrong architecture or a missed weight fails loudly.
#'
#' @param path Path to the decoder state_dict \code{.safetensors} (native
#'   \code{vae_decoder_native} key names, no \code{decoder.} prefix).
#' @param device Character. Target device.
#' @param strict Logical. Enforce the exact key census.
#'
#' @return Weights pytree for \code{\link{yq_zimage_vae_decode}}.
#'
#' @export
yq_zimage_vae_load_weights <- function(path, device = "cpu", strict = TRUE) {
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

    w <- list(
        conv_in_w = raw("conv_in.weight"),
        conv_in_b = raw("conv_in.bias"),
        norm_out_w = raw("conv_norm_out.weight"),
        norm_out_b = raw("conv_norm_out.bias"),
        conv_out_w = raw("conv_out.weight"),
        conv_out_b = raw("conv_out.bias")
    )

    mp <- "mid_block."
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
        bp <- sprintf("up_blocks.%d.", i)
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
        used <- ls(seen)
        unused <- setdiff(all_keys, used)
        extra <- setdiff(used, all_keys)
        if (length(extra)) {
            stop("Z-Image VAE anvl load: ", length(extra),
                 " requested keys not in file, e.g. ",
                 paste(utils::head(extra, 5), collapse = ", "))
        }
        if (length(unused)) {
            stop("Z-Image VAE anvl load: ", length(unused),
                 " keys unused, e.g. ",
                 paste(utils::head(unused, 5), collapse = ", "))
        }
    }

    w
}
