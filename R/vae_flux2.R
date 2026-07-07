#' FLUX.2 Latent Layout and VAE Helpers
#'
#' Fresh R port of the FLUX.2 latent packing chain from the diffusers
#' reference (Apache-2.0, src/diffusers/pipelines/flux2/
#' pipeline_flux2_klein.py). The 32-channel VAE latent is patchified
#' 2x2 into 128 channels, normalized with the VAE's BatchNorm running
#' statistics (there is no scalar scaling/shift factor in FLUX.2), and
#' flattened to channels-last tokens for the transformer.
#'
#' @name vae_flux2
NULL

#' Patchify FLUX.2 latents (2x2 -> channels)
#'
#' [B, C, H, W] -> [B, 4C, H/2, W/2], channel order (C, ph, pw).
#' Reference: Flux2KleinPipeline._patchify_latents.
#'
#' @param latents Tensor [B, C, H, W]; H and W must be even.
#'
#' @return Tensor [B, C * 4, H / 2, W / 2].
#'
#' @export
flux2_patchify_latents <- function(latents) {
    shape <- latents$shape
    b <- shape[1]
    ch <- shape[2]
    h <- shape[3]
    w <- shape[4]
    latents <- latents$view(c(b, ch, h %/% 2L, 2L, w %/% 2L, 2L))
    # Python permute (0, 1, 3, 5, 2, 4), 1-indexed here
    latents <- latents$permute(c(1L, 2L, 4L, 6L, 3L, 5L))
    latents$reshape(c(b, ch * 4L, h %/% 2L, w %/% 2L))
}

#' Unpatchify FLUX.2 latents (channels -> 2x2)
#'
#' Inverse of \code{\link{flux2_patchify_latents}}. Reference:
#' Flux2KleinPipeline._unpatchify_latents.
#'
#' @param latents Tensor [B, 4C, H, W].
#'
#' @return Tensor [B, C, H * 2, W * 2].
#'
#' @export
flux2_unpatchify_latents <- function(latents) {
    shape <- latents$shape
    b <- shape[1]
    ch <- shape[2]
    h <- shape[3]
    w <- shape[4]
    latents <- latents$reshape(c(b, ch %/% 4L, 2L, 2L, h, w))
    # Python permute (0, 1, 4, 2, 5, 3), 1-indexed here
    latents <- latents$permute(c(1L, 2L, 5L, 3L, 6L, 4L))
    latents$reshape(c(b, ch %/% 4L, h * 2L, w * 2L))
}

#' Pack patchified FLUX.2 latents into tokens
#'
#' [B, C, H, W] -> [B, H * W, C] (row-major spatial flatten,
#' channels-last). Reference: Flux2KleinPipeline._pack_latents.
#'
#' @param latents Tensor [B, C, H, W].
#'
#' @return Tensor [B, H * W, C].
#'
#' @export
flux2_pack_latents <- function(latents) {
    shape <- latents$shape
    b <- shape[1]
    ch <- shape[2]
    latents <- latents$reshape(c(b, ch, shape[3] * shape[4]))
    latents$permute(c(1L, 3L, 2L))
}

#' Unpack FLUX.2 tokens back to a latent grid via position ids
#'
#' Scatters tokens to (H, W) positions taken from the id columns
#' (H = column 2, W = column 3, 0-based values). Reference:
#' Flux2KleinPipeline._unpack_latents_with_ids.
#'
#' @param x Tensor [B, S, C] of tokens.
#' @param ids Tensor [S, 4] (or [B, S, 4]) of position ids.
#' @param height,width Integers. Packed grid dimensions.
#'
#' @return Tensor [B, C, height, width].
#'
#' @export
flux2_unpack_latents_with_ids <- function(x, ids, height, width) {
    if (ids$ndim == 3L) {
        ids <- ids[1,,]
    }
    ids <- ids$to(device = x$device)
    long <- torch::torch_long()
    h_ids <- ids[, 2]$to(dtype = long)
    w_ids <- ids[, 3]$to(dtype = long)
    flat <- h_ids$mul(width)$add(w_ids)$add(1L) # 1-based scatter index

    shape <- x$shape
    b <- shape[1]
    ch <- shape[3]
    out <- torch::torch_zeros(b, height * width, ch, dtype = x$dtype,
                              device = x$device)
    index <- flat$unsqueeze(1L)$unsqueeze(3L)$expand(c(b, -1L, ch))
    out$scatter_(2L, index, x)
    out$view(c(b, height, width, ch))$permute(c(1L, 4L, 2L, 3L))
}

#' Normalize patchified latents with the VAE BatchNorm statistics
#'
#' FLUX.2 has no scalar scaling/shift factor; latents are standardized
#' per packed channel with the VAE's \code{bn.running_mean} /
#' \code{bn.running_var} (eps 1e-4). Reference: encode/decode paths of
#' Flux2KleinPipeline.
#'
#' @param latents Tensor [B, 128, H, W] (patchified).
#' @param bn_mean,bn_var Float tensors [128].
#' @param eps Numeric. BatchNorm epsilon.
#' @param inverse Logical. De-normalize (decode path) instead.
#'
#' @return Tensor like \code{latents}.
#'
#' @export
flux2_bn_normalize <- function(latents, bn_mean, bn_var, eps = 1e-4,
                               inverse = FALSE) {
    mean <- bn_mean$view(c(1L, -1L, 1L, 1L))$to(device = latents$device,
        dtype = latents$dtype)
    std <- bn_var$add(eps)$sqrt()$view(c(1L, -1L, 1L, 1L))$
    to(device = latents$device, dtype = latents$dtype)
    if (inverse) {
        latents$mul(std)$add(mean)
    } else {
        latents$sub(mean)$div(std)
    }
}

#' FLUX.2 VAE decoder
#'
#' The AutoencoderKLFlux2 decode path: post_quant_conv (1x1, 32
#' channels) followed by the standard AutoencoderKL decoder body
#' (reused from \code{\link{vae_decoder_native}}), plus the BatchNorm
#' running statistics used for latent (de)normalization. Reference:
#' src/diffusers/models/autoencoders/autoencoder_kl_flux2.py.
#'
#' @param latent_channels Integer (32 for FLUX.2).
#' @param block_channels Decoder block channels (reversed encoder
#'   block_out_channels).
#' @param norm_groups Integer. Group norm groups.
#'
#' @return Module whose forward(z) decodes [B, 32, H, W] latents to
#'   [B, 3, 8H, 8W] images; \code{$bn$running_mean} /
#'   \code{$bn$running_var} carry the normalization statistics.
#'
#' @export
flux2_vae_decoder <- torch::nn_module(
    "flux2_vae_decoder",
    initialize = function(latent_channels = 32L,
                          block_channels = c(512L, 512L, 256L, 128L),
                          norm_groups = 32L) {
    self$post_quant_conv <- torch::nn_conv2d(latent_channels,
                                             latent_channels,
                                             kernel_size = 1)
    self$decoder <- vae_decoder_native(
                                       latent_channels = latent_channels,
                                       block_channels = block_channels,
                                       norm_groups = norm_groups
    )
    bn_stats <- torch::nn_module(
                                 "flux2_bn_stats",
                                 initialize = function(n) {
        self$running_mean <- torch::nn_buffer(torch::torch_zeros(n))
        self$running_var <- torch::nn_buffer(torch::torch_ones(n))
    },
                                 forward = function(x) x
    )
    self$bn <- bn_stats(latent_channels * 4L)
},
    forward = function(z) {
    self$decoder(self$post_quant_conv(z))
}
)

#' Load the FLUX.2 VAE decoder from safetensors
#'
#' Loads the decoder half plus post_quant_conv and the BatchNorm running
#' statistics; encoder and quant_conv keys are skipped (txt2img needs no
#' encoder).
#'
#' @param path Path to the VAE .safetensors file (or a directory
#'   containing diffusion_pytorch_model.safetensors).
#' @param latent_channels,block_channels,norm_groups Constructor
#'   arguments for \code{\link{flux2_vae_decoder}}.
#' @param verbose Logical.
#'
#' @return The loaded \code{flux2_vae_decoder} in eval mode.
#'
#' @export
load_flux2_vae_decoder <- function(path, latent_channels = 32L,
                                   block_channels = c(512L, 512L, 256L, 128L),
                                   norm_groups = 32L, verbose = TRUE) {
    path <- path.expand(path)
    if (dir.exists(path)) {
        path <- file.path(path, "diffusion_pytorch_model.safetensors")
    }
    dec <- flux2_vae_decoder(latent_channels = latent_channels,
                             block_channels = block_channels,
                             norm_groups = norm_groups)

    handle <- safetensors::safetensors$new(path, framework = "torch")
    keys <- setdiff(handle$keys(), "__metadata__")
    keep <- keys[!startsWith(keys, "encoder.") &
    !startsWith(keys, "quant_conv.") &
    keys != "bn.num_batches_tracked"]

    dests <- c(dec$named_parameters(), dec$named_buffers())
    filled <- character(0)
    unmapped <- character(0)
    torch::with_no_grad({
        for (key in keep) {
            dest <- dests[[key]]
            if (is.null(dest)) {
                unmapped <- c(unmapped, key)
                next
            }
            dest$copy_(handle$get_tensor(key))
            filled <- c(filled, key)
        }
    })
    unfilled <- setdiff(names(dests), filled)
    if (length(unmapped)) {
        stop("FLUX.2 VAE load: ", length(unmapped), " unmapped keys, e.g. ",
             paste(utils::head(unmapped, 3), collapse = ", "))
    }
    if (length(unfilled)) {
        stop("FLUX.2 VAE load: ", length(unfilled),
             " unfilled params, e.g. ",
             paste(utils::head(unfilled, 3), collapse = ", "))
    }
    if (verbose) {
        message("Loaded ", length(filled), " FLUX.2 VAE tensors from ", path)
    }
    dec$eval()
    dec
}
