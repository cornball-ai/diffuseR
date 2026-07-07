#' FLUX.1 Text-to-Image Pipeline
#'
#' FLUX latent packing helpers and (in later phases) the schnell
#' text-to-image pipeline. Ported from the diffusers reference
#' implementation (Apache-2.0, src/diffusers/pipelines/flux/
#' pipeline_flux.py).
#'
#' @name txt2img_flux
NULL

#' Pack FLUX latents into a patch sequence
#'
#' Packs a [B, C, H, W] latent into 2x2 patches, giving a sequence
#' [B, (H/2) * (W/2), C * 4]. Reference: FluxPipeline._pack_latents.
#'
#' @param latents Tensor of shape [B, C, H, W]; H and W must be even.
#'
#' @return Tensor of shape [B, (H/2) * (W/2), C * 4].
#'
#' @export
flux_pack_latents <- function(latents) {
    shape <- latents$shape
    b <- shape[1]
    ch <- shape[2]
    h <- shape[3]
    w <- shape[4]
    latents <- latents$view(c(b, ch, h %/% 2L, 2L, w %/% 2L, 2L))
    # Python permute (0, 2, 4, 1, 3, 5), 1-indexed here
    latents <- latents$permute(c(1L, 3L, 5L, 2L, 4L, 6L))
    latents$reshape(c(b, (h %/% 2L) * (w %/% 2L), ch * 4L))
}

#' Unpack a FLUX patch sequence back into latents
#'
#' Inverse of \code{flux_pack_latents}. Height and width are the target
#' image dimensions in pixels; the latent grid is derived via the VAE
#' scale factor and the 2x2 patch size. Reference:
#' FluxPipeline._unpack_latents.
#'
#' @param latents Tensor of shape [B, S, C_packed].
#' @param height,width Integers. Image height/width in pixels.
#' @param vae_scale_factor Integer. Spatial downsampling of the VAE (8).
#'
#' @return Tensor of shape [B, C_packed / 4, height / 8, width / 8].
#'
#' @export
flux_unpack_latents <- function(latents, height, width, vae_scale_factor = 8L) {
    shape <- latents$shape
    b <- shape[1]
    ch <- shape[3]
    h <- 2L * (as.integer(height) %/% (vae_scale_factor * 2L))
    w <- 2L * (as.integer(width) %/% (vae_scale_factor * 2L))
    latents <- latents$view(c(b, h %/% 2L, w %/% 2L, ch %/% 4L, 2L, 2L))
    # Python permute (0, 3, 1, 4, 2, 5), 1-indexed here
    latents <- latents$permute(c(1L, 4L, 2L, 5L, 3L, 6L))
    latents$reshape(c(b, ch %/% 4L, h, w))
}
