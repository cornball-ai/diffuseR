% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_generate}
\alias{yq_sdxl_generate}
\title{Stable Diffusion XL end-to-end generation (anvl)}
\usage{
yq_sdxl_generate(w_unet, w_clipl, w_bigg, w_vae = NULL, prompt = NULL,
                 ids = NULL, eos_index = NULL, noise = NULL, img_dim = 128L,
                 latent_dim = NULL, time_ids = NULL, num_inference_steps = 4L,
                 guidance_scale = 7.5, decode = !is.null(w_vae),
                 device = "cpu", precision = "highest", unet_fn = NULL)
}
\arguments{
\item{w_unet}{UNet weights pytree (\code{\link{yq_sdxl_unet_load_weights}}).}

\item{w_vae}{Optional VAE weights pytree
(\code{\link{yq_sdxl_vae_load_weights}}); required to decode pixels.}

\item{prompt}{Character. Prompt (ignored if \code{ids} given).}

\item{ids}{Optional integer matrix \code{[1, S]} of 0-based CLIP ids
overriding tokenization (for parity fixtures). Shared by both encoders.}

\item{eos_index}{Optional integer. 1-based EOS position for the pooled
gather (default \code{which.max(ids)}).}

\item{noise}{Optional AnvlArray \code{[1, 4, latent_dim, latent_dim]}
initial latents; supply the reference's noise for parity.}

\item{img_dim}{Integer. Image side in pixels (also the SDXL time-ids
original/target size).}

\item{latent_dim}{Integer. Latent H/W (default \code{img_dim / 8}).}

\item{time_ids}{Optional numeric length-6 SDXL micro-conditioning
\code{c(orig_h, orig_w, crop_top, crop_left, target_h, target_w)}
(default \code{c(img_dim, img_dim, 0, 0, img_dim, img_dim)}).}

\item{num_inference_steps}{Integer. DDIM steps.}

\item{guidance_scale}{Numeric. CFG scale (SDXL default 7.5).}

\item{decode}{Logical. Decode latents to pixels (needs \code{w_vae}).}

\item{device}{Character. Target device.}

\item{precision}{Character. Matmul precision for CLIP.}

\item{unet_fn}{Optional pre-jitted UNet closure (built from
\code{\link{yq_sdxl_unet}} if NULL).}

\item{w_clipl,w_bigg}{CLIP-L / bigG weights pytrees
(\code{\link{yq_sdxl_clip_load_weights}}).}
}
\value{
List with \code{latents} (final AnvlArray \code{[1, 4, H, W]}),
  \code{pixels} (AnvlArray \code{[1, 3, 8H, 8W]} in [-1, 1] or NULL),
  the \code{context} / \code{pooled} CLIP conditioning, and \code{step1}
  (the step-1 \code{noise_cond} / \code{noise_uncond} / \code{latents},
  for isolating the CFG + scheduler wiring in parity tests).
}
\description{
Runs the full text-to-image pipeline: tokenize (or take explicit ids)
-> dual CLIP encode (concatenated penultimate context + bigG pooled) ->
build the SDXL added text-time conditioning (pooled + sinusoidal
time-ids) -> CFG DDIM denoise loop over the UNet -> (optionally) VAE
decode to pixels. The dual CLIP encoders and the UNet are each
\code{anvl::jit()}ed once and reused; the UNet is called twice per step
(conditional, then unconditional) and combined with classifier-free
guidance \code{uncond + guidance_scale * (cond - uncond)}. The
unconditional pass uses zeroed context and pooled embeds (the
\code{negative_prompt = NULL} path of \code{txt2img_sdxl}). Feed
\code{noise} (and \code{ids}) for bit-for-bit parity with a torch
reference.
}
