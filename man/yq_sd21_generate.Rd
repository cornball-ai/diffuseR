% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sd21_generate}
\alias{yq_sd21_generate}
\title{Stable Diffusion 2.1 end-to-end generation (anvl)}
\usage{
yq_sd21_generate(prompt = NULL, w_clip, w_unet, w_vae = NULL,
                 negative_prompt = "", ids = NULL, uncond_ids = NULL,
                 noise = NULL, latent_dim = 96L, num_inference_steps = 50L,
                 guidance_scale = 7.5, seed = NULL, decode = !is.null(w_vae),
                 device = "cpu", precision = "highest", clip_fn = NULL,
                 unet_fn = NULL)
}
\arguments{
\item{prompt}{Character. Prompt (ignored if \code{ids} given).}

\item{w_clip}{CLIP weights pytree (\code{\link{yq_clip_load_weights}}).}

\item{w_unet}{UNet weights pytree (\code{\link{yq_sd_unet_load_weights}}).}

\item{w_vae}{Optional VAE weights pytree
(\code{\link{yq_sd_vae_load_weights}}); required to decode pixels.}

\item{negative_prompt}{Character. Unconditional prompt (default "").}

\item{noise}{Optional AnvlArray \code{[1, 4, latent_dim, latent_dim]}
initial latents; supply the reference's noise for parity.}

\item{latent_dim}{Integer. Latent H/W (image px / 8).}

\item{num_inference_steps}{Integer. DDIM steps.}

\item{guidance_scale}{Numeric. CFG scale (SD 2.1 default 7.5).}

\item{seed}{Optional integer seed for the anvl RNG fallback.}

\item{decode}{Logical. Decode latents to pixels (needs \code{w_vae}).}

\item{device}{Character. Target device.}

\item{precision}{Character. Matmul precision for CLIP.}

\item{ids,uncond_ids}{Optional integer matrices \code{[1, S]} of 0-based
CLIP ids overriding tokenization (for parity fixtures).}

\item{clip_fn,unet_fn}{Optional pre-jitted closures (built from
\code{\link{yq_clip_encoder}} / \code{\link{yq_sd_unet}} if NULL).}
}
\value{
List with \code{latents} (final AnvlArray \code{[1, 4, H, W]}),
  \code{pixels} (AnvlArray \code{[1, 3, 8H, 8W]} in [-1, 1] or NULL),
  and the \code{cond_embed} / \code{uncond_embed} CLIP states.
}
\description{
Runs the full text-to-image pipeline: tokenize (or take explicit ids)
-> CLIP encode conditional + unconditional prompts -> CFG DDIM denoise
loop over the UNet -> (optionally) VAE decode to pixels. The CLIP
encoder and UNet are each \code{anvl::jit()}ed once and reused across
steps; the UNet is called twice per step (unconditional, then
conditional) and combined with classifier-free guidance
\code{uncond + guidance_scale * (cond - uncond)}. Feed \code{noise}
(and \code{ids} / \code{uncond_ids}) for bit-for-bit parity with a torch
reference; the RNG fallback does not match torch.
}
