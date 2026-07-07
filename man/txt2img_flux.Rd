% tinyrox says don't edit this manually, but it can't stop you!
\name{txt2img_flux}
\alias{txt2img_flux}
\title{Generate an image with FLUX.1-schnell}
\usage{
txt2img_flux(prompt, pipeline = NULL, width = 1024L, height = 1024L,
             num_inference_steps = 4L, max_sequence_length = 256L, seed = NULL,
             prompt_embeds = NULL, pooled_prompt_embeds = NULL,
             save_file = TRUE, filename = NULL, verbose = TRUE, ...)
}
\arguments{
\item{prompt}{Character. The prompt.}

\item{pipeline}{A \code{flux_pipeline} from
\code{\link{flux_load_pipeline}}; NULL loads one (passing
\code{...} through).}

\item{num_inference_steps}{Integer. Denoising steps (schnell: 4).}

\item{max_sequence_length}{Integer. T5 token length (schnell: 256).}

\item{seed}{Integer or NULL. Initial latents are drawn on the CPU, so
a seed matches a Python diffusers run with a CPU generator.}

\item{save_file}{Logical. Write a PNG.}

\item{filename}{Output path (default derived from the prompt).}

\item{verbose}{Logical.}

\item{...}{Passed to \code{\link{flux_load_pipeline}} when
\code{pipeline} is NULL.}

\item{width,height}{Integers, divisible by 16.}

\item{prompt_embeds,pooled_prompt_embeds}{Optional precomputed text
embeddings (skip the text encoders).}
}
\value{
Invisibly, \code{list(image, metadata)} where \code{image} is
  an [H, W, 3] array in [0, 1].
}
\description{
4-step distilled text-to-image generation (no classifier-free
guidance): T5 + CLIP prompt encoding, flow-matching Euler denoising
over the packed latent sequence, and 16-channel VAE decode. With
phase offloading each component is the sole GPU tenant for its phase.
}
