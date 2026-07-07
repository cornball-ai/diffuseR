% tinyrox says don't edit this manually, but it can't stop you!
\name{txt2img_flux2}
\alias{txt2img_flux2}
\title{Generate an image with FLUX.2 klein}
\usage{
txt2img_flux2(prompt, pipeline = NULL, width = 1024L, height = 1024L,
              num_inference_steps = 4L, max_sequence_length = 512L,
              seed = NULL, prompt_embeds = NULL, save_file = TRUE,
              filename = NULL, verbose = TRUE, ...)
}
\arguments{
\item{prompt}{Character. The prompt.}

\item{pipeline}{A \code{flux2_pipeline} from
\code{\link{flux2_load_pipeline}}; NULL loads one (passing
\code{...} through).}

\item{num_inference_steps}{Integer. Denoising steps (klein-4B: 4).}

\item{max_sequence_length}{Integer. Qwen3 token length (512).}

\item{seed}{Integer or NULL. Latents are drawn on the CPU in the
packed shape, so a seed matches a Python diffusers run with a CPU
generator.}

\item{prompt_embeds}{Optional precomputed [B, S, 7680] embeddings.}

\item{save_file}{Logical. Write a PNG.}

\item{filename}{Output path (default derived from the prompt).}

\item{verbose}{Logical.}

\item{...}{Passed to \code{\link{flux2_load_pipeline}} when
\code{pipeline} is NULL.}

\item{width,height}{Integers, divisible by 16.}
}
\value{
Invisibly, \code{list(image, metadata)} where \code{image} is
  an [H, W, 3] array in [0, 1].
}
\description{
Step-distilled text-to-image (klein-4B: 4 steps, no guidance): Qwen3
prompt encoding (chat template, mid-stack hidden states), FlowMatch
denoising with the empirical dynamic shift, and 32-channel VAE decode
through the BatchNorm latent statistics.
}
