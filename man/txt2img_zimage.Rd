% tinyrox says don't edit this manually, but it can't stop you!
\name{txt2img_zimage}
\alias{txt2img_zimage}
\title{Generate an image with Z-Image-Turbo}
\usage{
txt2img_zimage(prompt, pipeline = NULL, width = 1024L, height = 1024L,
               num_inference_steps = 8L, max_sequence_length = 512L,
               seed = NULL, prompt_embeds = NULL, save_file = TRUE,
               filename = NULL, verbose = TRUE, ...)
}
\arguments{
\item{prompt}{Character. The prompt.}

\item{pipeline}{A \code{zimage_pipeline} from
\code{\link{zimage_load_pipeline}}; NULL loads one (passing
\code{...} through).}

\item{num_inference_steps}{Integer. Denoising steps (Turbo: 8).}

\item{max_sequence_length}{Integer. Qwen3 token length (512).}

\item{seed}{Integer or NULL. Latents are drawn on the CPU, so a seed
matches a Python diffusers run with a CPU generator.}

\item{prompt_embeds}{Optional precomputed [L, 2560] caption
embeddings (valid tokens only).}

\item{save_file}{Logical. Write a PNG.}

\item{filename}{Output path (default derived from the prompt).}

\item{verbose}{Logical.}

\item{...}{Passed to \code{\link{zimage_load_pipeline}} when
\code{pipeline} is NULL.}

\item{width,height}{Integers, divisible by 16.}
}
\value{
Invisibly, \code{list(image, metadata)} where \code{image} is
  an [H, W, 3] array in [0, 1].
}
\description{
Guidance-distilled text-to-image (8 steps, no CFG): Qwen3-4B prompt
encoding (thinking-enabled chat template, penultimate hidden state),
FlowMatch denoising with the reversed-timestep convention, and
16-channel VAE decode. Strong at legible text rendering, English and
Chinese both.
}
