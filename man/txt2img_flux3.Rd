% tinyrox says don't edit this manually, but it can't stop you!
\name{txt2img_flux3}
\alias{txt2img_flux3}
\title{Generate an image with FLUX 3 Image Klein}
\usage{
txt2img_flux3(
  prompt,
  pipeline = NULL,
  width = 1024L,
  height = 1024L,
  num_inference_steps = 16L,
  max_sequence_length = 8192L,
  seed = NULL,
  prompt_embeds = NULL,
  save_file = TRUE,
  filename = NULL,
  verbose = TRUE,
  ...
)
}
\arguments{
\item{prompt}{Character. The (full) caption.}

\item{pipeline}{A \code{flux3_pipeline} from
\code{\link{flux3_load_pipeline}}; NULL loads one (passing
\code{...} through).}

\item{num_inference_steps}{Integer. Denoising steps (16).}

\item{max_sequence_length}{Integer. Qwen3-VL token cap (8192).}

\item{seed}{Integer or NULL. Latents are drawn on the CPU in float32
in the packed shape, so a seed matches a diffusers run with a CPU
generator.}

\item{prompt_embeds}{Optional precomputed [1, S, 20480] embeddings.}

\item{save_file}{Logical. Write a PNG.}

\item{filename}{Output path (default derived from the prompt).}

\item{verbose}{Logical, or one of "silent", "progress", "steps".}

\item{...}{Passed to \code{\link{flux3_load_pipeline}} when
\code{pipeline} is NULL.}

\item{width,height}{Integers, divisible by 16.}
}
\value{
Invisibly, \code{list(image, metadata)} where \code{image} is
  an [H, W, 3] array in [0, 1].
}
\description{
Step-distilled text-to-image (16 steps, no guidance): Qwen3-VL prompt
encoding, flow-map FlowMatch denoising with the static shift 6.93, and
32-channel VAE decode through the BatchNorm latent statistics.
}
\details{
FLUX 3 takes full scene captions: a short prompt gives generic
output. Expand short requests into one detailed caption (framing,
subjects with colors, materials and positions, background, lighting)
with an LLM first. Layout boxes go in the caption as BFL documents
(a JSON list of elements with \code{bbox = [top, left, bottom,
right]} on a 0 to 1000 grid).

}
