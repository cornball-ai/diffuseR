% tinyrox says don't edit this manually, but it can't stop you!
\name{img2img_flux3}
\alias{img2img_flux3}
\title{Edit an image with FLUX 3 Image Klein}
\usage{
img2img_flux3(
  prompt,
  image,
  pipeline = NULL,
  width = NULL,
  height = NULL,
  max_area = 4 * 1024 * 1024,
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
\item{prompt}{Character. The full caption of the edited image.}

\item{image}{Path to a PNG/JPEG, or an [H, W, 3] array in [0, 1].}

\item{pipeline}{A \code{flux3_pipeline} loaded with
\code{encoder = TRUE}; NULL loads one (passing \code{...} through).}

\item{max_area}{Numeric. Reference canvas area cap (reference: 4
megapixels). The reference adds \code{area / 256} tokens to every
forward, so lower it on small cards.}

\item{num_inference_steps}{Integer. Denoising steps (16).}

\item{max_sequence_length}{Integer. Qwen3-VL token cap (8192).}

\item{seed}{Integer or NULL. See \code{\link{txt2img_flux3}}.}

\item{prompt_embeds}{Optional precomputed [1, S, 20480] embeddings.}

\item{save_file}{Logical. Write a PNG.}

\item{filename}{Output path (default derived from the prompt).}

\item{verbose}{Logical, or one of "silent", "progress", "steps".}

\item{...}{Passed to \code{\link{flux3_load_pipeline}} when
\code{pipeline} is NULL.}

\item{width,height}{Integers divisible by 16, or NULL (default) for
the reference canvas, cropped back to the reference afterwards.}
}
\value{
Invisibly, \code{list(image, metadata)} where \code{image} is
  an [H, W, 3] array in [0, 1].
}
\description{
Image + caption to image with the same weights as
\code{\link{txt2img_flux3}}. The reference is kept whole (see
\code{\link{flux3_preprocess_image}}): scaled so its canvas fits
\code{max_area}, reflect-padded to the 16 px grid, VAE-encoded once,
and passed to every transformer forward as its own token mode
(position ids at L = -1). The output defaults to the canvas size and
is cropped back to the reference's box. Reference: diffusers
Flux3DistilledModularPipeline with \code{image =} (Apache-2.0).
}
\details{
Write the caption as a full description of the edited result (the
reference with the edit folded in); a one-line instruction reframes
the shot. BFL's prompt upsampler (flux-3-image-klein-pup) writes such
captions from an instruction plus the reference image.

}
