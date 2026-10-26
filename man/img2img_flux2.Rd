% tinyrox says don't edit this manually, but it can't stop you!
\name{img2img_flux2}
\alias{img2img_flux2}
\title{Edit an image with FLUX.2 klein}
\usage{
img2img_flux2(
  prompt,
  image,
  pipeline = NULL,
  width = NULL,
  height = NULL,
  num_inference_steps = 4L,
  max_sequence_length = 512L,
  seed = NULL,
  prompt_embeds = NULL,
  save_file = TRUE,
  filename = NULL,
  verbose = TRUE,
  ...
)
}
\arguments{
\item{prompt}{Character. The edit instruction or prompt.}

\item{image}{Path to a PNG/JPEG, or an [H, W, 3] array in [0, 1].}

\item{pipeline}{A \code{flux2_pipeline} loaded with
\code{encoder = TRUE}; NULL loads one (passing \code{...} through).}

\item{num_inference_steps}{Integer. Denoising steps (klein-4B: 4).}

\item{max_sequence_length}{Integer. Qwen3 token length (512).}

\item{seed}{Integer or NULL. See \code{\link{txt2img_flux2}}.}

\item{prompt_embeds}{Optional precomputed [B, S, 7680] embeddings.}

\item{save_file}{Logical. Write a PNG.}

\item{filename}{Output path (default derived from the prompt).}

\item{verbose}{Logical, or one of "silent", "progress", "steps".}

\item{...}{Passed to \code{\link{flux2_load_pipeline}} when
\code{pipeline} is NULL.}

\item{width,height}{Integers divisible by 16, or NULL (default) for
the processed reference image's size.}
}
\value{
Invisibly, \code{list(image, metadata)} where \code{image} is
  an [H, W, 3] array in [0, 1].
}
\description{
Image + prompt to image with the same klein-4B weights as
\code{\link{txt2img_flux2}}: the reference image is VAE-encoded once,
its tokens ride after the noisy latents in every transformer forward
(position ids at T = 10, see \code{\link{flux2_prepare_image_ids}}),
and only the latents' predictions drive the scheduler. Reference:
diffusers Flux2KleinPipeline with \code{image =} (Apache-2.0).
}
\details{
The image is checked (both sides at least 64 px, aspect ratio at most
8), resized with Lanczos to 1024 x 1024 in area if it is larger, and
center-cropped to multiples of 16. One reference image is supported.

}
