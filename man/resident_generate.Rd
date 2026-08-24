% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_generate}
\alias{resident_generate}
\title{Generate from an active resident pipeline}
\usage{
resident_generate(res, prompt, ...)
}
\arguments{
\item{res}{A \code{diffuseR_resident} handle.}

\item{prompt}{Character. The text prompt.}

\item{...}{Passed to \code{\link{txt2img_flux}},
\code{\link{txt2img_flux2}}, \code{\link{txt2img_zimage}},
\code{\link{txt2vid_ltx2}} or \code{\link{txt2img_sdxl}}. For
\code{sdxl} the handle supplies \code{devices} matching its own
placement unless the caller names it.}
}
\value{
Whatever the family generator returns, which is always a list.

  The five image families (\code{flux1}, \code{flux2}, \code{zimage},
  \code{sdxl}, \code{sd21}) return \code{list(image, metadata)}, where
  \code{image} is an [H, W, 3] array in [0, 1], so a caller unwraps
  \code{$image} uniformly across all five.

  \code{ltx} returns \code{latents}, \code{audio_latents},
  \code{latent_shape} and \code{sample_rate}, plus \code{video} and
  \code{audio} -- but those two are produced only when
  \code{decode_video} and \code{decode_audio} are TRUE, which they are
  by default. A caller that turns either off gets a list without that
  field rather than a NULL one, so index it with \code{[[ ]]} and check,
  the way \code{\link{txt2vid_ltx2}} does internally.

  Only the visibility differs: \code{\link{txt2img_sdxl}} and
  \code{\link{txt2img_sd21}} use \code{return()} while the other three
  image families and \code{ltx} use \code{invisible()}, which affects
  auto-printing at the console and nothing else.
}
\description{
Dispatches to the family's generator with the resident pipeline
supplied, so no weights are re-read. The handle must be active.
}
