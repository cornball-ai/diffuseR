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
Whatever the family generator returns, and the families do not
  agree: an image array for \code{flux1}, \code{flux2} and
  \code{zimage}, a video array for \code{ltx}, and for \code{sdxl} a
  list of \code{image} and \code{metadata}, because
  \code{\link{txt2img_sdxl}} has always returned that pair and changing
  it would break every existing caller. A broker that wants one shape
  should normalise in its own wrapper.
}
\description{
Dispatches to the family's generator with the resident pipeline
supplied, so no weights are re-read. The handle must be active.
}
