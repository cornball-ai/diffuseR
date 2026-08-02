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
\code{\link{txt2img_flux2}}, \code{\link{txt2img_zimage}} or
\code{\link{txt2vid_ltx2}}.}
}
\value{
Whatever the family generator returns: an image array for the
  image families, a video array for \code{ltx}.
}
\description{
Dispatches to the family's generator with the resident pipeline
supplied, so no weights are re-read. The handle must be active.
}
