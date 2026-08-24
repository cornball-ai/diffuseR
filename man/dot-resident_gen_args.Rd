% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_gen_args}
\alias{.resident_gen_args}
\title{Family fixups for a resident generate call}
\usage{
.resident_gen_args(res, args)
}
\arguments{
\item{res}{A resident handle.}

\item{args}{The caller's \code{...}, as a list. An explicit
\code{devices} wins -- this fills a gap, it does not override.}
}
\value{
\code{args}, possibly with \code{devices} added.
}
\description{
Split out so the SDXL device injection can be asserted without running a
multi-gigabyte generation.
}
\details{
\code{\link{txt2img_sdxl}} does not read the pipeline's placement. With
its default \code{devices = "auto"} it calls \code{\link{auto_devices}}
afresh and moves the prompt embeds to whatever THAT returns. On a 12 GB
card auto can answer "unet on cuda, encoders on cpu", which contradicts a
bulk-activated handle whose encoders are on the card, and the text
encoder call then dies on a device mismatch. The handle knows where its
components actually are, so it says so instead of letting the generator
re-decide.

Only SDXL needs this: the other families phase-offload from their own
pinned copies and place each component themselves as its phase begins.

}
\keyword{internal}
