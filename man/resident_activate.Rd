% tinyrox says don't edit this manually, but it can't stop you!
\name{resident_activate}
\alias{resident_activate}
\title{Bring a resident pipeline onto the GPU}
\usage{
resident_activate(res)
}
\arguments{
\item{res}{A \code{diffuseR_resident} handle.}
}
\value{
Invisibly the handle, with state "active".
}
\description{
Copies every pinned component to the handle's bound device by DMA and
verifies the result tensor-by-tensor. A failure rolls back to the
pinned host state; a rollback that cannot itself be verified leaves
the handle broken.
}
\details{
What activation does depends on how the pipeline was loaded:

\itemize{
\item \code{phase_offload = TRUE} (the default, and what the
\code{txt2img_*} functions expect): no bulk transfer. The render
moves each component on as its phase begins and back off as it
ends, from these same pinned copies, so pre-loading them would be
undone within one phase. Activation is the ownership claim.
\item \code{phase_offload = FALSE}: every component is copied to the
card up front and stays there across renders. This is the fast
path, and it is checked against free VRAM first.
}

The distinction is not cosmetic. FLUX.1's pinned set is 15.73 GB,
which does not fit a 15.47 GiB card -- bulk-onloading it OOMs even
though the phased render fits comfortably. So \code{state} is a claim
about who owns the card, not a measurement of what is on it; read
\code{components_on_gpu} from \code{\link{resident_status}} for the
measurement.

}
