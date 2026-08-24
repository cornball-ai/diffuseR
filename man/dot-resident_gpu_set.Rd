% tinyrox says don't edit this manually, but it can't stop you!
\name{.resident_gpu_set}
\alias{.resident_gpu_set}
\title{Which components a bulk activation puts on the card}
\usage{
.resident_gpu_set(res)
}
\arguments{
\item{res}{A resident handle.}
}
\value{
Character vector of \code{res$staging} names.
}
\description{
All of them, unless the family says otherwise.
}
\details{
SDXL says otherwise. Its four components are only 8.0 GB pinned, so
bulk-onloading the set looks affordable on a 16 GB card -- and then the
VAE decode OOMs, because SDXL decodes 1024x1024 in float32 and that peak
arrives while the UNet is still resident. Measured: 8.0 GB of weights
plus the decode phase reached 14.38 GiB of 15.47 GiB and died asking for
another 512 MiB. A 12 GB card never had a chance.

So SDXL puts only the UNet on the card and computes the text encode and
the decode on the host from the same pinned copies. That is the placement
\code{\link{auto_devices}} already recommends for this model at this tier;
residency's contribution is that the 5 GB UNet stops being re-read from
disk between models.

}
\keyword{internal}
