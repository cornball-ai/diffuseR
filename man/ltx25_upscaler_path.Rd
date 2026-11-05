% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx25_upscaler_path}
\alias{ltx25_upscaler_path}
\title{Locate (or download) an LTX-2.5 latent upscaler file}
\usage{
ltx25_upscaler_path(which = c("spatial", "temporal"), download = FALSE)
}
\arguments{
\item{which}{"spatial" or "temporal".}

\item{download}{Logical. Download the file when it is not cached
(asks first in interactive sessions; set
\code{options(diffuseR.consent = TRUE)} otherwise).}
}
\value{
The local file path, for \code{\link{ltx23_load_upsampler}}.
}
\description{
LTX-2.5 ships a spatial x2 upscaler (1.0 GB) and a temporal x2
upscaler (0.26 GB) under \code{latent_upscale_models/} in the gated
\code{Lightricks/LTX-2.5} repository: accept the LTX-2 Community
License on Hugging Face and set \code{HF_TOKEN}. The weights are read
from the Hugging Face cache and never redistributed.
}
