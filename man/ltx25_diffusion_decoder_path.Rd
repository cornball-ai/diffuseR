% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx25_diffusion_decoder_path}
\alias{ltx25_diffusion_decoder_path}
\title{Locate (or download) the LTX-2.5 diffusion decoder file}
\usage{
ltx25_diffusion_decoder_path(download = FALSE)
}
\arguments{
\item{download}{Logical. Download the file when it is not cached
(asks first in interactive sessions; set
\code{options(diffuseR.consent = TRUE)} otherwise).}
}
\value{
The local file path.
}
\description{
The decoder ships as LTX-2.5's default video VAE,
\code{vae/ltx-2.5-video-vae-bf16.safetensors} (1.5 GB) in the gated
\code{Lightricks/LTX-2.5} repository: accept the LTX-2 Community
License on Hugging Face and set \code{HF_TOKEN}. The weights are read
from the Hugging Face cache and never redistributed.
}
