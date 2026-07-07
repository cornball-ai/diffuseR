% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_tone_map_latents}
\alias{ltx23_tone_map_latents}
\title{Sigmoid tone mapping for latents}
\usage{
ltx23_tone_map_latents(latents, compression)
}
\arguments{
\item{latents}{Tensor.}

\item{compression}{Numeric in [0, 1].}
}
\value{
Tone-mapped latents.
}
\description{
Compresses the latent dynamic range (cf. diffusers
\code{tone_map_latents}). \code{compression} 0 is identity, 1 is the
full effect.
}
