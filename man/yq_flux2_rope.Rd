% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_rope}
\alias{yq_flux2_rope}
\title{FLUX.2 Klein RoPE tables for a txt2img forward}
\usage{
yq_flux2_rope(text_len, height, width, theta = 2000, device = "cpu")
}
\arguments{
\item{text_len}{Integer. Text sequence length.}

\item{height}{Integer. Packed grid height (pixel height / 16).}

\item{width}{Integer. Packed grid width (pixel width / 16).}

\item{theta}{Numeric. Base frequency (2000).}

\item{device}{Character. Target device.}
}
\value{
List \code{list(cos, sin)}, each \code{[text_len + height*width, 128]}.
}
\description{
Builds the concatenated [text; image] position ids (text tokens carry
only the L/sequence axis; image latents carry H and W over the packed
grid) and returns the rotary tables. Reference:
\code{Flux2KleinPipeline._prepare_{text,latent}_ids}.
}
