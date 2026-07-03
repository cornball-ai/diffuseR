% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_upsampler}
\alias{ltx23_load_upsampler}
\title{Load the LTX-2.3 spatial upscaler weights}
\usage{
ltx23_load_upsampler(path, device = "cuda", dtype = "bfloat16", verbose = TRUE)
}
\arguments{
\item{path}{Path to e.g. \code{ltx-2.3-spatial-upscaler-x2-1.1.safetensors}.}

\item{verbose}{Logical.}

\item{device,dtype}{Placement for the loaded model.}
}
\value{
The loaded \code{ltx23_latent_upsampler}.
}
\description{
The checkpoint keys match this module tree directly.
}
