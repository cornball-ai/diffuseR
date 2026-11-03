% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_load_upsampler}
\alias{ltx23_load_upsampler}
\title{Load LTX latent upscaler weights}
\usage{
ltx23_load_upsampler(path, device = "cuda", dtype = "bfloat16", verbose = TRUE)
}
\arguments{
\item{path}{Path to e.g. \code{ltx-2.3-spatial-upscaler-x2-1.1.safetensors},
\code{ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors} or
\code{ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors}.}

\item{verbose}{Logical.}

\item{device,dtype}{Placement for the loaded model.}
}
\value{
The loaded \code{ltx23_latent_upsampler}.
}
\description{
Loads the LTX-2.3 spatial x2 upscaler and the LTX-2.5 spatial and
temporal x2 upscalers. The layout (spatial or temporal, channel
width, depth) is read from the tensor shapes; the checkpoint keys
match this module tree directly.
}
