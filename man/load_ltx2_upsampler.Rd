% tinyrox says don't edit this manually, but it can't stop you!
\name{load_ltx2_upsampler}
\alias{load_ltx2_upsampler}
\title{Load LTX-2 Spatial Upsampler}
\usage{
load_ltx2_upsampler(
  weights_path,
  device = "cpu",
  dtype = "float32",
  verbose = TRUE
)
}
\arguments{
\item{weights_path}{Character. Path to safetensors weight file.}

\item{device}{Character. Target device ("cpu" or "cuda").}

\item{dtype}{Character. Target dtype ("float32", "float16", or "bfloat16").}

\item{verbose}{Logical. Print progress.}
}
\value{
A \code{latent_upsampler} nn_module with loaded weights.
}
\description{
Loads the latent upsampler model from a safetensors file.
}
