% tinyrox says don't edit this manually, but it can't stop you!
\name{load_ltx2_vae}
\alias{load_ltx2_vae}
\title{Load LTX2 Video VAE from safetensors}
\usage{
load_ltx2_vae(
  weights_path,
  config_path = NULL,
  device = "cpu",
  dtype = "float32",
  verbose = TRUE
)
}
\arguments{
\item{weights_path}{Character. Path to safetensors file or directory containing weights.}

\item{config_path}{Character. Optional path to config.json. If NULL and weights_path
is a directory, looks for config.json in that directory. Otherwise uses default config.}

\item{device}{Character. Device to load weights to. Default: "cpu"}

\item{dtype}{Character or torch dtype. Data type. Default: "float32"}

\item{verbose}{Logical. Print loading progress. Default: TRUE}
}
\value{
Initialized ltx2_video_vae module
}
\description{
Load pre-trained LTX2 VAE weights from a HuggingFace safetensors file.
}
