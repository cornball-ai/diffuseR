% tinyrox says don't edit this manually, but it can't stop you!
\name{load_ltx2_connectors}
\alias{load_ltx2_connectors}
\title{Load LTX2 Text Connectors from safetensors}
\usage{
load_ltx2_connectors(
  weights_path,
  config_path = NULL,
  text_proj_path = NULL,
  device = "cpu",
  dtype = "float32",
  verbose = TRUE
)
}
\arguments{
\item{weights_path}{Character. Path to safetensors file.}

\item{config_path}{Character. Optional path to config.json.}

\item{device}{Character. Device to load weights to. Default: "cpu"}

\item{dtype}{Character. Data type ("float32", "float16"). Default: "float32"}

\item{verbose}{Logical. Print loading progress. Default: TRUE}
}
\value{
Initialized ltx2_text_connectors module
}
\description{
Load pre-trained LTX2 connector weights from HuggingFace safetensors file.
}
