% tinyrox says don't edit this manually, but it can't stop you!
\name{quantize_model_int4}
\alias{quantize_model_int4}
\title{Quantize Model Weights to INT4}
\usage{
quantize_model_int4(module, block_size = 64L, verbose = TRUE)
}
\arguments{
\item{module}{nn_module. The model to quantize.}

\item{block_size}{Integer. Block size for quantization.}

\item{verbose}{Logical. Print progress.}
}
\value{
List of quantized parameters (does not modify original module).
}
\description{
Quantizes all parameters in a torch module to INT4 format.
}
