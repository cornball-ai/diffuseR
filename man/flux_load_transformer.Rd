% tinyrox says don't edit this manually, but it can't stop you!
\name{flux_load_transformer}
\alias{flux_load_transformer}
\title{Load a FLUX transformer from any checkpoint format}
\usage{
flux_load_transformer(ckpt, device = "cuda", dtype = "bfloat16", pin = TRUE,
                      verbose = TRUE, ...)
}
\arguments{
\item{ckpt}{A checkpoint from \code{\link{flux_open_checkpoint}} or
\code{\link{flux_open_quantized}}.}

\item{device}{Character. Compute device.}

\item{dtype}{Character. Model dtype ("bfloat16" or "float32"). For
quantized formats this sets the resident (non-quantized) tensors
and must match the compute dtype: bfloat16 for GPU compute,
float32 for CPU compute.}

\item{pin}{Logical. Pin fp8 host memory for faster transfers.}

\item{verbose}{Logical.}

\item{...}{Overrides for \code{\link{flux_transformer}} arguments
(tiny test configs).}
}
\value{
The loaded \code{flux_transformer} in eval mode.
}
\description{
Builds \code{\link{flux_transformer}} from the checkpoint's embedded
config and loads the weights. Dispatches on the checkpoint format:
}
\details{
\itemize{
\item full precision (\code{\link{flux_open_checkpoint}}): weights
stream into the model in \code{dtype} on \code{device}.
\item \code{"nf4"} (\code{\link{flux_open_quantized}}): cast-set
linears become \code{ltx23_nf4_linear}; the whole model (packed
weights included) moves to \code{device} and stays resident.
\item \code{"fp8"}: cast-set linears become
\code{ltx23_fp8_linear}; fp8 weights stay CPU-resident (optionally
pinned) and stream to \code{device} inside each forward.
}

}
