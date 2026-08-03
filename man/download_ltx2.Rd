% tinyrox says don't edit this manually, but it can't stop you!
\name{download_ltx2}
\alias{download_ltx2}
\title{Download the LTX-2.3 checkpoint and build a quantized artifact}
\usage{
download_ltx2(quantize = TRUE, precision = c("nf4", "fp8"), output_dir = NULL,
              text_encoder = TRUE, verbose = TRUE)
}
\arguments{
\item{quantize}{Logical. Build the quantized artifact after downloading.}

\item{precision}{"nf4" (~19 GB, readable by every safetensors) or
"fp8" (~26 GB, needs float8 write support).}

\item{output_dir}{Directory for the artifact. NULL derives it from
\code{precision}.}

\item{text_encoder}{Logical. Also fetch the Gemma3 text encoder and
tokenizer (~25 GB, shared with LTX-2.0; from the Lightricks/LTX-2
repo).}

\item{verbose}{Logical.}
}
\value{
Invisibly, a list with \code{checkpoint} (source path or NULL),
  \code{artifact_dir}, \code{precision}, \code{text_encoder_dir}, and
  \code{fp8_dir} for back-compatibility -- the artifact directory when
  \code{precision} is "fp8", NULL otherwise, since a field named
  \code{fp8_dir} pointing at an nf4 artifact would be a trap.
}
\description{
Skips work that is already done: a valid manifest short-circuits
everything; a cached 46 GB source skips the download. The source file
may be deleted after quantization (it is never removed automatically).
}
\details{
Both quantized tiers are buildable here. \code{\link{recommend}}
returns nf4 for LTX on any card with 14 GB or more (it prefers nf4 at
1280 px over fp8 at 1024 px, since video trades weight precision for
resolution), so nf4 is the tier most users want. fp8 additionally
needs a safetensors that can \emph{write} float8; asking for it
without one warns and builds nf4 instead rather than failing inside
the quantizer.

}
