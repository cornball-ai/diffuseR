% tinyrox says don't edit this manually, but it can't stop you!
\name{download_ltx2}
\alias{download_ltx2}
\title{Download the LTX-2.3 checkpoint and build the fp8 artifact}
\usage{
download_ltx2(quantize = TRUE,
              output_dir = file.path(tools::R_user_dir("diffuseR", "data"), "ltx2.3-fp8"),
              text_encoder = TRUE, verbose = TRUE)
}
\arguments{
\item{quantize}{Logical. Build the fp8 artifact after downloading.}

\item{output_dir}{Directory for the fp8 artifact.}

\item{text_encoder}{Logical. Also fetch the Gemma3 text encoder and
tokenizer (~25 GB, shared with LTX-2.0; from the Lightricks/LTX-2
repo).}

\item{verbose}{Logical.}
}
\value{
Invisibly, a list with \code{checkpoint} (source path or NULL),
  \code{fp8_dir}, and \code{text_encoder_dir}.
}
\description{
Skips work that is already done: a valid fp8 manifest short-circuits
everything; a cached 46 GB source skips the download. The source file
may be deleted after quantization (it is never removed automatically).
}
