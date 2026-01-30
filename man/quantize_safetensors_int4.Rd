% tinyrox says don't edit this manually, but it can't stop you!
\name{quantize_safetensors_int4}
\alias{quantize_safetensors_int4}
\title{Quantize Safetensor Weights to INT4}
\usage{
quantize_safetensors_int4(paths, output_path, block_size = 64L, verbose = TRUE)
}
\arguments{
\item{paths}{Character vector. Paths to safetensor files.}

\item{output_path}{Character. Path to save INT4 weights (.safetensors).}

\item{block_size}{Integer. Block size for quantization (default 64).}

\item{verbose}{Logical. Print progress.}
}
\value{
Invisible NULL. Writes quantized weights to output_path.
}
\description{
Loads weights from safetensors file(s) and quantizes to INT4.
Useful for quantizing large models without loading the full module.
}
\examples{
\dontrun{
# Quantize sharded transformer weights
paths <- list.files("~/.cache/huggingface/.../transformer",
                    pattern = "safetensors$", full.names = TRUE)
quantize_safetensors_int4(paths, "dit_int4.safetensors")
}
}
