% tinyrox says don't edit this manually, but it can't stop you!
\name{reshard_safetensors}
\alias{reshard_safetensors}
\title{Re-shard a large safetensors file into sub-2 GB shards}
\usage{
reshard_safetensors(input, output_dir, base = "diffusion_pytorch_model",
                    shard_bytes = 1.9e+09, verbose = TRUE)
}
\arguments{
\item{input}{Path to the source \code{.safetensors} file, or a
directory containing \code{<base>.safetensors}.}

\item{output_dir}{Output directory for the shards + index.}

\item{base}{Shard basename (default
\code{"diffusion_pytorch_model"}).}

\item{shard_bytes}{Target shard size; the default 1.9e9 keeps each
shard under the ~2.15 GB ceiling.}

\item{verbose}{Logical.}
}
\value{
Invisibly, the path to the written index.json.
}
\description{
Splits a single \code{.safetensors} file into diffusers-style shards
(\code{<base>-00001-of-000NN.safetensors} plus a
\code{<base>.safetensors.index.json} weight map) so it loads on stock
CRAN safetensors, which overflows a 32-bit offset on any file at or
above 2^31 bytes. Reading the oversize source requires a
fork-patched safetensors (a build-machine step); the shards it writes
are fork-free to read. Used to host large fp16 diffusers weights
(e.g. the 5 GB SDXL UNet) unchanged, without quantization.
}
