% tinyrox says don't edit this manually, but it can't stop you!
\name{quantize_int4}
\alias{quantize_int4}
\title{Quantize Tensor to INT4}
\usage{
quantize_int4(x, block_size = 64L)
}
\arguments{
\item{x}{Tensor. Input float tensor.}

\item{block_size}{Integer. Number of values per scale factor (default 64).}
}
\value{
A list with:
  - `packed`: uint8 tensor with packed INT4 values
  - `scales`: float tensor with per-block scale factors
  - `orig_shape`: original tensor shape
  - `orig_numel`: original number of elements
  - `block_size`: block size used
}
\description{
Quantizes a float tensor to 4-bit integers with block-wise scaling.
Two INT4 values are packed per byte for 7-8x compression.
}
\details{
INT4 range is -8 to 7. Values are scaled per block, quantized, shifted to
unsigned (0-15), and packed two per byte. Compression is ~7x vs float32,
~3.5x vs float16. Typical reconstruction error is 10-12\% of std.
}
\examples{
\dontrun{
x <- torch_randn(c(4096, 4096)) * 0.02
q <- quantize_int4(x)
x_back <- dequantize_int4(q)
}
}
