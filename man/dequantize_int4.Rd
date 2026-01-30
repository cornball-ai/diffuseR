% tinyrox says don't edit this manually, but it can't stop you!
\name{dequantize_int4}
\alias{dequantize_int4}
\title{Dequantize INT4 Tensor}
\usage{
dequantize_int4(q, dtype = torch::torch_float16(), device = "cpu")
}
\arguments{
\item{q}{List. Quantized data from `quantize_int4()`.}

\item{dtype}{Torch dtype. Output dtype (default float16).}

\item{device}{Character. Target device.}
}
\value{
Tensor with original shape and specified dtype.
}
\description{
Reconstructs a float tensor from INT4-quantized data.
}
\examples{
\dontrun{
q <- quantize_int4(weights)
weights_approx <- dequantize_int4(q, dtype = torch_float16(), device = "cuda")
}
}
