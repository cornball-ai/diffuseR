% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_nf4_dequantize}
\alias{ltx23_nf4_dequantize}
\title{Dequantize NF4 data to a float tensor}
\usage{
ltx23_nf4_dequantize(packed, absmax, shape, dtype = torch::torch_bfloat16(),
                     chunk_elements = 33554432L)
}
\arguments{
\item{packed}{uint8 tensor of packed index pairs.}

\item{absmax}{float32 tensor of per-block scales.}

\item{shape}{Integer vector. Original tensor shape.}

\item{dtype}{Target torch dtype.}

\item{chunk_elements}{Integer. Elements dequantized per slice (bounds
the int64 index temporary).}
}
\value{
Tensor of \code{shape} in \code{dtype} on the input's device.
}
\description{
Dequantize NF4 data to a float tensor
}
