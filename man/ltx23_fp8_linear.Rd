% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_fp8_linear}
\alias{ltx23_fp8_linear}
\title{FP8 linear layer}
\usage{
ltx23_fp8_linear(out_features, in_features, bias = TRUE)
}
\arguments{
\item{bias}{Logical.}

\item{out_features,in_features}{Integers.}
}
\description{
Weight lives as float8_e4m3fn plus a float32 scale in plain module
fields (so \code{$to(device)} moves only the bias); the forward pass
ships 1 byte/param to the input's device, upcasts, rescales, and runs
\code{nnf_linear}.
}
