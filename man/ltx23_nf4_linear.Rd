% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_nf4_linear}
\alias{ltx23_nf4_linear}
\title{NF4 linear layer}
\usage{
ltx23_nf4_linear(out_features, in_features, bias = TRUE)
}
\arguments{
\item{bias}{Logical.}

\item{out_features,in_features}{Integers.}
}
\description{
Packed weights and per-block scales are registered as buffers, so
they move with the module (uint8 packs are untouched by dtype
conversions). The forward pass dequantizes on the weight's device.
}
