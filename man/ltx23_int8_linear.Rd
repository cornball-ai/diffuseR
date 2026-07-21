% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_int8_linear}
\alias{ltx23_int8_linear}
\title{INT8 linear layer}
\usage{
ltx23_int8_linear(out_features, in_features, bias = TRUE)
}
\arguments{
\item{bias}{Logical.}

\item{out_features,in_features}{Integers.}
}
\description{
Weight lives as int8 plus per-channel float32 scales in plain module
fields (so \code{$to(device)} moves only the bias); the forward pass
ships 1 byte/param to the input's device, upcasts into a persistent
per-shape buffer, and rescales in place.
}
