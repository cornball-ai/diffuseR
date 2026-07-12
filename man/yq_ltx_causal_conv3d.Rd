% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_causal_conv3d}
\alias{yq_ltx_causal_conv3d}
\title{Causal 3D convolution (anvl)}
\usage{
yq_ltx_causal_conv3d(x, weight, bias, causal = FALSE, kernel_t = 3L)
}
\arguments{
\item{x}{AnvlArray \code{[B, C_in, F, H, W]}.}

\item{weight}{AnvlArray \code{[C_out, C_in, k_t, k_h, k_w]} (raw torch
layout).}

\item{bias}{AnvlArray \code{[C_out]}.}

\item{causal}{Logical. Left-only temporal padding when \code{TRUE};
symmetric when \code{FALSE} (the decoder default).}

\item{kernel_t}{Integer. Temporal kernel size (3 in LTX; padding is
skipped when 1).}
}
\value{
AnvlArray \code{[B, C_out, F, H, W]}.
}
\description{
Temporal edge-replication padding then a stride-1 Conv3d with
symmetric spatial zero padding, via \code{\link[anvl]{prim_convolution}}.
Matches \code{diffuseR::ltx23_causal_conv3d}'s forward for both the
causal (left-only) and non-causal (symmetric) temporal padding modes.
}
