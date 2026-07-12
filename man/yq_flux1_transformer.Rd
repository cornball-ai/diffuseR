% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux1_transformer}
\alias{yq_flux1_transformer}
\title{FLUX.1 transformer forward (anvl port)}
\usage{
yq_flux1_transformer(num_layers = 19L, num_single_layers = 38L, heads = 24L,
                     head_dim = 128L, mlp_ratio = 4, eps = 1e-06,
                     precision = "highest")
}
\arguments{
\item{num_layers}{Integer. Double-stream blocks.}

\item{num_single_layers}{Integer. Single-stream blocks.}

\item{heads}{Integer. Attention heads.}

\item{head_dim}{Integer. Per-head dim.}

\item{mlp_ratio}{Numeric. Single-block MLP multiplier (4.0).}

\item{eps}{Numeric. Norm epsilon.}

\item{precision}{Character. Matmul precision.}
}
\value{
Function of (latents, text_embeds, pooled, time_sin, cos, sin,
  w):
  \itemize{
    \item latents \code{[B, S_img, in_channels]}
    \item text_embeds \code{[B, S_txt, joint_dim]}
    \item pooled \code{[B, pooled_dim]} pooled CLIP projection
    \item time_sin \code{[B, 256]} from \code{yq_flux1_time_embed()}
    \item cos, sin \code{[S_txt + S_img, head_dim]} RoPE tables
    \item w weights pytree
  }
  returning velocity \code{[B, S_img, out_channels]}.
}
\description{
anvl re-implementation of \code{diffuseR::flux_transformer} forward:
x/context embedders, timestep + pooled-text conditioning, a stack of
double-stream blocks over separate text/image streams, then
single-stream blocks over the concatenated sequence, an
adaLN-continuous output norm, and the velocity projection.
}
\details{
Returns a closure over the static config; \code{anvl::jit()} it.
Weights travel as a named pytree (see \code{\link{yq_flux1_load_weights}}).
The timestep sinusoid (\code{\link{yq_flux1_time_embed}}) and RoPE
tables are precomputed host-side and passed as inputs.

}
