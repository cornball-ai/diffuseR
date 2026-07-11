% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_transformer}
\alias{yq_flux2_transformer}
\title{FLUX.2 Klein transformer forward (anvl port)}
\usage{
yq_flux2_transformer(num_layers = 5L, num_single_layers = 20L, heads = 24L,
                     head_dim = 128L, mlp_ratio = 3, eps = 1e-06,
                     precision = "highest")
}
\arguments{
\item{num_layers}{Integer. Double-stream blocks (klein: 5).}

\item{num_single_layers}{Integer. Single-stream blocks (klein: 20).}

\item{heads}{Integer. Attention heads (24).}

\item{head_dim}{Integer. Per-head dim (128).}

\item{mlp_ratio}{Numeric. FF multiplier (3.0).}

\item{eps}{Numeric. Norm epsilon.}

\item{precision}{Character. Matmul precision (see
\code{\link[yunque]{yq_linear}}).}
}
\value{
Function of (latents, text_embeds, time_proj, cos, sin, w):
  \itemize{
    \item latents \code{[B, S_img, in_channels]}
    \item text_embeds \code{[B, S_txt, joint_dim]}
    \item time_proj \code{[B, 256]} from \code{yq_flux2_time_proj()}
    \item cos, sin \code{[S_txt + S_img, head_dim]} RoPE tables
    \item w weights pytree
  }
  returning velocity \code{[B, S_img, out_channels]}.
}
\description{
anvl re-implementation of \code{diffuseR::flux2_transformer} forward
(Flux2Transformer2DModel, klein-4B): x/context embedders, a shared
timestep-MLP feeding three modulation projections, a stack of
double-stream blocks over separate text/image streams, then
single-stream blocks over the concatenated sequence, an
adaLN-continuous output norm, and the velocity projection.
}
\details{
Returns a closure over the static config; \code{anvl::jit()} it.
Weights travel as a named pytree (see \code{yq_flux2_load_weights}).
The timestep sinusoid (\code{\link{yq_flux2_time_proj}}) and RoPE
tables (built from position ids) are precomputed host-side and passed
as inputs, matching the diffusers pipeline boundary.

}
