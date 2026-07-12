% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_ltx_dit}
\alias{yq_ltx_dit}
\title{LTX-2.3 audio-video DiT forward (anvl port)}
\usage{
yq_ltx_dit(heads = 3L, head_dim = 8L, a_heads = 2L, a_head_dim = 6L,
           num_layers = 2L, isolate = TRUE, eps = 1e-06, precision = "highest")
}
\arguments{
\item{num_layers}{Integer. Transformer block count.}

\item{isolate}{Logical. \code{TRUE} runs the streams independently
(\code{isolate_modalities}); \code{FALSE} adds a2v/v2a cross-attention.}

\item{eps}{Numeric. Norm epsilon.}

\item{precision}{Character. Matmul precision.}

\item{heads,head_dim}{Video attention shape (inner = heads*head_dim).}

\item{a_heads,a_head_dim}{Audio attention shape.}
}
\value{
Function of (act, sins, ropes, w):
  \itemize{
    \item act = list(hidden \code{[B,Sv,in]}, audio_hidden \code{[B,Sa,ain]},
      enc \code{[B,St,inner]}, audio_enc \code{[B,Sat,ainner]})
    \item sins = list(time \code{[B*Sv,256]}, audio_time \code{[B*Sa,256]},
      prompt \code{[B,256]}, audio_prompt \code{[B,256]})
    \item ropes = list(v_cos, v_sin, a_cos, a_sin, vca_cos, vca_sin,
      aca_cos, aca_sin)
    \item w weights pytree (\code{\link{yq_ltx_dit_load_weights}})
  }
  returning \code{list(video \code{[B,Sv,out]}, audio \code{[B,Sa,aout]})}.
}
\description{
Returns a closure over the static config; \code{anvl::jit()} it.
Activations, precomputed timestep sinusoids, and RoPE cos/sin tables
travel as three named lists plus the weights pytree.
}
