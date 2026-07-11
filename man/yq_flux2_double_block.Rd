% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_double_block}
\alias{yq_flux2_double_block}
\title{FLUX.2 double-stream (MMDiT) block (anvl port)}
\usage{
yq_flux2_double_block(heads = 24L, head_dim = 128L, mlp_ratio = 3, eps = 1e-06,
                      precision = "highest")
}
\arguments{
\item{heads}{Integer. Attention heads (klein-4B: 24).}

\item{head_dim}{Integer. Per-head dimension (128).}

\item{mlp_ratio}{Numeric. Feed-forward multiplier (3.0).}

\item{eps}{Numeric. Norm epsilon.}

\item{precision}{Character. Matmul precision (see
\code{\link[yunque]{yq_linear}}).}
}
\value{
Function of (h, c, mod_img, mod_txt, cos, sin, w):
  \itemize{
    \item h \code{[B, S_img, dim]} image hidden states
    \item c \code{[B, S_txt, dim]} text (encoder) hidden states
    \item mod_img, mod_txt \code{[B, 6*dim]} shared modulation outputs
    \item cos, sin \code{[S_txt + S_img, head_dim]} rotary tables for
      the concatenated sequence
    \item w named list of block weights (to_q/to_k/to_v, norm_q/k,
      add_{q,k,v}_proj, norm_added_{q,k}, to_out, to_add_out,
      ff.linear_{in,out}, ff_context.linear_{in,out})
  }
  returning \code{list(c_out, h_out)}.
}
\description{
anvl re-implementation of \code{diffuseR::flux2_double_block} +
\code{flux_attention(added_kv = TRUE)}: separate image and text
streams, each with parameterless-LayerNorm adaLN modulation and a
SwiGLU feed-forward, joined by a single attention over the
concatenated [text; image] sequence (text tokens first, matching the
rotary layout).
}
\details{
Returns a closure over the static config; \code{anvl::jit()} the
closure. Weights travel as a named list (pytree) mirroring the
checkpoint keys under \code{transformer_blocks.N.}, all 2-D linears
pre-transposed to \code{[in, out]} (see
\code{\link{yq_read_safetensors}}).

}
