% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_single_block}
\alias{yq_flux2_single_block}
\title{FLUX.2 single-stream block (anvl port)}
\usage{
yq_flux2_single_block(heads = 24L, head_dim = 128L, mlp_ratio = 3, eps = 1e-06,
                      precision = "highest")
}
\arguments{
\item{heads}{Integer. Attention heads (klein-4B: 24).}

\item{head_dim}{Integer. Per-head dimension (128).}

\item{mlp_ratio}{Numeric. MLP hidden multiplier (3.0).}

\item{eps}{Numeric. Norm epsilon.}

\item{precision}{Character. Matmul precision for every linear and
attention matmul in the block (see \code{\link[yunque]{yq_linear}}).}
}
\value{
Function of (h, shift, scale, gate, cos, sin, w_qkv_t,
  w_out_t, norm_q_w, norm_k_w):
  \itemize{
    \item h \code{[B, S, dim]} joint (text; image) hidden states
    \item shift, scale, gate \code{[1, 1, dim]} modulation triple
    \item cos, sin \code{[S, head_dim]} rotary tables
    \item w_qkv_t \code{[dim, 3*heads*head_dim + 2*mlp_hidden]}
      (checkpoint weight transposed)
    \item w_out_t \code{[heads*head_dim + mlp_hidden, dim]}
    \item norm_q_w, norm_k_w \code{[head_dim]}
  }
}
\description{
anvl re-implementation of \code{diffuseR::flux2_single_block} +
\code{flux2_parallel_self_attention}: parameterless LayerNorm with
external (shift, scale, gate) modulation, one fused projection
producing QKV and the SwiGLU MLP input, RMS-normed q/k, rotary
embeddings, attention, and one fused output projection consuming
cat(attention, MLP).
}
\details{
Returns a closure over the static configuration; \code{anvl::jit()}
the closure. All array arguments are dynamic inputs.

}
