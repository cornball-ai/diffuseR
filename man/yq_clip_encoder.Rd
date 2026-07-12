% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_clip_encoder}
\alias{yq_clip_encoder}
\title{Stable Diffusion 2.1 CLIP text encoder forward (anvl)}
\usage{
yq_clip_encoder(num_layers = 23L, num_heads = 16L, head_dim = 64L, eps = 1e-05,
                apply_final_ln = TRUE, precision = "highest")
}
\arguments{
\item{num_layers}{Integer. Transformer layers (SD 2.1: 23).}

\item{num_heads}{Integer. Attention heads (16).}

\item{head_dim}{Integer. Per-head dimension (64).}

\item{eps}{Numeric. LayerNorm epsilon.}

\item{apply_final_ln}{Logical. Apply the final LayerNorm (SD 2.1 uses
the final-LN hidden state; \code{FALSE} gives the pre-final-LN state,
e.g. SDXL's penultimate prompt embeds).}

\item{precision}{Character. Matmul precision.}
}
\value{
Function of \code{(embeds, mask, w)}:
  \itemize{
    \item embeds \code{[B, S, hidden]} from \code{\link{yq_clip_embed}}
    \item mask \code{[B, 1, S, S]} from \code{\link{yq_clip_mask}}
    \item w weights pytree (\code{\link{yq_clip_load_weights}})
  }
  returning the hidden states \code{[B, S, hidden]}.
}
\description{
Returns a closure over the static OpenCLIP ViT-H configuration;
\code{anvl::jit()} the closure. Runs the 23 transformer layers over
pre-embedded tokens and (for SD 2.1) applies the final LayerNorm.
}
