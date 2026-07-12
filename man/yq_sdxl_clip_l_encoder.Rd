% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_clip_l_encoder}
\alias{yq_sdxl_clip_l_encoder}
\title{SDXL CLIP-L (OpenAI ViT-L/14) penultimate encoder forward (anvl)}
\usage{
yq_sdxl_clip_l_encoder(num_layers = 12L, num_heads = 12L, head_dim = 64L,
                       eps = 1e-05, precision = "highest")
}
\arguments{
\item{num_layers}{Integer. Transformer layers (CLIP-L: 12).}

\item{num_heads}{Integer. Attention heads (12).}

\item{head_dim}{Integer. Per-head dimension (64).}

\item{eps}{Numeric. LayerNorm epsilon.}

\item{precision}{Character. Matmul precision.}
}
\value{
Function of \code{(embeds, mask, w)} returning \code{[B, S, 768]}.
}
\description{
Returns a closure over the static CLIP-L configuration;
\code{anvl::jit()} the closure. Runs the first \code{num_layers - 1}
transformer layers over pre-embedded tokens (quick-GELU MLP) and
returns that penultimate hidden state \code{[B, S, 768]}
(\code{hidden_states[-2]}, no final LayerNorm). The final layer's
weights load but are unused, matching diffusers' SDXL clip-skip=None
selection.
}
