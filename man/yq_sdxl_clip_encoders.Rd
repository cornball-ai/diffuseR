% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_clip_encoders}
\alias{yq_sdxl_clip_encoders}
\title{Run both SDXL CLIP encoders (anvl)}
\usage{
yq_sdxl_clip_encoders(clipl_embeds, bigg_embeds, mask, eos_index, w_clipl,
                      w_bigg, clipl_layers = 12L, clipl_heads = 12L,
                      bigg_layers = 32L, bigg_heads = 20L, head_dim = 64L,
                      eps = 1e-05, precision = "highest", jit = TRUE)
}
\arguments{
\item{clipl_embeds}{AnvlArray \code{[B, S, 768]} CLIP-L token+pos
embeddings (\code{\link{yq_sdxl_clip_embed}}).}

\item{bigg_embeds}{AnvlArray \code{[B, S, 1280]} bigG token+pos
embeddings.}

\item{mask}{AnvlArray \code{[B, 1, S, S]} causal mask
(\code{\link{yq_sdxl_clip_mask}}); shared by both encoders.}

\item{eos_index}{Integer. 1-based EOS position (\code{which.max} over
the token ids). The pooled vector is gathered here. Batch = 1.}

\item{precision}{Character. Matmul precision.}

\item{jit}{Logical. Compile the transformer stacks with
\code{\link[anvl]{jit}}.}

\item{w_clipl,w_bigg}{Weight pytrees from
\code{\link{yq_sdxl_clip_load_weights}}.}

\item{clipl_layers,clipl_heads,bigg_layers,bigg_heads,head_dim,eps}{Architecture (SDXL defaults).}
}
\value{
List with \code{context} \code{[B, seq, 2048]} and
  \code{pooled} \code{[B, 1280]}.
}
\description{
Runs the CLIP-L and OpenCLIP bigG encoders on pre-embedded tokens and
assembles the two SDXL conditioning tensors: the penultimate context
(concatenated CLIP-L + bigG penultimate states,
\code{[B, seq, 2048]}) and bigG's pooled text embed
(\code{[B, 1280]}). The two transformer stacks are \code{jit}-compiled;
the EOS gather and \code{text_projection} run eagerly.
}
