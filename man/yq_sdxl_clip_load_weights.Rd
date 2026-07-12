% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_clip_load_weights}
\alias{yq_sdxl_clip_load_weights}
\title{Load one SDXL CLIP text-encoder's weights into an anvl pytree}
\usage{
yq_sdxl_clip_load_weights(path, num_layers, has_text_projection = FALSE,
                          device = "cpu", strict = TRUE)
}
\arguments{
\item{path}{Path to the encoder state_dict safetensors.}

\item{num_layers}{Integer. Transformer layers (CLIP-L: 12, bigG: 32).}

\item{has_text_projection}{Logical. bigG carries \code{text_projection}
(pooled path); CLIP-L does not.}

\item{device}{Character. Target device.}

\item{strict}{Logical. Enforce the exact key census.}
}
\value{
Weights pytree for \code{\link{yq_sdxl_clip_l_encoder}} /
  \code{\link{yq_sdxl_bigg_encoder}} (plus \code{token_embedding} /
  \code{position_embedding} R matrices, and \code{text_projection} when
  present).
}
\description{
Reads a \code{diffuseR::text_encoder_native} /
\code{text_encoder2_native} state_dict (F16/F32 upcast to f32) and
mirrors its key tree. The token- and positional-embedding tables stay R
matrices \code{[rows, hidden]} for host-side gather
(\code{\link{yq_sdxl_clip_embed}}); every other weight is an
\code{AnvlArray}, with attention / MLP / projection linears transposed
to \code{[in, out]} for \code{\link[yunque]{yq_linear}}. With
\code{strict = TRUE} every checkpoint key must be consumed exactly once.
}
