% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_clip_load_weights}
\alias{yq_clip_load_weights}
\title{Load SD 2.1 CLIP text-encoder weights into an anvl pytree}
\usage{
yq_clip_load_weights(path, num_layers = 23L, device = "cpu", strict = TRUE)
}
\arguments{
\item{path}{Path to \code{text_encoder/model.safetensors} (or a
directory containing it).}

\item{num_layers}{Integer. Transformer layers (SD 2.1: 23).}

\item{device}{Character. Target device.}

\item{strict}{Logical. Enforce the exact key census.}
}
\value{
Weights pytree for \code{\link{yq_clip_encoder}} (plus
  \code{token_embedding} / \code{position_embedding} R matrices for
  \code{\link{yq_clip_embed}}).
}
\description{
Reads the diffusers \code{CLIPTextModel} \code{model.safetensors}
(F16 upcast to f32) and mirrors its key tree. The token- and
positional-embedding tables stay R matrices \code{[rows, hidden]} for
host-side gather (\code{\link{yq_clip_embed}}); every other weight is
wrapped as an \code{AnvlArray}, with attention / MLP linears transposed
to \code{[in, out]} for \code{\link[yunque]{yq_linear}}. With
\code{strict = TRUE} every checkpoint key must be consumed exactly once,
so a wrong architecture or a missed weight fails loudly.
}
