% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_split_keys}
\alias{ltx23_split_keys}
\title{Split checkpoint keys by component}
\usage{
ltx23_split_keys(keys)
}
\arguments{
\item{keys}{Character vector of checkpoint tensor names.}
}
\value{
Named list of character vectors: \code{dit},
  \code{connectors}, \code{vae}, \code{audio_vae}, \code{vocoder},
  and \code{other} (anything unrecognized; should be empty).
}
\description{
Splits the flat key space of an LTX single-file checkpoint into its
component groups. Connector tensors live under the
\code{model.diffusion_model.} prefix alongside the transformer, plus a
top-level \code{text_embedding_projection.} group; both are routed to
the \code{connectors} component.
}
