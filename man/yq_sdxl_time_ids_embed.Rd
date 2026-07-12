% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_time_ids_embed}
\alias{yq_sdxl_time_ids_embed}
\title{Host-side SDXL added-conditioning time-ids embedding}
\usage{
yq_sdxl_time_ids_embed(time_ids, dim = 256L, device = "cpu")
}
\arguments{
\item{time_ids}{Numeric matrix \code{[batch, n_time_ids]} (a length-
\code{n_time_ids} vector is treated as one batch row).}

\item{dim}{Integer. Per-id embedding width (SDXL: 256, even).}

\item{device}{Character. Target device.}
}
\value{
AnvlArray \code{[batch, n_time_ids * dim]}, f32.
}
\description{
Reproduces the SDXL \code{add_embedding} Fourier features:
\code{timestep_embedding(time_ids$flatten(), addition_time_embed_dim)}
reshaped to \code{[batch, n_time_ids * dim]}. Each of the
\code{n_time_ids} (6) micro-conditioning scalars per batch row is
sinusoidally embedded (same \code{[cos, sin]} convention as
\code{\link{yq_sdxl_time_embed}}) and the row-major concatenation
becomes the \code{time_ids} half of the \code{add_embedding} input
(the pooled text embeds are the other half).
}
