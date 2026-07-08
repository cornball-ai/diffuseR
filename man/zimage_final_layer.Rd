% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_final_layer}
\alias{zimage_final_layer}
\title{Z-Image final layer}
\usage{
zimage_final_layer(hidden_size, out_channels)
}
\arguments{
\item{hidden_size}{Integer. Model width.}

\item{out_channels}{Integer. Patch output dim
(patch^2 * f_patch * latent channels).}
}
\description{
Parameterless LayerNorm scaled by 1 + adaLN(c) (scale only, no
shift), then the token-to-patch projection.
}
