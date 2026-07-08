% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_t_embedder}
\alias{zimage_t_embedder}
\title{Z-Image timestep embedder}
\usage{
zimage_t_embedder(out_size, mid_size = 1024L, freq_size = 256L)
}
\arguments{
\item{out_size}{Integer. Output width, min(dim, 256).}

\item{mid_size}{Integer. Hidden width. The full model uses 1024.}

\item{freq_size}{Integer. Sinusoid width. Default 256.}
}
\description{
256-dim cos-first sinusoid (computed in float32) through a
Linear-SiLU-Linear MLP. The model feeds t * t_scale with the
pipeline's t already in [0, 1].
}
