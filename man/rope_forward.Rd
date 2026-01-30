% tinyrox says don't edit this manually, but it can't stop you!
\name{rope_forward}
\alias{rope_forward}
\title{Compute RoPE frequencies from coordinates}
\usage{
rope_forward(embedder, coords, device = NULL)
}
\arguments{
\item{embedder}{List. RoPE embedder configuration.}

\item{coords}{torch tensor. Coordinate tensor from rope_prepare_video_coords().}

\item{device}{Character or torch device. Device for output tensors.}
}
\value{
A list with:
  \describe{
    \item{cos_freqs}{Cosine frequencies tensor}
    \item{sin_freqs}{Sine frequencies tensor}
  }
}
\description{
Converts spatiotemporal coordinates to (cos, sin) frequency tensors
for applying rotary embeddings.
}
