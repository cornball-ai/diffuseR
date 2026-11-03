% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx25_na_plan}
\alias{.ltx25_na_plan}
\title{Plan a 3D neighborhood attention}
\usage{
.ltx25_na_plan(
  grid,
  kernel,
  device,
  channels = 1L,
  tile = c(8L, 8L, 8L),
  budget = 2^27
)
}
\arguments{
\item{grid}{Integer [T, H, W].}

\item{kernel}{Integer [kt, kh, kw]; each at most its grid size.}

\item{device}{Device for the index and mask tensors.}

\item{channels}{Width of the gathered keys (heads x head_dim).}

\item{tile}{Query tile size per axis.}

\item{budget}{Maximum gathered key elements (key tokens x channels)
per SDPA call.}
}
\value{
List of chunks, each list(q_idx, kv_idx, mask, n_tiles, n_q,
  n_k), indices 1-based long tensors.
}
\description{
Groups query tiles of a [T, H, W] grid into SDPA calls that share one
mask, with token indices into the flattened (t, h, w) sequence. The
plan depends only on the grid and kernel, so a decoder stage builds it
once for all its blocks.
}
\keyword{internal}
