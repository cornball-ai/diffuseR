% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx25_na3d}
\alias{.ltx25_na3d}
\title{Exact 3D neighborhood attention}
\usage{
.ltx25_na3d(query, key, value, plan)
}
\arguments{
\item{plan}{From \code{.ltx25_na_plan()} for this grid and kernel.}

\item{query,key,value}{Tensors [B, T, H, W, heads, d] (query already
scaled).}
}
\value{
Tensor [B, T, H, W, heads * d].
}
\description{
Exact 3D neighborhood attention
}
\keyword{internal}
