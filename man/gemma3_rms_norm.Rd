% tinyrox says don't edit this manually, but it can't stop you!
\name{gemma3_rms_norm}
\alias{gemma3_rms_norm}
\title{Gemma3 RMS Normalization}
\usage{
gemma3_rms_norm(dim, eps = 1e-06)
}
\arguments{
\item{dim}{Integer. Hidden dimension.}

\item{eps}{Numeric. Epsilon for numerical stability.}
}
\description{
RMSNorm with optional addition of 1 to weights (Gemma-style).
}
\keyword{internal}
