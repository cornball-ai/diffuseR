% tinyrox says don't edit this manually, but it can't stop you!
\name{.cuda_index}
\alias{.cuda_index}
\title{Device ordinal for a torch device string}
\usage{
.cuda_index(device)
}
\arguments{
\item{device}{Character, e.g. "cuda", "cuda:0", "cuda:1".}
}
\value{
Integer ordinal. An unqualified device gives the current one.
}
\description{
\code{torch::cuda_memory_stats()} defaults to
\code{cuda_current_device()}, so reading it without an argument reports
whichever device happens to be current rather than the one a handle is
bound to. \code{resident_load()} binds an explicit \code{"cuda:N"}
precisely so transitions cannot drift, and a handle on \code{cuda:1}
deciding from \code{cuda:0}'s pool would either skip a pre-warm it needs
or repeat one it does not.
}
\keyword{internal}
