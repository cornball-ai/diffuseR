% tinyrox says don't edit this manually, but it can't stop you!
\name{load_motion_adapter}
\alias{load_motion_adapter}
\title{Load an AnimateDiff motion adapter from safetensors}
\usage{
load_motion_adapter(path, verbose = TRUE)
}
\arguments{
\item{path}{A \code{.safetensors} file.}

\item{verbose}{Logical.}
}
\value{
A \code{\link{motion_adapter}} in eval mode (float32, CPU).
}
\description{
Reads a motion module checkpoint in either the original AnimateDiff key
layout (as converted by \code{\link{download_animatediff}}) or diffusers'
\code{MotionAdapter} layout, builds a matching
\code{\link{motion_adapter}} and fills every parameter. The stored
position tables are skipped (the module computes the same sinusoids);
any other unmatched key, shape mismatch or unfilled parameter is an
error.
}
