% tinyrox says don't edit this manually, but it can't stop you!
\name{download_animatediff}
\alias{download_animatediff}
\title{Download an AnimateDiff motion module}
\usage{
download_animatediff(version = c("v3", "v2"), verbose = TRUE)
}
\arguments{
\item{version}{"v3" (\code{v3_sd15_mm.ckpt}, no mid-block module) or
"v2" (\code{mm_sd_v15_v2.ckpt}).}

\item{verbose}{Logical.}
}
\value{
Invisibly, the path of the converted safetensors file.
}
\description{
Fetches the original Apache-2.0 checkpoint from
\code{guoyww/animatediff} and converts it once to an fp16 safetensors
file (the pickle is read by \code{torch::load_state_dict()}, which only
rebuilds tensors). Keys keep their original names;
\code{\link{load_motion_adapter}} maps them.
}
