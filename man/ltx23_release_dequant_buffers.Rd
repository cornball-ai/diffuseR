% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_release_dequant_buffers}
\alias{ltx23_release_dequant_buffers}
\title{Release the NF4 dequantization buffers}
\usage{
ltx23_release_dequant_buffers()
}
\value{
Invisibly, NULL.
}
\description{
Frees the cached per-shape weight buffers (e.g. before decoding at
high resolution).
}
