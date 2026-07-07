% tinyrox says don't edit this manually, but it can't stop you!
\name{ltx23_preprocess_frames}
\alias{ltx23_preprocess_frames}
\title{Preprocess an image (or frame stack) for VAE encoding}
\usage{
ltx23_preprocess_frames(x, height, width)
}
\arguments{
\item{x}{Path to a PNG/JPEG, or an array [H, W, 3] (values in
[0, 1]), or a [F, H, W, 3] array of frames.}

\item{height,width}{Integers. Target size (multiples of 32).}
}
\value{
Float32 tensor [1, 3, F, height, width] in [-1, 1].
}
\description{
Mirrors the diffusers VideoProcessor: bilinear resize so the shorter
relative side matches, center-crop to the exact target, and scale to
[-1, 1].
}
