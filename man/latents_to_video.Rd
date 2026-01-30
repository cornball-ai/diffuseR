% tinyrox says don't edit this manually, but it can't stop you!
\name{latents_to_video}
\alias{latents_to_video}
\title{Create Video from Latents (Helper)}
\usage{
latents_to_video(latents, vae, file, fps = 24, ...)
}
\arguments{
\item{latents}{Tensor of latents from generation.}

\item{vae}{VAE decoder module.}

\item{file}{Output file path.}

\item{fps}{Frames per second.}

\item{...}{Additional arguments to save_video.}
}
\value{
Invisibly returns the output file path.
}
\description{
Convenience function to decode latents and save video in one step.
}
