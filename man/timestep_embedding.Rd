\name{timestep_embedding}
\alias{timestep_embedding}
\title{Sinusoidal Timestep Embedding}
\description{Sinusoidal Timestep Embedding}
\usage{
timestep_embedding(timesteps, dim)
}
\arguments{
  \item{timesteps}{Tensor of timesteps (batch_size,)}
  \item{dim}{Embedding dimension}
}
\value{
Tensor (batch_size, dim)
}
\keyword{internal}
