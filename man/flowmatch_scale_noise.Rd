% tinyrox says don't edit this manually, but it can't stop you!
\name{flowmatch_scale_noise}
\alias{flowmatch_scale_noise}
\title{Scale noise for flow matching forward process}
\usage{
flowmatch_scale_noise(sample, timestep, noise, schedule)
}
\arguments{
\item{sample}{torch tensor. The clean sample.}

\item{timestep}{torch tensor. The current timestep.}

\item{noise}{torch tensor. The noise tensor.}

\item{schedule}{List. The FlowMatch scheduler object.}
}
\value{
torch tensor. The noisy sample at timestep t.
}
\description{
Applies the forward process in flow-matching: interpolates between
the clean sample and noise.
}
