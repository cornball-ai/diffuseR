% tinyrox says don't edit this manually, but it can't stop you!
\name{sequential_cfg_forward}
\alias{sequential_cfg_forward}
\title{Sequential CFG Forward Pass}
\usage{
sequential_cfg_forward(
  model,
  latents,
  timestep,
  prompt_embeds,
  negative_prompt_embeds,
  guidance_scale,
  ...
)
}
\arguments{
\item{model}{The DiT model.}

\item{latents}{Current latent tensor.}

\item{timestep}{Current timestep tensor.}

\item{prompt_embeds}{Conditional prompt embeddings.}

\item{negative_prompt_embeds}{Unconditional prompt embeddings.}

\item{guidance_scale}{CFG scale.}

\item{...}{Additional arguments to model forward pass.}
}
\value{
The CFG-combined noise prediction.
}
\description{
Runs unconditional and conditional forward passes separately to halve
peak activation memory. For GPU-poor scenarios.
}
\examples{
\dontrun{
noise_pred <- sequential_cfg_forward(
  model, latents, timestep,
  prompt_embeds, negative_prompt_embeds,
  guidance_scale = 4.0
)
}
}
