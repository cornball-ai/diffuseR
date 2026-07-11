% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_flux2_sample}
\alias{yq_flux2_sample}
\title{FLUX.2 Klein denoising loop (anvl)}
\usage{
yq_flux2_sample(latents, text_embeds, sigmas, cos, sin, w_dit, dit_fn = NULL,
                device = "cpu", precision = "highest")
}
\arguments{
\item{latents}{AnvlArray \code{[B, S_img, in_channels]} initial noise
(packed).}

\item{text_embeds}{AnvlArray \code{[B, S_txt, joint_dim]} from
\code{\link{yq_qwen3_encoder}}.}

\item{sigmas}{Numeric vector from \code{\link{yq_flux2_sigmas}}.}

\item{w_dit}{DiT weights pytree from
\code{\link{yq_flux2_load_weights}}.}

\item{dit_fn}{Optional pre-jitted DiT closure; built from
\code{\link{yq_flux2_transformer}} if NULL.}

\item{device}{Character. Device for the timestep projections.}

\item{precision}{Character. Matmul precision.}

\item{cos,sin}{RoPE tables from \code{\link{yq_flux2_rope}} (built
over the concatenated text + image ids).}
}
\value{
AnvlArray \code{[B, S_img, out_channels]} final latents.
}
\description{
Runs the FlowMatch Euler loop over the jitted DiT: at each step,
predicts velocity and takes \code{latents <- latents + dt * velocity}
with \code{dt = sigmas[i+1] - sigmas[i]}. The DiT is jitted once and
called \code{n_steps} times; latents stay on-device across steps.
Guidance-free (distilled klein). Produces final packed latents (VAE
decode is a separate, convolution-dependent stage).
}
