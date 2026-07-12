% tinyrox says don't edit this manually, but it can't stop you!
\name{yq_sdxl_ddim_schedule}
\alias{yq_sdxl_ddim_schedule}
\title{Host-side DDIM schedule for SDXL (alpha coefficients per step)}
\usage{
yq_sdxl_ddim_schedule(num_inference_steps, num_train_timesteps = 1000L,
                      beta_start = 0.00085, beta_end = 0.012)
}
\arguments{
\item{num_inference_steps}{Integer. Denoising steps.}

\item{num_train_timesteps}{Integer. Training timesteps (1000).}

\item{beta_start,beta_end}{Numeric. Scaled-linear beta endpoints.}
}
\value{
List with \code{timesteps} (integer vector, the UNet timesteps
  per step), \code{alphas_cumprod} (length \code{num_train_timesteps}),
  and \code{coeff} (a list of per-step \code{a_t/b_t/a_prev/d_prev}).
}
\description{
Ports \code{diffuseR::ddim_scheduler_create} +
\code{ddim_scheduler_step} (epsilon prediction, eta 0) to plain R
scalars. Builds the scaled-linear beta schedule, the cumulative-alpha
product, the reversed inference timesteps, and, for each step, the four
square-root coefficients the epsilon DDIM update needs:
\code{a_t = sqrt(alpha_prod_t)}, \code{b_t = sqrt(1 - alpha_prod_t)},
\code{a_prev = sqrt(alpha_prod_t_prev)},
\code{d_prev = sqrt(1 - alpha_prod_t_prev)}. The timestep -> alpha-index
convention (\code{index = timestep + 1}, the final step falling back to
\code{alpha_prod_t_prev = alphas_cumprod[1]}) mirrors the reference
exactly. Computed in f64 (more accurate than the reference's f32
\code{torch_cumprod}; the ~1e-6 relative gap is far below the f32
end-to-end tolerance). The coefficients are prediction-type agnostic;
only \code{\link{yq_sdxl_ddim_step}}'s formula is epsilon-specific.
}
