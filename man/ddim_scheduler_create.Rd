\name{ddim_scheduler_create}
\alias{ddim_scheduler_create}
\title{Create a DDIM Scheduler}
\description{Creates a Denoising Diffusion Implicit Models (DDIM) scheduler for use with diffusion models. DDIM schedulers provide a deterministic sampling process that offers faster inference compared to DDPM while maintaining high quality outputs.}
\usage{
ddim_scheduler_create(num_train_timesteps, num_inference_steps, eta, beta_schedule, beta_start, beta_end, rescale_betas_zero_snr, dtype, device)
}
\arguments{
  \item{num_train_timesteps}{Integer. The number of diffusion steps used to train the model. Default: 1000}
  \item{num_inference_steps}{Integer. The number of diffusion steps used for inference. Fewer steps typically means faster inference at the cost of sample quality. Default: 50}
  \item{eta}{Numeric. Controls the amount of stochasticity. When eta=0, the sampling process is deterministic. When eta=1, the sampling process is equivalent to DDPM. Default: 0}
  \item{beta_schedule}{Character. The beta schedule to use. Options are: \\describe\{ \\item\{"linear"\}\{Linear beta schedule from beta_start to beta_end\} \\item\{"scaled_linear"\}\{Scaled linear schedule, generally gives better results\} \\item\{"cosine"\}\{Cosine schedule that approaches zero smoothly\} \} Default: "linear"}
  \item{beta_start}{Numeric. The starting value for the beta schedule. Default: 0.00085}
  \item{beta_end}{Numeric. The final value for the beta schedule. Default: 0.012}
  \item{rescale_betas_zero_snr}{Logical. If TRUE, rescales the beta values}
  \item{dtype}{The data type to use for computations. Default is torch_float32(). Options are torch_float16() and torch_float32().}
  \item{device}{The device to use for computations. Options are torch_device("cpu"), torch_device("cuda").}
}
\details{
DDIM (Denoising Diffusion Implicit Models) was introduced by Song et al. (2020)
as an extension to DDPM (Denoising Diffusion Probabilistic Models). It offers
a deterministic sampling process and allows for controlling the number of
inference steps independently from the training process.

The scheduler contains the noise schedule and methods for computing
alpha, beta, and other parameters used in the diffusion process.
}
\value{
A DDIM scheduler object that can be used with diffusion models to
  generate samples.
}
\references{
Song, J., Meng, C., & Ermon, S. (2020).
"Denoising Diffusion Implicit Models."
\\url\{https://arxiv.org/abs/2010.02502\}
}
\examples{
\dontrun{
# Create a DDIM scheduler with custom parameters
scheduler <- ddim_scheduler_create(
  num_train_timesteps = 1000,
  num_inference_steps = 30,
  eta = 0.5,
  beta_schedule = "scaled_linear"
)
}
}
