#' Create a FlowMatch Euler Discrete Scheduler
#'
#' Creates a FlowMatch scheduler for use with flow-matching diffusion models
#' like LTX-2. FlowMatch schedulers use Euler integration for sampling,
#' which is simpler and often faster than DDIM-style schedulers.
#'
#' @param num_train_timesteps Integer. The number of diffusion steps used to
#'   train the model. Default: 1000
#' @param shift Numeric. The shift value for the timestep schedule. Default: 1.0
#' @param use_dynamic_shifting Logical. Whether to apply timestep shifting
#'   on-the-fly based on the image/video resolution. Default: FALSE
#' @param base_shift Numeric. Value to stabilize generation. Increasing
#'   reduces variation. Default: 0.5
#' @param max_shift Numeric. Maximum shift allowed. Increasing encourages
#'   more variation. Default: 1.15
#' @param base_seq_len Integer. Base sequence length for dynamic shifting. Default: 256
#' @param max_seq_len Integer. Maximum sequence length for dynamic shifting. Default: 4096
#' @param invert_sigmas Logical. Whether to invert the sigmas (used by some models
#'   like Mochi). Default: FALSE
#' @param shift_terminal Numeric or NULL. End value of shifted schedule. Default: NULL
#' @param time_shift_type Character. Type of dynamic shifting: "exponential" or
#'   "linear". Default: "exponential"
#'
#' @return A FlowMatch scheduler object (list) containing:
#'   \describe{
#'     \item{sigmas}{The noise schedule}
#'     \item{timesteps}{The timestep schedule}
#'     \item{num_train_timesteps}{Training timesteps}
#'     \item{config}{All configuration parameters}
#'   }
#'
#' @details
#' FlowMatch (Flow Matching) is a framework for training continuous normalizing
#' flows by regressing onto target probability paths. The Euler discrete scheduler
#' implements simple Euler integration for sampling from trained flow models.
#'
#' The core update rule is:
#' \code{prev_sample = sample + dt * model_output}
#' where \code{dt = sigma_next - sigma_current}.
#'
#' @references
#' Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022).
#' "Flow Matching for Generative Modeling."
#' \url{https://arxiv.org/abs/2210.02747}
#'
#' @examples
#' \dontrun{
#' # Create a FlowMatch scheduler
#' scheduler <- flowmatch_scheduler_create(
#'   num_train_timesteps = 1000,
#'   shift = 1.0
#' )
#'
#' # Set timesteps for inference
#' scheduler <- flowmatch_set_timesteps(scheduler, num_inference_steps = 8)
#' }
#' @export
flowmatch_scheduler_create <- function(
  num_train_timesteps = 1000L,
  shift = 1.0,
  use_dynamic_shifting = FALSE,
  base_shift = 0.5,
  max_shift = 1.15,
  base_seq_len = 256L,
  max_seq_len = 4096L,
  invert_sigmas = FALSE,
  shift_terminal = NULL,
  time_shift_type = c("exponential", "linear")
) {
  time_shift_type <- match.arg(time_shift_type)

  # Initial timesteps (reversed, from max to min)
  timesteps <- seq(from = 1, to = num_train_timesteps, length.out = num_train_timesteps)
  timesteps <- rev(timesteps)

  # Initial sigmas (normalized timesteps)
  sigmas <- timesteps / num_train_timesteps

  # Apply shift if not using dynamic shifting
  if (!use_dynamic_shifting) {
    sigmas <- shift * sigmas / (1 + (shift - 1) * sigmas)
  }

  timesteps <- sigmas * num_train_timesteps

  list(
    sigmas = sigmas,
    timesteps = timesteps,
    num_train_timesteps = num_train_timesteps,
    num_inference_steps = NULL,
    step_index = NULL,
    config = list(
      num_train_timesteps = num_train_timesteps,
      shift = shift,
      use_dynamic_shifting = use_dynamic_shifting,
      base_shift = base_shift,
      max_shift = max_shift,
      base_seq_len = base_seq_len,
      max_seq_len = max_seq_len,
      invert_sigmas = invert_sigmas,
      shift_terminal = shift_terminal,
      time_shift_type = time_shift_type
    )
  )
}

#' Calculate shift for dynamic shifting
#'
#' Computes the shift parameter (mu) based on sequence length for
#' resolution-dependent timestep shifting.
#'
#' @param seq_len Integer. The sequence length (num_patches).
#' @param base_seq_len Integer. Base sequence length. Default: 256
#' @param max_seq_len Integer. Maximum sequence length. Default: 4096
#' @param base_shift Numeric. Base shift value. Default: 0.5
#' @param max_shift Numeric. Maximum shift value. Default: 1.15
#'
#' @return Numeric. The computed shift value (mu).
#' @export
flowmatch_calculate_shift <- function(
  seq_len,
  base_seq_len = 256L,
  max_seq_len = 4096L,
  base_shift = 0.5,
  max_shift = 1.15
) {
  m <- (max_shift - base_shift) / (max_seq_len - base_seq_len)
  b <- base_shift - m * base_seq_len
  mu <- seq_len * m + b
  mu
}

#' Set timesteps for inference
#'
#' Configures the scheduler timesteps for a specific number of inference steps.
#' This must be called before using the scheduler for denoising.
#'
#' @param schedule List. The FlowMatch scheduler object.
#' @param num_inference_steps Integer. Number of denoising steps. Default: 50
#' @param device Character or torch device. Device for tensors. Default: "cpu"
#' @param mu Numeric or NULL. Shift parameter for dynamic shifting. Required
#'   if use_dynamic_shifting is TRUE. Default: NULL
#' @param sigmas Numeric vector or NULL. Custom sigma values. Default: NULL
#' @param timesteps Numeric vector or NULL
#'
#' @return Updated scheduler with configured timesteps and sigmas.
#' @export
flowmatch_set_timesteps <- function(
  schedule,
  num_inference_steps = 50L,
  device = "cpu",
  mu = NULL,
  sigmas = NULL,
  timesteps = NULL
) {
  config <- schedule$config

  if (config$use_dynamic_shifting && is.null(mu)) {
    stop("`mu` must be provided when use_dynamic_shifting is TRUE")
  }

  schedule$num_inference_steps <- num_inference_steps

  # Get sigma_max and sigma_min
  sigma_max <- max(schedule$sigmas)
  sigma_min <- min(schedule$sigmas)

  # Create sigmas if not provided
  if (is.null(sigmas)) {
    if (is.null(timesteps)) {
      # Linear spacing from sigma_max to sigma_min
      timesteps <- seq(
        from = sigma_max * config$num_train_timesteps,
        to = sigma_min * config$num_train_timesteps,
        length.out = num_inference_steps
      )
    }
    sigmas <- timesteps / config$num_train_timesteps
  }

  # Apply timestep shifting
  if (config$use_dynamic_shifting) {
    sigmas <- .flowmatch_time_shift(
      mu = mu,
      sigma = 1.0,
      t = sigmas,
      shift_type = config$time_shift_type
    )
  } else {
    shift <- config$shift
    sigmas <- shift * sigmas / (1 + (shift - 1) * sigmas)
  }

  # Stretch to terminal if configured
  if (!is.null(config$shift_terminal)) {
    sigmas <- .flowmatch_stretch_shift_to_terminal(sigmas, config$shift_terminal)
  }

  # Convert to tensors
  sigmas <- torch::torch_tensor(sigmas, dtype = torch::torch_float32())
  timesteps <- sigmas * config$num_train_timesteps

  # Append terminal sigma
  if (config$invert_sigmas) {
    sigmas <- 1.0 - sigmas
    timesteps <- sigmas * config$num_train_timesteps
    sigmas <- torch::torch_cat(list(sigmas, torch::torch_ones(1)))
  } else {
    sigmas <- torch::torch_cat(list(sigmas, torch::torch_zeros(1)))
  }

  # Move to device
  sigmas <- sigmas$to(device = device)
  timesteps <- timesteps$to(device = device)

  schedule$sigmas <- sigmas
  schedule$timesteps <- timesteps
  schedule$step_index <- NULL

  schedule
}

#' Perform a FlowMatch scheduler step
#'
#' Performs a single denoising step using Euler integration. This is the
#' core sampling function for FlowMatch models.
#'
#' @param model_output torch tensor. The output from the diffusion model
#'   (velocity prediction).
#' @param timestep Numeric. The current timestep.
#' @param sample torch tensor. The current noisy sample.
#' @param schedule List. The FlowMatch scheduler object.
#' @param generator torch generator or NULL. Random generator for reproducibility.
#'
#' @return A list containing:
#'   \describe{
#'     \item{prev_sample}{The denoised sample at the previous timestep}
#'     \item{schedule}{The updated scheduler with incremented step_index}
#'   }
#'
#' @details
#' The FlowMatch Euler step is remarkably simple:
#' \code{prev_sample = sample + dt * model_output}
#' where \code{dt = sigma_next - sigma_current}.
#'
#' This implements the Euler method for solving the probability flow ODE
#' in continuous normalizing flows.
#'
#' @export
flowmatch_scheduler_step <- function(
  model_output,
  timestep,
  sample,
  schedule,
  generator = NULL
) {
  # Initialize step index if needed
  if (is.null(schedule$step_index)) {
    schedule$step_index <- .flowmatch_init_step_index(timestep, schedule)
  }

  # Upcast sample for precision
  sample <- sample$to(dtype = torch::torch_float32())

  # Get current and next sigma
  sigma_idx <- schedule$step_index
  sigma <- schedule$sigmas[sigma_idx]$item()
  sigma_next <- schedule$sigmas[sigma_idx + 1]$item()

  # Compute dt (step size)
  dt <- sigma_next - sigma

  # Euler step: x_{t-1} = x_t + dt * v
  prev_sample <- sample + dt * model_output

  # Cast back to model dtype
  prev_sample <- prev_sample$to(dtype = model_output$dtype)

  # Increment step index
  schedule$step_index <- schedule$step_index + 1L

  list(
    prev_sample = prev_sample,
    schedule = schedule
  )
}

#' Scale noise for flow matching forward process
#'
#' Applies the forward process in flow-matching: interpolates between
#' the clean sample and noise.
#'
#' @param sample torch tensor. The clean sample.
#' @param timestep torch tensor. The current timestep.
#' @param noise torch tensor. The noise tensor.
#' @param schedule List. The FlowMatch scheduler object.
#'
#' @return torch tensor. The noisy sample at timestep t.
#' @export
flowmatch_scale_noise <- function(
  sample,
  timestep,
  noise,
  schedule
) {
  # Get sigma for this timestep
  sigmas <- schedule$sigmas$to(device = sample$device, dtype = sample$dtype)

  step_indices <- .flowmatch_index_for_timestep(timestep, schedule)
  sigma <- sigmas[step_indices]

  # Expand sigma dimensions to match sample
  while (length(sigma$shape) < length(sample$shape)) {
    sigma <- sigma$unsqueeze(- 1)
  }

  # Flow matching interpolation: x_t = (1 - sigma) * x_0 + sigma * noise
  noisy_sample <- (1.0 - sigma) * sample + sigma * noise

  noisy_sample
}

# Internal helper functions

.flowmatch_time_shift <- function(
  mu,
  sigma,
  t,
  shift_type
) {
  if (shift_type == "exponential") {
    exp(mu) / (exp(mu) + (1 / t - 1) ^ sigma)
  } else {
    # linear
    mu / (mu + (1 / t - 1) ^ sigma)
  }
}

.flowmatch_stretch_shift_to_terminal <- function(
  t,
  shift_terminal
) {
  one_minus_z <- 1 - t
  scale_factor <- one_minus_z[length(one_minus_z)] / (1 - shift_terminal)
  stretched_t <- 1 - (one_minus_z / scale_factor)
  stretched_t
}

.flowmatch_init_step_index <- function(
  timestep,
  schedule
) {
  # Find index for this timestep
  idx <- .flowmatch_index_for_timestep(timestep, schedule)
  idx
}

.flowmatch_index_for_timestep <- function(
  timestep,
  schedule
) {
  timesteps <- schedule$timesteps

  if (inherits(timestep, "torch_tensor")) {
    timestep <- timestep$item()
  }

  # Find matching index
  indices <- which(abs(as.numeric(timesteps) - timestep) < 1e-6)

  if (length(indices) == 0) {
    # Find closest
    indices <- which.min(abs(as.numeric(timesteps) - timestep))
  }

  # Return first match (or second if multiple, for numerical stability)
  if (length(indices) > 1) {
    indices[2]
  } else {
    indices[1]
  }
}

