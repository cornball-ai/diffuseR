#' Create a DDIM Scheduler
#'
#' Creates a Denoising Diffusion Implicit Models (DDIM) scheduler for use with
#' diffusion models. DDIM schedulers provide a deterministic sampling process
#' that offers faster inference compared to DDPM while maintaining high quality
#' outputs.
#'
#' @param num_train_timesteps Integer. The number of diffusion steps used to
#'   train the model. Default: 1000
#' @param num_inference_steps Integer. The number of diffusion steps used for
#'   inference. Fewer steps typically means faster inference at the cost of
#'   sample quality. Default: 50
#' @param eta Numeric. Corresponds to η in DDIM paper, controls the amount of
#'   stochasticity. When eta=0, the sampling process is deterministic.
#'   When eta=1, the sampling process is equivalent to DDPM. Default: 0
#' @param beta_schedule Character. The beta schedule to use. Options are:
#'   \describe{
#'     \item{"linear"}{Linear beta schedule from beta_start to beta_end}
#'     \item{"scaled_linear"}{Scaled linear schedule, generally gives better results}
#'     \item{"cosine"}{Cosine schedule that approaches zero smoothly}
#'   }
#'   Default: "linear"
#' @param beta_start Numeric. The starting value for the beta schedule. Default: 0.00085
#' @param beta_end Numeric. The final value for the beta schedule. Default: 0.012
#' @param dtype The data type to use for computations. Default is torch_float32(). Options are torch_float16() and torch_float32().
#' @param device The device to use for computations. Options are torch_device("cpu"), torch_device("cuda").
#'
#' @return A DDIM scheduler object that can be used with diffusion models to
#'   generate samples.
#'
#' @details
#' DDIM (Denoising Diffusion Implicit Models) was introduced by Song et al. (2020)
#' as an extension to DDPM (Denoising Diffusion Probabilistic Models). It offers
#' a deterministic sampling process and allows for controlling the number of
#' inference steps independently from the training process.
#'
#' The scheduler contains the noise schedule and methods for computing
#' alpha, beta, and other parameters used in the diffusion process.
#'
#' @references
#' Song, J., Meng, C., & Ermon, S. (2020).
#' "Denoising Diffusion Implicit Models."
#' \url{https://arxiv.org/abs/2010.02502}
#'
#' @examples
#' #' \dontrun{
#' # Create a DDIM scheduler with custom parameters
#' scheduler <- ddim_scheduler_create(
#'   num_train_timesteps = 1000,
#'   num_inference_steps = 30,
#'   eta = 0.5,
#'   beta_schedule = "scaled_linear"
#' )
#' }
#' @export
ddim_scheduler_create <- function(num_train_timesteps = 1000,
                                  num_inference_steps = 50, eta = 0,
                                  beta_schedule = c("linear", "scaled_linear", "cosine"),
                                  beta_start =  0.00085, beta_end = 0.012,
                                  dtype = torch::torch_float32(),
                                  device = c(torch_device("cpu"),
                                             torch_device("cuda"))) {
  betas <- switch(beta_schedule,
                  "linear" = seq(beta_start, beta_end,
                                 length.out = num_train_timesteps),
                  "scaled_linear" = seq(sqrt(beta_start), sqrt(beta_end),
                                        length.out = num_train_timesteps) ^ 2,
                  "cosine" = NULL)
  betas <- torch::torch_tensor(betas, dtype = dtype, device = device)
  alphas <- 1 - betas
  alphas <- torch::torch_tensor(alphas, dtype = dtype, device = device)
  alphas_cumprod <- torch::torch_cumprod(alphas, dim = 1, dtype = dtype)
  alphas_cumprod_prev <- torch::torch_cat(list(torch::torch_ones(1, device = device),
                                               alphas_cumprod[1:(length(alphas_cumprod) - 1)]))
  
  step_ratio <- num_train_timesteps %/% num_inference_steps
  # Could add 2 here instead of 1 and remove it from timestep_index and prev_timestep_index below
  timesteps <- as.integer(rev(round((0:(num_inference_steps - 1)) * step_ratio) + 1))
  # timesteps <- as.integer(seq(num_train_timesteps, 0, length.out = num_inference_steps + 1)[-1] + 1)
  
  list(
    betas = betas,
    alphas = alphas,
    alphas_cumprod = alphas_cumprod,
    alphas_cumprod_prev = alphas_cumprod_prev,
    timesteps = timesteps,
    eta = eta
  )
}

#' Perform a DDIM scheduler step
#'
#' Performs a single denoising step using the DDIM (Denoising Diffusion Implicit Models)
#' algorithm. This function takes the output from a diffusion model at a specific timestep
#' and computes the previous (less noisy) sample in the diffusion process.
#'
#' @param model_output Numeric array. The output from the diffusion model, typically
#'   representing predicted noise or the denoised sample depending on `prediction_type`.
#' @param timestep Integer. The current timestep in the diffusion process.
#' @param sample Numeric array. The current noisy sample at timestep `t`.
#' @param eta Numeric. Controls the stochasticity of the process. When eta=0, DDIM is
#'   deterministic. When eta=1, it's equivalent to DDPM. Default: 0
#' @param use_clipped_model_output Logical. Whether to clip the model output before
#'   computing the sample update. Can improve stability. Default: FALSE
#' @param scheduler_cfg Logical. Whether to use classifier-free guidance with the
#'   scheduler. Default: TRUE
#' @param thresholding Logical. Whether to apply thresholding to the output.
#'   Default: FALSE
#' @param generator An optional random number generator for reproducibility.
#'   Default: NULL
#' @param variance_noise Optional pre-generated noise for the variance when eta > 0.
#'   If NULL and eta > 0, noise will be generated. Default: NULL
#' @param clip_sample Logical. Whether to clip the sample. Default: FALSE 
#' @param set_alpha_to_one Logical. Whether to override the final alpha value to 1.
#'   Used for numerical stability in the final step. Default: FALSE
#' @param prediction_type Character. The type of prediction the model outputs.
#'   Options are:
#'   \describe{
#'     \item{"epsilon"}{The model predicts the noise (ε)}
#'     \item{"sample"}{The model predicts the denoised sample directly}
#'     \item{"v_prediction"}{The model predicts the velocity vector (v)}
#'   }
#'   Default: "epsilon"
#' @param dtype The data type to use for computations. Default is torch_float32().
#' @param device The device to use for computations. Options are "cpu" and "cuda".
#'
#' @return A list containing:
#'   \describe{
#'     \item{`prev_sample`}{The less noisy sample at timestep t-1}
#'     \item{`pred_original_sample`}{The predicted denoised sample}
#'   }
#'
#' @details
#' The DDIM step function implements the core sampling algorithm of DDIM described in
#' Song et al. 2020. It computes the previous sample x_t-1 given the current
#' sample x_t and the model output.
#'
#' The algorithm differs from DDPM by using a non-Markovian diffusion process that
#' allows for deterministic sampling and fewer inference steps without sacrificing
#' quality.
#' 
#' When using `prediction_type="epsilon"` (most common), the model predicts the
#' noise that was added to create the current noisy sample. For `prediction_type="sample"`,
#' the model predicts the clean sample directly. The `v_prediction` option implements
#' the v-parameterization from Salimans & Ho (2022).
#'
#' @references
#' Song, J., Meng, C., & Ermon, S. (2020).
#' "Denoising Diffusion Implicit Models."
#' \url{https://arxiv.org/abs/2010.02502}
#'
#' Salimans, T., & Ho, J. (2022).
#' "Progressive Distillation for Fast Sampling of Diffusion Models."
#' \url{https://arxiv.org/abs/2202.00512}
#'
#' @examples
#' # Create a DDIM scheduler
#' scheduler <- ddim_scheduler_create()
#' 
#' # Assume we have a model output and current sample
#' # model_output <- predict_noise(model, sample, timestep)
#' 
#' # Perform a denoising step
#' result <- ddim_scheduler_step(
#'   model_output = model_output,
#'   timestep = timestep,
#'   sample = sample,
#'   eta = 0,  # Deterministic sampling
#'   prediction_type = "epsilon"
#' )
#' 
#' # Get the denoised sample for the next iteration
#' prev_sample <- result$prev_sample
#'
#' @export
ddim_scheduler_step <- function(model_output, timestep, sample, scheduler_cfg,
                                eta = 0,
                                use_clipped_model_output = FALSE,
                                thresholding = FALSE,
                                generator = NULL,
                                variance_noise = NULL,
                                clip_sample = FALSE,
                                set_alpha_to_one = FALSE,
                                prediction_type = c("epsilon", "sample",
                                                    "v_prediction"),
                                dtype = torch_float32(),
                                device = "cpu"){
  
  # 1. get previous step value (= timestep + 1); i.e. python-indexing
  # Need to cast timestep_index to long
  timestep_index <- torch::torch_tensor(timestep + 1,
                                        dtype = torch::torch_long(),
                                        device = torch::torch_device(device))
  if(as.numeric(timestep_index) <= 2){ #scheduler_cfg$timesteps[1]) {
    prev_timestep <- 1 #length(scheduler_cfg$alphas)
    prev_timestep_index <- torch::torch_tensor(prev_timestep,
                                               dtype = torch::torch_long(),
                                               device = torch::torch_device(device))
  } else {
    prev_timestep <- scheduler_cfg$timesteps[which(as.logical(scheduler_cfg$timesteps == (timestep))) + 1]
    prev_timestep_index <- torch::torch_tensor(prev_timestep + 1,
                                               dtype = torch::torch_long(),
                                               device = torch::torch_device(device))
  }
  
  # 2. compute alphas, betas
  alpha_prod_t <- scheduler_cfg$alphas_cumprod[timestep_index]
  alpha_prod_t_prev <- scheduler_cfg$alphas_cumprod[prev_timestep_index]
  # alpha_prod_t <- alpha_prod_t + 1e-7  # Prevent division by zero
  # alpha_prod_t_prev <- alpha_prod_t_prev + 1e-7
  if (set_alpha_to_one & prev_timestep == 1){
    alpha_prod_t_prev <- torch::torch_tensor(1.0,
                                             dtype = dtype,
                                             device = torch::torch_device(device))
  } else {
    alpha_prod_t_prev <- scheduler_cfg$alphas_cumprod[prev_timestep_index]
    alpha_prod_t_prev <- alpha_prod_t_prev$to(dtype = dtype, device = torch::torch_device(device))
  }
  
  beta_prod_t <- 1 - alpha_prod_t
  beta_prod_t <- beta_prod_t$to(dtype = dtype, device = torch::torch_device(device))
  beta_prod_t_prev = 1 - alpha_prod_t_prev
  beta_prod_t_prev <- beta_prod_t_prev$to(dtype = dtype, device = torch::torch_device(device))
  
  # 3. Handle different prediction types (epsilon or v-prediction)
  preds <- switch(prediction_type,
                  "epsilon" = list(pred_original_sample = (sample - beta_prod_t ^ (0.5) * model_output) / alpha_prod_t ^ (0.5),
                                   pred_epsilon = model_output),
                  "sample" = list(pred_original_sample = model_output,
                                  pred_epsilon = (sample - alpha_prod_t ^ (0.5) * model_output) / beta_prod_t ^ (0.5)),
                  "v_prediction" = list(pred_original_sample = (alpha_prod_t ^ 0.5) * sample - (beta_prod_t ^ 0.5) * model_output,
                                        pred_epsilon = (alpha_prod_t ^ 0.5) * model_output + (beta_prod_t ^ 0.5) * sample))
  pred_original_sample <- preds$pred_original_sample$to(dtype = dtype, device = torch::torch_device(device))
  pred_epsilon <- preds$pred_epsilon$to(dtype = dtype, device = torch::torch_device(device))
  # if self.config.prediction_type == "epsilon":
  # pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
  # pred_epsilon = model_output
  
  # 4. Clip or threshold "predicted x_0"
  # Check thresholding sample code
  if (thresholding) {
    flat <- pred_original_sample$view(c(pred_original_sample$size(1), -1))
    abs_flat <- torch::torch_abs(flat, device = torch::torch_device(device))
    s <- torch::torch_quantile(abs_flat,
                               torch::torch_tensor(0.995, device = torch::torch_device(device)),
                               dim = 2,
                               device = torch::torch_device(device)) # dim=2 because batch is dim=1 in R torch
    s <- torch::torch_max(s, device = torch::torch_device(device))
    s <- s$view(c(pred_original_sample$size(1), 1, 1, 1))
    pred_original_sample <- torch::torch_clamp(pred_original_sample,
                                               min = -s, max = s,
                                               device = torch::torch_device(device)) / s
  } else {
    if(clip_sample){
      # Clip to [-1, 1] range
      pred_original_sample <- torch::torch_clamp(pred_original_sample,
                                                 min = -1, max = 1,
                                                 device = torch::torch_device(device))
    }
  }
  
  # 5. compute variance: "sigma_t(η)" -> see formula (16)
  # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
  variance <- (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
  std_dev_t = eta * sqrt(variance)
  
  # Re-compute model_output (noise prediction) from clipped pred_original_sample if needed
  if(use_clipped_model_output) {
    if(prediction_type == "v_prediction") {
      # Recompute v-prediction from clipped pred_original_sample
      model_output <- sqrt(alpha_prod_t) * pred_epsilon - sqrt(beta_prod_t) * pred_original_sample
    } else {
      # Recompute epsilon from clipped pred_original_sample
      model_output <- (sample - sqrt(alpha_prod_t) * pred_original_sample) / sqrt(beta_prod_t)
    }
  }
  
  # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
  pred_sample_direction <- sqrt(1 - alpha_prod_t_prev - std_dev_t^2) * pred_epsilon
  
  # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
  prev_sample <- sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
  
  
  if (eta > 0) {
    if (!is.null(variance_noise) && !is.null(generator)) {
      stop("Provide either `generator` or `variance_noise`, not both.")
    }
    if (is.null(variance_noise)) {
      variance_noise <- torch::torch_randn(model_output$shape,
                                           generator = generator,
                                           device = torch::torch_device(device),
                                           dtype = dtype)
    }
    variance <- std_dev_t * eta * variance_noise
    prev_sample <- prev_sample + variance
  }
  
  prev_sample
}

#' Add noise to latents using DDIM scheduler
#' 
#' This function adds noise to the original latents according to the DDIM
#' scheduler's diffusion process. It computes the noisy latents based on the
#' original latents, noise, and the current timestep.
#' 
#' @param original_latents A torch tensor representing the original latents.
#' 
#' @param noise A torch tensor representing the noise to be added.
#' 
#' @param timestep An integer representing the current timestep in the diffusion
#' process.
#' 
#' @param scheduler_obj A list containing the DDIM scheduler parameters, including
#' alphas_cumprod and timesteps. The alphas_cumprod represents how much of the
#' original signal remains at each timestep of the diffusion process.
#' 
#' @return A torch tensor containing the noised latents, which represents the 
#' original latents with the appropriate amount of noise added for the given
#' timestep.
#' 
#' @details 
#' The noise is added according to the standard diffusion forward process formula:
#' noised_latents = sqrt(alpha_cumprod) * original_latents + sqrt(1-alpha_cumprod) * noise
#' 
#' Where alpha_cumprod is the cumulative product of (1-beta) values up to the 
#' specified timestep, with beta being the noise schedule.
#' 
#' @examples
#' \dontrun{
#' # Assuming we have latents, noise, and a scheduler
#' noised_latents <- scheduler_add_noise(
#'   original_latents = latents,
#'   noise = torch::torch_randn_like(latents),
#'   timestep = scheduler$timesteps[1],
#'   scheduler_obj = scheduler
#' )
#' }
#' 
#' @export
scheduler_add_noise <- function(original_latents, noise, timestep, scheduler_obj) {
  # Get the alpha_cumprod value for this timestep
  # In DDIM/DDPM schedulers, alphas_cumprod represents 
  # how much of the original signal remains at each timestep
  
  # Get alpha_cumprod for this timestep
  alpha_cumprod <- scheduler_obj$alphas_cumprod[timestep]
  
  # Calculate the noise scaling factors
  sqrt_alpha_prod <- torch::torch_sqrt(torch::torch_tensor(alpha_cumprod,
                                                           device = original_latents$device))
  sqrt_one_minus_alpha_prod <- torch::torch_sqrt(
    torch::torch_tensor(1 - alpha_cumprod, device = original_latents$device)
  )
  
  # Add noise to the original latents according to the diffusion formula:
  # noisy = sqrt(alpha) * original + sqrt(1-alpha) * noise
  noised_latents <- sqrt_alpha_prod * original_latents +
                      sqrt_one_minus_alpha_prod * noise
  
  return(noised_latents)
}
