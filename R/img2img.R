#' Image-to-Image Generation with Stable Diffusion
#' 
#' This function generates an image based on an input image and a text prompt using the Stable Diffusion model.
#' It allows for various configurations such as model name, device, scheduler, and more.
#' @param input_image Path to the input image or a tensor representing the image.
#' @param prompt Text prompt to guide the image generation.
#' @param negative_prompt Optional negative prompt to guide the image generation.`
#' @param img_dim Dimension of the output image (default: 512).
#' @param model_name Name of the Stable Diffusion model to use (default: "sd21").
#' @param devices A named list of devices for each model component (e.g., `list(unet = "cuda", decoder = "cpu", text_encoder = "cpu", encoder = "cpu")`).
#' @param unet_dtype_str Optional A character for dtype of the unet component (typically "torch_float16" for cuda and "torch_float32" for cpu).
#' @param scheduler Scheduler to use for the diffusion process (default: "ddim").
#' @param num_inference_steps Number of diffusion steps (default: 50).
#' @param strength Strength of the image-to-image transformation (default: 0.8).
#' @param guidance_scale Scale for classifier-free guidance (default: 7.5).
#' @param seed Random seed for reproducibility (default: NULL).
#' @param save_file Logical indicating whether to save the generated image.
#' @param filename Optional filename for saving the image. If `NULL`, a default name is generated.
#' @param metadata_path Path to save metadata (default: NULL).
#' @param ... Additional arguments for future use.
#' @return An image array and metadata
#' @export

img2img <- function(input_image,
                    prompt,
                    negative_prompt = NULL,
                    img_dim = 512,
                    model_name = c("sd21", "sdxl"),
                    devices = "cpu",
                    unet_dtype_str = "float16",
                    scheduler = "ddim",
                    num_inference_steps = 50,
                    strength = 0.8,
                    guidance_scale = 7.5,
                    seed = NULL,
                    save_file = TRUE,
                    filename = NULL,
                    metadata_path = NULL,
                    ...) {
  # 1. Get models
  m2d <- models2devices(model_name, devices = devices, unet_dtype_str = NULL)
  model_dir <- m2d$model_dir
  model_files <- m2d$model_files
  devices <- m2d$devices
  unet_dtype <- m2d$unet_dtype
  device_cpu <- m2d$device_cpu
  device_cuda <- m2d$device_cuda
  
  if(model_name %in% c("sd21", "sdxl")) {
    num_train_timesteps <- 1000
  } else {
    stop("Model not supported")
  }
  
  # 2. Encode input image to latents
  image_tensor <- preprocess_image(input_image, width = img_dim, height = img_dim,
                                   device = torch::torch_device(devices$encoder))  # Resize & normalize
  message("Loading encoder...")
  encoder <- load_model_component("encoder", model_name,
                                  torch::torch_device(devices$encoder))
  message("Encoding image...")
  encoded <- encoder(image_tensor)
  message("Loading quant_conv...")
  conv_latents <- quant_conv(encoded, dtype = unet_dtype,
                             device = devices$unet)

  latents_mean <- conv_latents[, 1:4, , ]           # First 4 channels
  latents_log_var <- conv_latents[, 5:8, , ]        # Last 4 channels
  init_latents <- latents_mean$to(dtype = unet_dtype,
                                   device = torch::torch_device(devices$unet)) * 0.18215
  # Need to FIX
  # Sample from the distribution (reparameterization trick)
  # if(eps > 0){
  #   std <- torch_exp(0.5 * latents_log_var)
  #   eps <- torch_randn_like(std)
  #   sampled_latents <- mean + eps * std
  # }
  
  # 3. Compute noise timestep from strength
  t_strength <- as.integer(strength * num_train_timesteps)
  schedule <- ddim_scheduler_create(num_train_timesteps = 1000,
                                    num_inference_steps = num_inference_steps,
                                    beta_schedule = "scaled_linear",
                                    device = torch::torch_device(devices$unet))
  
  all_inference_timesteps <- schedule$timesteps
  timestep_idx <- which.min(abs(all_inference_timesteps - t_strength))
  timestep_start <- all_inference_timesteps[timestep_idx]
  timesteps <- all_inference_timesteps[timestep_idx:length(all_inference_timesteps)]
  
  # 4. Add noise to latents
  message("Adding noise to latent image...")
  if (!is.null(seed)) set.seed(seed)
  noised_latents <- scheduler_add_noise(original_latents = init_latents,
                                        noise = torch::torch_randn_like(init_latents),
                                        timestep = timestep_start,
                                        scheduler_obj = schedule)
  noised_latents <- noised_latents$to(dtype = unet_dtype,
                                      device = torch::torch_device(devices$unet))
  
  txt2img(
    prompt = prompt,
    negative_prompt = negative_prompt,
    img_dim = img_dim,
    model_name = model_name,
    devices = devices,
    unet_dtype_str = unet_dtype_str,
    scheduler = "ddim",
    timesteps = timesteps,
    initial_latents = noised_latents,
    num_inference_steps = num_inference_steps,
    guidance_scale = guidance_scale,
    seed = seed,
    save_file = save_file,
    filename = filename,
    metadata_path = metadata_path
  )
}
