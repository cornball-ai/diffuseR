#' Image-to-Image Generation with Stable Diffusion
#' 
#' This function generates an image based on an input image and a text prompt using the Stable Diffusion model.
#' It allows for various configurations such as model name, device, scheduler, and more.
#' @param input_image Path to the input image or a tensor representing the image.
#' @param prompt Text prompt to guide the image generation.
#' @param negative_prompt Optional negative prompt to guide the image generation.`
#' @param img_dim Dimension of the output image (default: 512).
#' @param model_name Name of the Stable Diffusion model to use (default: "stable-diffusion-2-1").
#' @param devices A named list of devices for each model component (e.g., `list(unet = "cuda", decoder = "cpu", text_encoder = "cpu", encoder = "cpu")`).
#' @param unet_dtype_str Optional A character for dtype of the unet component (typically "torch_float16" for cuda and "torch_float32" for cpu).
#' @param scheduler Scheduler to use for the diffusion process (default: "ddim").
#' @param scheduler_params Additional parameters for the scheduler (default: empty list).
#' @param num_inference_steps Number of diffusion steps (default: 50).
#' @param strength Strength of the image-to-image transformation (default: 0.8).
#' @param guidance_scale Scale for classifier-free guidance (default: 7.5).
#' @param seed Random seed for reproducibility (default: NULL).
#' @param save_to Path to save the generated image (default: NULL).
#' @param metadata_path Path to save metadata (default: NULL).
#' @param ... Additional arguments for future use.
#' @return Generated image tensor.
#' @examples
#' # Example usage
#' img <- img2img(
#'          input_image = "path/to/input.jpg",
#'          prompt = "A fantasy landscape",
#'          model_name = "stable-diffusion-2-1",
#'          device = "cuda",
#'          scheduler = "ddim",
#'          num_inference_steps = 50,
#'          strength = 0.8,
#'          guidance_scale = 7.5,
#'          seed = 42,
#'          save_to = "output.jpg")
#`
#' @export

img2img <- function(input_image,
                    prompt,
                    negative_prompt = NULL,
                    img_dim = 512,
                    model_name = "stable-diffusion-2-1",
                    devices = "cpu",
                    unet_dtype_str = "float16",
                    scheduler = "ddim",
                    num_inference_steps = 50,
                    strength = 0.8,
                    guidance_scale = 7.5,
                    seed = NULL,
                    save_to = NULL,
                    metadata_path = NULL,
                    ...) {
  # 1. Get models
  m2d <- models2devices(model_name, devices = "cpu", unet_dtype_str = NULL)
  model_dir <- m2d$model_dir
  model_files <- m2d$model_files
  devices <- m2d$devices
  unet_dtype <- m2d$unet_dtype
  device_cpu <- m2d$device_cpu
  device_cuda <- m2d$device_cuda
  
  if(model_name == "stable-diffusion-2-1") {
    num_train_timesteps <- 1000
  } else {
    stop("Model not supported")
  }
  
  # 2. Encode input image to latents
  image_tensor <- preprocess_image(input_image,
                                   device = torch::torch_device(devices$encoder))  # Resize & normalize
  message("Loading encoder...")
  encoder <- load_model_component("encoder", model_name,
                                  torch::torch_device(devices$encoder))
  message("Encoding image...")
  init_latents <- encoder(image_tensor)
  
  # 3. Compute noise timestep from strength
  t_strength <- as.integer(strength * num_train_timesteps)
  scheduler_cfg <- ddim_scheduler_create(num_train_timesteps = 1000,
                                         num_inference_steps = num_inference_steps,
                                         beta_schedule = "scaled_linear",
                                         device = torch::torch_device(devices$unet))
  
  all_inference_timesteps <- scheduler_cfg$timesteps
  timestep_idx <- which.min(abs(scheduler_cfg$timesteps - t_strength))
  timestep_start <- scheduler_cfg$timesteps[timestep_idx]
  timesteps <- scheduler_cfg$timesteps[timestep_idx:length(all_inference_timesteps)]
  
  # 4. Add noise to latents
  message("Adding noise to latent image...")
  if (!is.null(seed)) set.seed(seed)
  noised_latents <- scheduler_add_noise(original_latents = init_latents,
                                        noise = torch::torch_randn_like(init_latents),
                                        timestep = timestep_start,
                                        scheduler_obj = scheduler_cfg)
  noised_latents <- noised_latents$to(dtype = unet_dtype,
                                      device = torch::torch_device(devices$unet))
  txt2img(
    prompt = prompt,
    negative_prompt = negative_prompt,
    img_dim = img_dim,
    model_name = model_name,
    devices = devices,
    unet_dtype_str = unet_dtype_str,
    scheduler = scheduler,
    timesteps = timesteps,
    initial_latents = noised_latents,
    num_inference_steps = num_inference_steps,
    guidance_scale = guidance_scale,
    seed = seed,
    save_to = save_to,
    metadata_path = NULL
  )
}
