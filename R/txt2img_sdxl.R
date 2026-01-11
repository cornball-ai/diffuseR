#' Generate an image from a text prompt using a diffusion pipeline
#'
#' @param prompt A character string prompt describing the image to generate.
#' @param negative_prompt Optional negative prompt to guide the generation.
#' @param img_dim Dimension of the output image (e.g., 512 for 512x512).
#' @param pipeline Optional A pre-loaded diffusion pipeline. If `NULL`, it will be loaded based on the model name and devices.
#' @param devices A named list of devices for each model component (e.g., `list(unet = "cuda", decoder = "cpu", text_encoder = "cpu")`).
#' @param unet_dtype_str Optional A character for dtype of the unet component (typically "float16" for cuda and "float32" for cpu; float32 is available for cuda).
#' @param download_models Logical indicating whether to download the model files if they are not found.
#' @param scheduler Scheduler to use (e.g., `"ddim"`, `"euler"`).
#' @param timesteps Optional A vector of timesteps to use.
#' @param initial_latents Optional initial latents for the diffusion process.
#' @param num_inference_steps Number of inference steps to run.
#' @param guidance_scale Scale for classifier-free guidance (typically 7.5).
#' @param seed Optional seed for reproducibility.
#' @param save_file Logical indicating whether to save the generated image.
#' @param filename Optional filename for saving the image. If `NULL`, a default name is generated.
#' @param metadata_path Optional file path to save metadata.
#' @param use_native_decoder Logical; if TRUE, uses native R torch decoder instead of TorchScript.
#'   Native decoder has better GPU compatibility (especially Blackwell).
#' @param use_native_text_encoder Logical; if TRUE, uses native R torch text encoder instead of TorchScript.
#'   Native text encoder has better GPU compatibility (especially Blackwell).
#' @param use_native_unet Logical; if TRUE, uses native R torch UNet instead of TorchScript.
#'   Native UNet has better GPU compatibility (especially Blackwell).
#' @param ... Additional parameters passed to the diffusion process.
#'
#' @return An image array and metadata
#' @export
#'
#' @examples
#' \dontrun{
#' img <- txt2img("a cat wearing sunglasses in space", device = "cuda")
#' }
txt2img_sdxl <- function(prompt,
                         negative_prompt = NULL,
                         img_dim = 1024,
                         pipeline = NULL,
                         devices = "auto",
                         unet_dtype_str = NULL,
                         download_models = FALSE,
                         scheduler = "ddim",
                         timesteps = NULL,
                         initial_latents = NULL,
                         num_inference_steps = 30,
                         guidance_scale = 7.5,
                         seed = NULL,
                         save_file = TRUE,
                         filename = NULL,
                         metadata_path = NULL,
                         use_native_decoder = FALSE,
                         use_native_text_encoder = FALSE,
                         use_native_unet = FALSE,
                         ...) {
  model_name <- "sdxl"

  # Handle "auto" devices
  if (identical(devices, "auto")) {
    devices <- auto_devices(model_name)
  }

  m2d <- models2devices(model_name = model_name, devices = devices,
                        unet_dtype_str = unet_dtype_str)
  model_dir <- m2d$model_dir
  model_files <- m2d$model_files
  devices <- m2d$devices
  unet_dtype <- m2d$unet_dtype
  device_cpu <- m2d$device_cpu
  device_cuda <- m2d$device_cuda
  
  if(is.null(pipeline)){
    pipeline <- load_pipeline(model_name = model_name, m2d = m2d,
                              unet_dtype_str = unet_dtype_str,
                              use_native_decoder = use_native_decoder,
                              use_native_text_encoder = use_native_text_encoder,
                              use_native_unet = use_native_unet)
  }
  
  
  # Start timing
  start_time <- proc.time()
  # Process text prompt
  message("Processing prompt...")
  ## Tokenizer
  tokens <- CLIPTokenizer(prompt)
  prompt_embed1 <- pipeline$text_encoder(tokens)
  te2_output <- pipeline$text_encoder2(tokens)
  prompt_embed2 <- te2_output[[1]]
  prompt_embed <- torch::torch_cat(list(prompt_embed1, prompt_embed2), dim = 3)
  
  text_embeds <- te2_output[[2]]
  text_embeds <- text_embeds$to(dtype = unet_dtype,
                                        device = torch::torch_device(devices$unet))
  time_ids = torch::torch_tensor(c(img_dim, img_dim, 0, 0, img_dim, img_dim), # zero indexed as python
                                 dtype=unet_dtype,
                                 device = torch::torch_device(devices$unet))$unsqueeze(1) 
  # clip-vit-large-patch14
  if (is.null(negative_prompt)) {
    empty_tokens <- CLIPTokenizer("")
    empty_prompt_embed <- torch::torch_zeros_like(prompt_embed)
    empty_text_embeds <- torch::torch_zeros_like(text_embeds)
  } else {
    empty_tokens <- CLIPTokenizer(negative_prompt)
    empty_prompt_embed1 <- pipeline$text_encoder(empty_tokens)
    empty_te2_output <- pipeline$text_encoder2(empty_tokens)
    empty_prompt_embed2 <- empty_te2_output[[1]]
    empty_prompt_embed <- torch::torch_cat(list(empty_prompt_embed1,
                                                empty_prompt_embed2), dim = 3)
    empty_text_embeds <- empty_te2_output[[2]]
  }
  empty_prompt_embed <- empty_prompt_embed$to(dtype = unet_dtype,
                                              device = torch::torch_device(devices$unet))
  prompt_embed       <- prompt_embed$to(dtype = unet_dtype,
                                        device = torch::torch_device(devices$unet))
  empty_text_embeds <- empty_text_embeds$to(dtype = unet_dtype,
                                            device = torch::torch_device(devices$unet))

  message("Creating schedule...")
  # Load scheduler
  schedule <- ddim_scheduler_create(num_inference_steps = num_inference_steps,
                                    beta_schedule = "scaled_linear",
                                    beta_start = 0.00085,
                                    beta_end = 0.012,
                                    rescale_betas_zero_snr = FALSE,
                                    device = torch::torch_device(devices$unet))
  if(is.null(timesteps)){
    timesteps <- schedule$timesteps
  }

  # Run diffusion process
  message("Generating image...")
  if(!is.null(seed)){
    set.seed(seed)
    torch::torch_manual_seed(seed = seed)
  }
  
  latent_dim <- img_dim / 8
  if(!is.null(initial_latents)){
    latents <- initial_latents
    latents <- latents$to(dtype = unet_dtype, device = torch::torch_device(devices$unet))
  } else {
    # Create random latents
    latents <- torch::torch_randn(c(1, 4, latent_dim, latent_dim),
                                  dtype = unet_dtype,
                                  device = torch::torch_device(devices$unet))
  }
  # Denoising loop (no gradients needed for inference)
  pb <- utils::txtProgressBar(min = 0, max = length(timesteps), style = 3)
  torch::with_no_grad({
    for (i in seq_along(timesteps)){
      timestep <- torch::torch_tensor(timesteps[i],
                                      dtype = torch::torch_long(),
                                      device = torch::torch_device(devices$unet))

      # Get both conditional and unconditional predictions
      noise_pred_cond   <- pipeline$unet(latents, timestep, prompt_embed,
                                         text_embeds, time_ids)

      if(guidance_scale != 1){
        # If guidance scale is not 1, we need to calculate the unconditional prediction
        # with an empty prompt
        noise_pred_uncond <- pipeline$unet(latents, timestep, empty_prompt_embed,
                                           empty_text_embeds, time_ids)
        # CFG step
        noise_pred <- noise_pred_uncond + guidance_scale *
                        (noise_pred_cond - noise_pred_uncond)
      } else {
        # If guidance scale is 1, we can use the conditional prediction directly
        noise_pred <- noise_pred_cond
      }

      # Calculating latent
      latents <- ddim_scheduler_step(model_output = noise_pred,
                                     timestep = timestep,
                                     sample = latents,
                                     schedule = schedule,
                                     prediction_type = "epsilon",
                                     device = devices$unet)
      latents <- latents$to(dtype = unet_dtype, device = torch::torch_device(devices$unet))
      utils::setTxtProgressBar(pb, i)
    }
  })
  close(pb)
  
  # Decode latents to image
  scaled_latent <- latents / 0.18215
  scaled_latent <- scaled_latent$to(dtype = torch::torch_float32(),
                                    device = torch::torch_device(devices$decoder))

  # message("Loading post_quant_conv...")
  post_conv_latent <- post_quant_conv(x = scaled_latent,
                                      dtype = torch::torch_float32(),
                                      device = devices$decoder)
  message("Decoding image...")
  decoded_output <- pipeline$decoder(post_conv_latent)
  # Ensure tensor is on CPU
  img <- decoded_output$cpu()
  
  # Remove batch dimension if present
  if (length(img$shape) == 4) {
    img <- img$squeeze(1)
  }
  
  # Reorder channels: [3, H, W] → [H, W, 3]
  img <- img$permute(c(2, 3, 1))
  
  # Normalize
  img <- (img + 1) / 2  # scale from [-1, 1] → [0, 1]
  img <- torch::torch_clamp(img, min = 0, max = 1)
  
  # Convert to R array
  img_array <- as.array(img)
  # Save if requested
  if (save_file) {
    # Creating filename while we're here
    if (is.null(filename)) {
      filename <- filename_from_prompt(prompt, datetime = TRUE)
    }
    message("Saving image to ", filename)
    save_image(img = img_array, filename)
  } else {
    # Display in RStudio Viewer if interactive
    if (interactive()) {
      grid::grid.raster(img_array)
    }
  }
  # Save metadata
  metadata <- list(
    prompt = prompt,
    negative_prompt = negative_prompt,
    width = img_dim,
    height = img_dim,
    num_inference_steps = num_inference_steps,
    guidance_scale = guidance_scale,
    seed = seed,
    scheduler = scheduler,
    model = model_name,
    filename = filename
  )
  if (!is.null(metadata_path)) {
    utils::write.csv(metadata, file = metadata_path, row.names = FALSE)
    message("Metadata saved to: ", metadata_path)
  }
  # Report timing
  elapsed <- proc.time() - start_time
  message(sprintf("Image generated in %.2f seconds", elapsed[3]))
  
  # Return the generated image and metadata
  return(list(
    image = img_array,
    metadata = metadata
  ))
}
