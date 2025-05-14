#' Generate an image from a text prompt using a diffusion pipeline
#'
#' @param prompt A character string prompt describing the image to generate.
#' @param negative_prompt Optional negative prompt to guide the generation.
#' @param img_dim Dimension of the output image (e.g., 512 for 512x512).
#' @param model_name Name of the model to use (e.g., `"stable-diffusion-2-1"`).
#' @param devices A named list of devices for each model component (e.g., `list(unet = "cuda", decoder = "cpu", text_encoder = "cpu")`).
#' @param unet_dtype Optional A character for dtype of the unet component (typically "torch_float16" for cuda and "torch_float32" for cpu).
#' @param scheduler Scheduler to use (e.g., `"ddim"`, `"euler"`).
#' @param scheduler_params A named list of parameters for the scheduler.
#' @param num_inference_steps Number of inference steps to run.
#' @param guidance_scale Scale for classifier-free guidance (typically 7.5).
#' @param seed Optional seed for reproducibility.
#' @param save_to Optional file path to save the final image.
#' @param metadata_path Optional file path to save metadata.
#' @param ... Additional parameters passed to the diffusion process.
#'
#' @return A tensor or image object, depending on implementation.
#' @export
#'
#' @examples
#' \dontrun{
#' img <- txt2img("a cat wearing sunglasses in space", device = "cuda")
#' }
txt2img <- function(prompt,
                    negative_prompt = NULL,
                    img_dim = 512,
                    model_name = "stable-diffusion-2-1",
                    devices = "cpu",
                    unet_dtype = NULL,
                    scheduler = "ddim",
                    scheduler_params = list(),
                    num_inference_steps = 50,
                    guidance_scale = 7.5,
                    seed = NULL,
                    save_to = "output.png",
                    metadata_path = NULL,
                    ...) {
  # Standardize the devices parameter
  if (is.character(devices) && length(devices) == 1) {
    devices <- list(unet = devices, decoder = devices, text_encoder = devices)
  } else if (!is.list(devices) || 
             !all(c("unet", "decoder", "text_encoder") %in% names(devices))) {
    stop(paste0("'devices' must be either a single device string or a named ",
                "list with 'unet', 'decoder', and 'text_encoder' elements"))
  }

  if(devices$unet == device_cpu){
    unet_dtype <- torch_float32() # aka torch_half() -> torch_Half 
  } else {
    if(device == device_cuda){
      if(is.null(unet_dtype)){
        unet_dtype <- torch_float16() # aka torch_half() -> torch_Half
      } else {
        if(unet_dtype == "torch_float16"){
          unet_dtype <- torch_float16() # aka torch_half() -> torch_Half
        } else if(unet_dtype == "torch_float32"){
          unet_dtype <- torch_float32() # aka torch_float() -> torch_Float
        } else {
          stop("Invalid dtype")
        }
      }
    } else {
      stop("Invalid device")
    }
  }
  
  # Check if the model is downloaded
  models <- download_model(model_name, devices, unet_dtype)
  model_dir <- models$model_dir
  model_files <- models$model_files
  # Start timing
  start_time <- proc.time()
  
  # Load models with specified devices
  message("Loading text_encoder...")
  text_encoder <- load_model_component("text_encoder", model_name,
                                       devices$text_encoder)
  
  # Process text prompt
  message("Processing prompt...")
  ## Tokenizer
  # tokens <- CLIPTokenizer(prompt,
  #                         merges = "inst/tokenizer/merges.txt",
  #                         vocab_file = "inst/tokenizer/vocab.json")
  tokens <- CLIPTokenizer(prompt)
  prompt_embed <- text_encoder(tokens)

  # empty_tokens <- CLIPTokenizer("",
  #                               merges = "inst/tokenizer/merges.txt",
  #                               vocab_file = "inst/tokenizer/vocab.json")
  if (is.null(negative_prompt)) {
    empty_tokens <- CLIPTokenizer("")
  } else {
    empty_tokens <- CLIPTokenizer(negative_prompt)
  }
  empty_prompt_embed <- text_encoder(empty_tokens)
  
  empty_prompt_embed <- empty_prompt_embed$to(device = devices$text_encoder)
  prompt_embed       <- prompt_embed$to(device = devices$text_encoder)

  message("Loading scheduler...")
  # Load scheduler  
  scheduler_cfg <- ddim_scheduler_create(num_inference_steps = num_inference_steps,
                                         beta_schedule = "scaled_linear")
  timesteps <- scheduler_cfg$timesteps

  # Load Unet
  message("Loading unet...")
  unet <- load_model_component("unet", model_name,
                               device = devices$unet,
                               unet_dtype = unet_dtype)
  
  # Run diffusion process
  message("Generating image...")
  if(!is.null(seed)){
    set.seed(seed)
    torch::torch_manual_seed(seed = seed)
  }
  
  latent_dim <- img_dim / 8
  latents <- torch::torch_randn(c(1, 4, latent_dim, latent_dim),
                                device = devices$unet)
  
  # Denoising loop
  pb <- utils::txtProgressBar(min = 0, max = length(scheduler_cfg$timesteps),
                              style = 3)
  for (i in seq_along(scheduler_cfg$timesteps)){
    timestep <- torch::torch_tensor(scheduler_cfg$timesteps[i],
                                    dtype = torch::torch_long(),
                                    device = devices$unet)
    
    # Get both conditional and unconditional predictions
    noise_pred_uncond <- unet(latents, timestep, empty_prompt_embed)
    noise_pred_cond   <- unet(latents, timestep, prompt_embed)
    
    # CFG step
    noise_pred <- noise_pred_uncond + guidance_scale *
                    (noise_pred_cond - noise_pred_uncond)
    
    # Calculating latent
    latents <- ddim_scheduler_step(model_output = noise_pred,
                                   timestep = timestep,
                                   sample = latents,
                                   scheduler_cfg = scheduler_cfg,
                                   prediction_type = "v_prediction")
    latents <- latents$to(dtype = unet_dtype, device = devices$unet)
    utils::setTxtProgressBar(pb, i)
  }
  close(pb)
  
  # Decode latents to image
  message("Loading decoder...")
  decoder <- load_model_component("decoder", model_name, devices$decoder)
  scaled_latent <- latents / 0.18215
  scaled_latent <- scaled_latent$to(dtype = torch_float32(), device = device)
  message("Decoding image...")
  decoded_output <- decoder(scaled_latent)
  # Ensure tensor is on CPU
  img <- decoded_output$cpu()
  
  # Remove batch dimension if present
  if (length(img$shape) == 4) {
    img <- img$squeeze(1)
  }
  
  # Reorder channels: [3, H, W] → [H, W, 3]
  img <- img$permute(c(2, 3, 1))
  
  # Normalize if needed
  img <- (img + 1) / 2  # scale from [-1, 1] → [0, 1]
  img <- torch::torch_clamp(img, min = 0, max = 1)
  
  # Convert to R array
  img_array <- as.array(img)
  # Save if requested
  if (!is.null(save_to)) {
    message("Saving image to ", save_to)
    save_image(img = img_array, save_to)
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
    model = model_name
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
