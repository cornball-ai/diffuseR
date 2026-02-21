#' Inpaint an image using Stable Diffusion 2.1
#'
#' Generates a new image by inpainting masked regions of an input image guided
#' by a text prompt. Uses the standard SD 2.1 pipeline with mask blending at
#' each denoising step.
#'
#' @param input_image Path to the input image (.jpg/.jpeg/.png) or a 3D array.
#' @param mask_image Path to the mask image or a matrix/array. White (1) = inpaint,
#'   Black (0) = keep.
#' @param prompt Text prompt to guide the inpainting.
#' @param negative_prompt Optional negative prompt.
#' @param img_dim Dimension of the output image (default: 512).
#' @param pipeline Optional pre-loaded pipeline. If NULL, loaded automatically.
#' @param devices A named list of devices for each model component, or "auto".
#' @param unet_dtype_str Data type for the UNet (e.g., "float16", "float32").
#' @param download_models Logical indicating whether to download models if not found.
#' @param num_inference_steps Number of diffusion steps (default: 50).
#' @param strength Strength of the transformation (default: 0.8). Higher values
#'   change the masked region more.
#' @param guidance_scale Scale for classifier-free guidance (default: 7.5).
#' @param seed Random seed for reproducibility.
#' @param save_file Logical indicating whether to save the generated image.
#' @param filename Optional filename for saving the image.
#' @param metadata_path Path to save metadata.
#' @param use_native_decoder Logical; if TRUE, uses native R torch decoder.
#' @param use_native_text_encoder Logical; if TRUE, uses native R torch text encoder.
#' @param use_native_unet Logical; if TRUE, uses native R torch UNet.
#' @param ... Additional arguments.
#'
#' @return A list containing the generated image array and metadata.
#' @export
inpaint_sd21 <- function(
  input_image,
  mask_image,
  prompt,
  negative_prompt = NULL,
  img_dim = 512,
  pipeline = NULL,
  devices = "auto",
  unet_dtype_str = NULL,
  download_models = FALSE,
  num_inference_steps = 50,
  strength = 0.8,
  guidance_scale = 7.5,
  seed = NULL,
  save_file = TRUE,
  filename = NULL,
  metadata_path = NULL,
  use_native_decoder = FALSE,
  use_native_text_encoder = FALSE,
  use_native_unet = FALSE,
  ...
) {
  model_name <- "sd21"
  num_train_timesteps <- 1000

  # Handle "auto" devices
  if (identical(devices, "auto")) {
    devices <- auto_devices(model_name)
  }

  # 1. Get models
  m2d <- models2devices(model_name, devices = devices, unet_dtype_str = NULL,
    download_models = download_models)
  devices <- m2d$devices
  unet_dtype <- m2d$unet_dtype
  device_cpu <- m2d$device_cpu
  device_cuda <- m2d$device_cuda

  if (is.null(pipeline)) {
    pipeline <- load_pipeline(model_name = model_name, m2d = m2d,
      i2i = TRUE,
      unet_dtype_str = unet_dtype_str,
      use_native_decoder = use_native_decoder,
      use_native_text_encoder = use_native_text_encoder,
      use_native_unet = use_native_unet)
  }

  # Start timing
  start_time <- proc.time()

  # 2. Encode input image to latents
  image_tensor <- preprocess_image(input_image, width = img_dim, height = img_dim,
    device = torch::torch_device(devices$encoder))
  message("Encoding image...")
  encoded <- pipeline$encoder(image_tensor)
  conv_latents <- quant_conv(encoded, dtype = unet_dtype, device = devices$unet)

  latents_mean <- conv_latents[, 1:4, , ]
  init_latents <- latents_mean$to(dtype = unet_dtype,
    device = torch::torch_device(devices$unet)) * 0.18215

  # 3. Preprocess mask to latent dimensions
  message("Preprocessing mask...")
  mask_latent <- preprocess_mask(mask_image, height = img_dim, width = img_dim,
    device = devices$unet, dtype = unet_dtype)

  # 4. Compute noise timestep from strength
  t_strength <- as.integer(strength * num_train_timesteps)
  schedule <- ddim_scheduler_create(num_train_timesteps = 1000,
    num_inference_steps = num_inference_steps,
    beta_schedule = "scaled_linear",
    device = torch::torch_device(devices$unet))

  all_inference_timesteps <- schedule$timesteps
  timestep_idx <- which.min(abs(all_inference_timesteps - t_strength))
  timestep_start <- all_inference_timesteps[timestep_idx]
  timesteps <- all_inference_timesteps[timestep_idx:length(all_inference_timesteps)]

  # 5. Add noise to latents
  message("Adding noise to latent image...")
  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed = seed)
  }
  noise <- torch::torch_randn_like(init_latents)
  noised_latents <- scheduler_add_noise(original_latents = init_latents,
    noise = noise,
    timestep = timestep_start,
    scheduler_obj = schedule)
  latents <- noised_latents$to(dtype = unet_dtype,
    device = torch::torch_device(devices$unet))

  # 6. Process text prompt
  message("Processing prompt...")
  tokens <- CLIPTokenizer(prompt)
  prompt_embed <- pipeline$text_encoder(tokens)

  if (is.null(negative_prompt)) {
    empty_tokens <- CLIPTokenizer("")
  } else {
    empty_tokens <- CLIPTokenizer(negative_prompt)
  }
  empty_prompt_embed <- pipeline$text_encoder(empty_tokens)

  empty_prompt_embed <- empty_prompt_embed$to(dtype = unet_dtype,
    device = torch::torch_device(devices$unet))
  prompt_embed <- prompt_embed$to(dtype = unet_dtype,
    device = torch::torch_device(devices$unet))

  # 7. Denoising loop with mask blending
  message("Inpainting...")
  pb <- utils::txtProgressBar(min = 0, max = length(timesteps), style = 3)
  torch::with_no_grad({
    for (i in seq_along(timesteps)) {
      timestep <- torch::torch_tensor(timesteps[i],
        dtype = torch::torch_long(),
        device = torch::torch_device(devices$unet))

      # CFG: get conditional and unconditional predictions
      noise_pred_uncond <- pipeline$unet(latents, timestep, empty_prompt_embed)
      noise_pred_cond <- pipeline$unet(latents, timestep, prompt_embed)

      noise_pred <- noise_pred_uncond + guidance_scale *
        (noise_pred_cond - noise_pred_uncond)

      # DDIM step
      latents <- ddim_scheduler_step(model_output = noise_pred,
        timestep = timestep,
        sample = latents,
        schedule = schedule,
        prediction_type = "v_prediction",
        device = devices$unet)
      latents <- latents$to(dtype = unet_dtype,
        device = torch::torch_device(devices$unet))

      # Mask blending: keep original in unmasked regions
      # mask=1 means inpaint (use denoised), mask=0 means keep (use original)
      # Re-noise original latents to current timestep for seamless blending
      if (i < length(timesteps)) {
        next_timestep <- timesteps[i + 1]
        original_at_t <- scheduler_add_noise(
          original_latents = init_latents,
          noise = noise,
          timestep = next_timestep,
          scheduler_obj = schedule)
        original_at_t <- original_at_t$to(dtype = unet_dtype,
          device = torch::torch_device(devices$unet))
      } else {
        # Final step: use clean original latents
        original_at_t <- init_latents
      }
      latents <- latents * mask_latent + original_at_t * (1 - mask_latent)

      utils::setTxtProgressBar(pb, i)
    }
  })
  close(pb)

  # 8. Decode latents to image
  scaled_latent <- latents / 0.18215
  scaled_latent <- scaled_latent$to(dtype = torch::torch_float32(),
    device = torch::torch_device(devices$decoder))
  message("Decoding image...")
  decoded_output <- pipeline$decoder(scaled_latent)
  img <- decoded_output$cpu()

  if (length(img$shape) == 4) {
    img <- img$squeeze(1)
  }

  img <- img$permute(c(2, 3, 1))
  img <- (img + 1) / 2
  img <- torch::torch_clamp(img, min = 0, max = 1)
  img_array <- as.array(img)

  # Save if requested
  if (save_file) {
    if (is.null(filename)) {
      filename <- filename_from_prompt(prompt, datetime = TRUE)
    }
    message("Saving image to ", filename)
    save_image(img = img_array, filename)
  } else {
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
    strength = strength,
    guidance_scale = guidance_scale,
    seed = seed,
    model = model_name,
    mode = "inpaint"
  )
  if (!is.null(metadata_path)) {
    utils::write.csv(metadata, file = metadata_path, row.names = FALSE)
    message("Metadata saved to: ", metadata_path)
  }

  elapsed <- proc.time() - start_time
  message(sprintf("Inpainting completed in %.2f seconds", elapsed[3]))

  list(
    image = img_array,
    metadata = metadata
  )
}
