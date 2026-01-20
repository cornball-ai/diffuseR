#' Generate Video from Text Prompt using LTX-2
#'
#' Generates video using the LTX-2 diffusion transformer model.
#'
#' @param prompt Character. Text prompt describing the video to generate.
#' @param negative_prompt Character. Optional negative prompt.
#' @param width Integer. Video width in pixels (default 768).
#' @param height Integer. Video height in pixels (default 512).
#' @param num_frames Integer. Number of frames to generate (default 121).
#' @param fps Numeric. Frames per second (default 24).
#' @param num_inference_steps Integer. Number of denoising steps (default 8 for distilled).
#' @param guidance_scale Numeric. CFG scale (default 4.0).
#' @param memory_profile Character or list. Memory profile: "auto" for auto-detection,
#'   or a profile from `ltx2_memory_profile()`.
#' @param text_backend Character. Text encoding backend: "gemma3" (native), "api", "precomputed", or "random".
#' @param text_model_path Character. Path to Gemma3 model (for "gemma3" backend). Supports glob patterns.
#' @param text_api_url Character. URL for text encoding API (if backend = "api").
#' @param vae Optional. Pre-loaded VAE module.
#' @param dit Optional. Pre-loaded DiT transformer module.
#' @param connectors Optional. Pre-loaded text connectors module.
#' @param seed Integer. Random seed for reproducibility.
#' @param output_file Character. Path to save output video (NULL for no save).
#' @param output_format Character. Output format: "mp4", "gif", or "frames".
#' @param return_latents Logical. If TRUE, also return final latents.
#' @param verbose Logical. Print progress messages.
#'
#' @return A list with:
#'   - `video`: Array of video frames [frames, height, width, channels]
#'   - `latents`: (if return_latents=TRUE) Final latent tensor
#'   - `metadata`: Generation metadata
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Basic usage
#' result <- txt2vid_ltx2("A cat walking on a beach at sunset")
#'
#' # With specific settings
#' result <- txt2vid_ltx2(
#'   prompt = "A timelapse of clouds moving over mountains",
#'   width = 512,
#'   height = 320,
#'   num_frames = 61,
#'   num_inference_steps = 8,
#'   seed = 42,
#'   output_file = "clouds.mp4"
#' )
#' }
txt2vid_ltx2 <- function(
  prompt,
  negative_prompt = NULL,
  width = 768L,
  height = 512L,
  num_frames = 121L,
  fps = 24.0,
  num_inference_steps = 8L,
  guidance_scale = 4.0,
  memory_profile = "auto",
  text_backend = "gemma3",
  text_model_path = NULL,
  text_api_url = NULL,
  vae = NULL,
  dit = NULL,
  connectors = NULL,
  seed = NULL,
  output_file = NULL,
  output_format = "mp4",
  return_latents = FALSE,
  verbose = TRUE
) {
  # Start timing
  start_time <- Sys.time()

  # Ensure integers
  width <- as.integer(width)
  height <- as.integer(height)
  num_frames <- as.integer(num_frames)
  num_inference_steps <- as.integer(num_inference_steps)

  # Set seed if provided
  if (!is.null(seed)) {
    torch::torch_manual_seed(seed)
    # torch_manual_seed sets both CPU and CUDA seeds
  }

  # Resolve memory profile
  if (identical(memory_profile, "auto")) {
    memory_profile <- ltx2_memory_profile()
  } else if (is.character(memory_profile)) {
    # Named profile
    vram <- switch(memory_profile,
      "high" = 20,
      "medium" = 12,
      "low" = 8,
      "very_low" = 6,
      "cpu_only" = 0,
      8# default
    )
    memory_profile <- ltx2_memory_profile(vram_gb = vram)
  }

  if (verbose) {
    message(sprintf("Using memory profile: %s", memory_profile$name))
  }

  # Validate and adjust resolution for profile
  validated <- validate_resolution(height, width, num_frames, memory_profile)
  if (validated$adjusted) {
    height <- validated$height
    width <- validated$width
    num_frames <- validated$num_frames
  }

  # LTX-2 VAE compression ratios
  spatial_ratio <- 32L
  temporal_ratio <- 8L

  # Calculate latent dimensions
  latent_height <- height %/% spatial_ratio
  latent_width <- width %/% spatial_ratio
  latent_frames <- (num_frames - 1L) %/% temporal_ratio + 1L

  if (verbose) {
    message(sprintf("Video: %dx%d, %d frames @ %.1f fps", width, height, num_frames, fps))
    message(sprintf("Latents: %dx%d, %d frames", latent_width, latent_height, latent_frames))
  }

  # Device setup
  dit_device <- memory_profile$dit_device
  vae_device <- memory_profile$vae_device

  torch::with_no_grad({

      # ---- Step 1: Text Encoding ----
      if (verbose) message("Encoding text prompt...")

      # Determine dtype based on profile
      latent_dtype <- if (memory_profile$dtype == "float16") {
        torch::torch_float16()
      } else {
        torch::torch_float32()
      }

      # Resolve model path (use hfhub or explicit path)
      resolved_model_path <- NULL
      if (text_backend == "gemma3") {
        if (!is.null(text_model_path)) {
          # Explicit path provided
          expanded_path <- path.expand(text_model_path)
          if (grepl("\\*", expanded_path)) {
            # Glob pattern - find matching directories
            matches <- Sys.glob(expanded_path)
            if (length(matches) > 0) {
              resolved_model_path <- matches[1]
            }
          } else if (dir.exists(expanded_path)) {
            resolved_model_path <- expanded_path
          }
          if (is.null(resolved_model_path)) {
            stop("Gemma3 model not found at: ", text_model_path)
          }
        } else {
          # Use hfhub to find/download model
          if (!requireNamespace("hfhub", quietly = TRUE)) {
            stop("Package 'hfhub' is required. Install with: install.packages('hfhub')")
          }
          gemma_repo <- "google/gemma-3-12b-it"
          resolved_model_path <- tryCatch({
              hfhub::hub_snapshot(gemma_repo, local_files_only = TRUE)
            }, error = function(e) NULL)

          if (is.null(resolved_model_path)) {
            stop("Gemma3 model not found in HuggingFace cache.\n",
              "Download with: huggingface-cli download ", gemma_repo)
          }
        }
      }

      # Encode prompt
      text_result <- encode_text_ltx2(
        prompt = prompt,
        backend = text_backend,
        model_path = resolved_model_path,
        tokenizer_path = resolved_model_path,
        api_url = text_api_url,
        max_sequence_length = 128L,
        caption_channels = 3840L,
        device = "cpu", # Text encoder on CPU (GPU-poor)
        dtype = torch::torch_float32() # CPU always float32
      )
      prompt_embeds <- text_result$prompt_embeds
      prompt_attention_mask <- text_result$prompt_attention_mask

      # Encode negative prompt
      if (is.null(negative_prompt)) {
        negative_prompt <- ""
      }
      neg_result <- encode_text_ltx2(
        prompt = negative_prompt,
        backend = text_backend,
        model_path = resolved_model_path,
        tokenizer_path = resolved_model_path,
        api_url = text_api_url,
        max_sequence_length = 128L,
        caption_channels = 3840L,
        device = "cpu",
        dtype = torch::torch_float32()
      )
      negative_prompt_embeds <- neg_result$prompt_embeds
      negative_attention_mask <- neg_result$prompt_attention_mask

      # ---- Step 1b: Apply Connectors ----
      # Connectors transform packed text embeddings for video/audio cross-attention
      if (is.null(connectors)) {
        # Load connectors from HuggingFace using hfhub
        connector_path <- tryCatch({
            if (!requireNamespace("hfhub", quietly = TRUE)) NULL
            else hfhub::hub_download(
              "Lightricks/LTX-2",
              "connectors/diffusion_pytorch_model.safetensors",
              local_files_only = TRUE
            )
          }, error = function(e) NULL)

        if (!is.null(connector_path) && file.exists(connector_path)) {
          if (verbose) message("Loading text connectors...")
          connectors <- load_ltx2_connectors(
            weights_path = connector_path,
            device = "cpu",
            dtype = "float32",
            verbose = verbose
          )
        } else {
          if (verbose) message("Text connectors not found - using embeddings directly")
        }
      }

      if (!is.null(connectors)) {
        # Run connectors to get video/audio conditioning
        if (verbose) message("Applying text connectors...")
        connector_result <- connectors(prompt_embeds, prompt_attention_mask)
        video_embeds <- connector_result[[1]]
        audio_embeds <- connector_result[[2]]

        neg_connector_result <- connectors(negative_prompt_embeds, negative_attention_mask)
        neg_video_embeds <- neg_connector_result[[1]]
        neg_audio_embeds <- neg_connector_result[[2]]
      } else {
        # Fallback: use packed embeddings directly (may not match DiT dimensions)
        video_embeds <- prompt_embeds
        audio_embeds <- prompt_embeds
        neg_video_embeds <- negative_prompt_embeds
        neg_audio_embeds <- negative_prompt_embeds
      }

      # Move to GPU with correct dtype
      video_embeds <- video_embeds$to(device = dit_device, dtype = latent_dtype)
      audio_embeds <- audio_embeds$to(device = dit_device, dtype = latent_dtype)
      neg_video_embeds <- neg_video_embeds$to(device = dit_device, dtype = latent_dtype)
      neg_audio_embeds <- neg_audio_embeds$to(device = dit_device, dtype = latent_dtype)

      # ---- Step 2: Initialize Latents ----
      if (verbose) message("Initializing latents...")

      # LTX-2 has 128 latent channels
      latent_channels <- 128L
      batch_size <- 1L

      # Random noise
      latents <- torch::torch_randn(
        c(batch_size, latent_channels, latent_frames, latent_height, latent_width),
        device = dit_device,
        dtype = latent_dtype
      )

      # Flatten spatial dims for transformer: [B, C, T, H, W] -> [B, T*H*W, C]
      num_patches <- latent_frames * latent_height * latent_width
      latents <- latents$permute(c(1, 3, 4, 5, 2)) # [B, T, H, W, C]
      latents <- latents$reshape(c(batch_size, num_patches, latent_channels))

      # ---- Step 3: Create Scheduler ----
      if (verbose) message("Setting up FlowMatch scheduler...")

      schedule <- flowmatch_set_timesteps(
        flowmatch_scheduler_create(shift = 9.0),
        num_inference_steps = num_inference_steps,
        device = dit_device
      )

      # ---- Step 4: Load/Create DiT if needed ----
      if (is.null(dit)) {
        # Try to load INT4 quantized weights from R_user_dir cache
        cache_dir <- tools::R_user_dir("diffuseR", "cache")
        int4_path <- file.path(cache_dir, "ltx2_transformer_int4.safetensors")
        if (file.exists(int4_path)) {
          if (verbose) message("Loading INT4 quantized DiT...")

          # Enable INT4-native model creation
          old_use_int4 <- getOption("diffuseR.use_int4", FALSE)
          old_int4_device <- getOption("diffuseR.int4_device", "cuda")
          old_int4_dtype <- getOption("diffuseR.int4_dtype", torch::torch_float16())

          options(diffuseR.use_int4 = TRUE)
          options(diffuseR.int4_device = dit_device)
          options(diffuseR.int4_dtype = latent_dtype)

          # Create model with int4_linear layers via make_linear()
          dit <- ltx2_video_transformer_3d_model(
            in_channels = latent_channels,
            out_channels = latent_channels,
            num_attention_heads = 32L,
            attention_head_dim = 128L,
            cross_attention_dim = 4096L,
            audio_in_channels = 128L,
            audio_out_channels = 128L,
            audio_num_attention_heads = 32L,
            audio_attention_head_dim = 64L,
            audio_cross_attention_dim = 2048L,
            num_layers = 48L,
            caption_channels = 3840L,
            vae_scale_factors = c(temporal_ratio, spatial_ratio, spatial_ratio)
          )

          # Load INT4 weights (keeps them compressed, dequantizes during forward)
          int4_weights <- load_int4_weights(int4_path, verbose = verbose)
          load_int4_weights_into_model(dit, int4_weights, verbose = verbose)

          # Move model to GPU (non-INT4 params like biases, norms)
          dit <- dit$to(device = dit_device, dtype = latent_dtype)

          # Restore options
          options(diffuseR.use_int4 = old_use_int4)
          options(diffuseR.int4_device = old_int4_device)
          options(diffuseR.int4_dtype = old_int4_dtype)
        } else {
          if (verbose) message("NOTE: INT4 weights not found - run quantize_ltx2_transformer() first")
          stop("DiT model required. Run: quantize_ltx2_transformer()")
        }
      }

      # ---- Step 5: Denoising Loop ----
      if (verbose) message(sprintf("Denoising (%d steps)...", num_inference_steps))

      # Audio placeholder (zeros for video-only generation)
      audio_latents <- torch::torch_zeros(
        c(batch_size, 50L, 128L), # Placeholder audio: [B, seq, audio_channels=128]
        device = dit_device,
        dtype = latent_dtype
      )

      timesteps_vec <- schedule$timesteps
      sigmas <- schedule$sigmas

      for (i in seq_len(num_inference_steps)) {
        t_idx <- i
        t <- timesteps_vec[t_idx]
        sigma <- sigmas[t_idx]
        if (i < num_inference_steps) {
          sigma_next <- sigmas[t_idx + 1L]
        } else {
          sigma_next <- 0
        }

        if (verbose && i %% max(1, num_inference_steps %/% 4) == 1) {
          message(sprintf("  Step %d/%d (sigma=%.3f)", i, num_inference_steps, as.numeric(sigma)))
        }

        # Prepare timestep tensor
        timestep <- torch::torch_tensor(c(as.numeric(t))) $unsqueeze(2L)
        timestep <- timestep$to(device = dit_device, dtype = latent_dtype)

        # CFG: conditional and unconditional pass
        if (memory_profile$cfg_mode == "sequential") {
          # Sequential CFG (memory efficient)
          noise_pred <- sequential_cfg_forward(
            model = dit,
            latents = latents,
            timestep = timestep,
            prompt_embeds = video_embeds,
            negative_prompt_embeds = neg_video_embeds,
            guidance_scale = guidance_scale,
            audio_hidden_states = audio_latents,
            audio_encoder_hidden_states = audio_embeds,
            num_frames = latent_frames,
            height = latent_height,
            width = latent_width,
            fps = fps,
            audio_num_frames = 50L
          )
        } else {
          # Batched CFG
          latents_input <- torch::torch_cat(list(latents, latents), dim = 1L)
          video_input <- torch::torch_cat(list(neg_video_embeds, video_embeds), dim = 1L)
          audio_input <- torch::torch_cat(list(neg_audio_embeds, audio_embeds), dim = 1L)
          timestep_input <- torch::torch_cat(list(timestep, timestep), dim = 1L)

          output <- dit(
            hidden_states = latents_input,
            audio_hidden_states = torch::torch_cat(list(audio_latents, audio_latents), dim = 1L),
            encoder_hidden_states = video_input,
            audio_encoder_hidden_states = audio_input,
            timestep = timestep_input,
            num_frames = latent_frames,
            height = latent_height,
            width = latent_width,
            fps = fps,
            audio_num_frames = 50L
          )

          noise_pred_all <- output$sample
          noise_pred_uncond <- noise_pred_all[1,,]$unsqueeze(1L)
          noise_pred_cond <- noise_pred_all[2,,]$unsqueeze(1L)
          # CFG: use tensor method to preserve dtype
          noise_pred <- noise_pred_uncond + (noise_pred_cond - noise_pred_uncond) $mul(guidance_scale)
        }

        # FlowMatch step
        dt <- torch::torch_tensor(sigma_next - sigma, dtype = latent_dtype, device = dit_device)
        latents <- latents + dt * noise_pred

        # Cleanup for low memory
        if (memory_profile$name %in% c("low", "very_low") && i %% 2 == 0) {
          clear_vram()
        }
      }

      # ---- Step 6: Decode Latents ----
      if (verbose) message("Decoding video...")

      # Reshape latents back to spatial: [B, T*H*W, C] -> [B, C, T, H, W]
      latents <- latents$reshape(c(batch_size, latent_frames, latent_height, latent_width, latent_channels))
      latents <- latents$permute(c(1, 5, 2, 3, 4)) # [B, C, T, H, W]

      # Load/create VAE if needed
      if (is.null(vae)) {
        if (verbose) message("NOTE: VAE not provided - skipping decode")
        video_tensor <- latents# Return latents as placeholder
      } else {
        # Configure VAE for memory profile
        configure_vae_for_profile(vae, memory_profile)

        # Move VAE to device
        vae <- vae$to(device = vae_device)

        # Decode
        video_tensor <- vae$decode(latents)
      }

      # Prepare tensor for conversion to R array
      # NOTE: Must use as.array() instead of $numpy() due to R torch bug where

      # tensors returned from with_no_grad() have corrupted method references
      # (error: "could not find function 'fn'"). See cornyverse CLAUDE.md.
      video_cpu <- video_tensor$squeeze(1L) $permute(c(2, 3, 4, 1)) $cpu()

    }) # end with_no_grad

  # Convert to R array (as.array works, $numpy() fails on tensors from with_no_grad)
  video_array <- as.array(video_cpu)

  # Clamp to [0, 1]
  video_array <- pmax(pmin(video_array, 1), 0)

  # ---- Step 7: Save Output ----
  if (!is.null(output_file)) {
    if (verbose) message(sprintf("Saving to %s...", output_file))
    # TODO: Implement save_video in Phase 8
    # save_video(video_array, output_file, fps = fps, format = output_format)
    message("NOTE: Video saving not yet implemented - use save_video()")
  }

  # Build result
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  if (verbose) {
    message(sprintf("Generation complete in %.1f seconds", elapsed))
  }

  result <- list(
    video = video_array,
    metadata = list(
      prompt = prompt,
      negative_prompt = negative_prompt,
      width = width,
      height = height,
      num_frames = num_frames,
      fps = fps,
      num_inference_steps = num_inference_steps,
      guidance_scale = guidance_scale,
      seed = seed,
      memory_profile = memory_profile$name,
      elapsed_seconds = elapsed
    )
  )

  if (return_latents) {
    result$latents <- latents$cpu()
  }

  result
}

