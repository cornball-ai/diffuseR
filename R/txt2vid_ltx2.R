#' Generate Video from Text Prompt using LTX-2
#'
#' Generates video using the LTX-2 diffusion transformer model.
#' Uses the WanGP-style distilled pipeline by default: no classifier-free
#' guidance, specific sigma schedule, and phase-based memory management
#' (components loaded/unloaded sequentially to minimize VRAM usage).
#'
#' @param prompt Character. Text prompt describing the video to generate.
#' @param negative_prompt Character. Optional negative prompt (only used when
#'   distilled=FALSE).
#' @param width Integer. Video width in pixels (default 768).
#' @param height Integer. Video height in pixels (default 512).
#' @param num_frames Integer. Number of frames to generate (default 121).
#' @param fps Numeric. Frames per second (default 24).
#' @param num_inference_steps Integer. Number of denoising steps (default 8
#'   for distilled). Ignored when distilled=TRUE (uses fixed 8-step schedule).
#' @param guidance_scale Numeric. CFG scale (default 1.0, no guidance).
#'   Only used when distilled=FALSE.
#' @param distilled Logical. Use distilled pipeline (default TRUE). Distilled
#'   mode uses a fixed 8-step sigma schedule with no CFG, matching the WanGP
#'   container behavior.
#' @param memory_profile Character or list. Memory profile: "auto" for auto-detection,
#'   or a profile from `ltx2_memory_profile()`.
#' @param model_dir Character. Path to directory containing LTX-2 model files
#'   (VAE, connectors, text projection). When provided, loads from local files
#'   instead of HuggingFace cache.
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
#' # Basic usage (distilled, no CFG)
#' result <- txt2vid_ltx2("A cat walking on a beach at sunset")
#'
#' # With specific settings
#' result <- txt2vid_ltx2(
#'   prompt = "A timelapse of clouds moving over mountains",
#'   width = 512,
#'   height = 320,
#'   num_frames = 61,
#'   seed = 42,
#'   output_file = "clouds.mp4"
#' )
#' }
txt2vid_ltx2 <- function (prompt, negative_prompt = NULL, width = 768L,
                          height = 512L, num_frames = 121L, fps = 24.0,
                          num_inference_steps = 8L, guidance_scale = 1.0,
                          distilled = TRUE,
                          memory_profile = "auto", model_dir = NULL,
                          text_backend = "gemma3", text_model_path = NULL,
                          text_api_url = NULL, vae = NULL, dit = NULL,
                          connectors = NULL, seed = NULL, output_file = NULL,
                          output_format = "mp4", return_latents = FALSE,
                          verbose = TRUE) {
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
        message(sprintf("Video: %dx%d, %d frames @ %.1f fps", width, height,
                        num_frames, fps))
        message(sprintf("Latents: %dx%d, %d frames", latent_width,
                        latent_height, latent_frames))
    }

    # Device setup
    dit_device <- memory_profile$dit_device
    vae_device <- memory_profile$vae_device

    torch::with_no_grad({
            # ---- Step 1: Text Encoding ----
            if (verbose) { message("Encoding text prompt...") }

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
                    # Use hfhub to find model via config.json
                    if (!requireNamespace("hfhub", quietly = TRUE)) {
                        stop("Package 'hfhub' is required. Install with: install.packages('hfhub')")
                    }
                    gemma_repo <- "google/gemma-3-12b-it"
                    config_path <- tryCatch({
                            hfhub::hub_download(gemma_repo, "config.json",
                                                local_files_only = TRUE)
                        }, error = function (e) NULL)

                    if (is.null(config_path)) {
                        stop("Gemma3 model not found in HuggingFace cache.\n",
                             "Download with: huggingface-cli download ",
                             gemma_repo)
                    }
                    resolved_model_path <- dirname(config_path)
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

            # Encode negative prompt (skip in distilled mode - no CFG needed)
            use_cfg <- !distilled && guidance_scale > 1.0
            if (use_cfg) {
                if (is.null(negative_prompt)) negative_prompt <- ""
                neg_result <- encode_text_ltx2(
                    prompt = negative_prompt,
                    backend = text_backend,
                    model_path = resolved_model_path,
                    tokenizer_path = resolved_model_path,
                    api_url = text_api_url,
                    max_sequence_length = 128L,
                    caption_channels = 3840L,
                    device = "cpu",
                    dtype = torch::torch_float32())
                negative_prompt_embeds <- neg_result$prompt_embeds
                negative_attention_mask <- neg_result$prompt_attention_mask
            }

            # ---- Step 1b: Apply Connectors ----
            # Connectors transform packed text embeddings for video/audio cross-attention
            if (is.null(connectors)) {
                connector_path <- NULL
                text_proj_path <- NULL

                if (!is.null(model_dir)) {
                    # Look for Wan2GP split connector files
                    model_dir_exp <- path.expand(model_dir)
                    # Try distilled first, then dev variant
                    for (prefix in c("ltx-2-19b-distilled", "ltx-2-19b-dev")) {
                        cand <- file.path(model_dir_exp,
                                          paste0(prefix,
                                                 "_embeddings_connector.safetensors"))
                        if (file.exists(cand)) { connector_path <- cand;break }
                    }
                    tp <- file.path(model_dir_exp,
                                    "ltx-2-19b_text_embedding_projection.safetensors")
                    if (file.exists(tp)) { text_proj_path <- tp }
                }

                if (is.null(connector_path)) {
                    # Fall back to HuggingFace cache
                    connector_path <- tryCatch({
                            if (!requireNamespace("hfhub", quietly = TRUE)) { NULL } else { hfhub::hub_download("Lightricks/LTX-2", "connectors/diffusion_pytorch_model.safetensors", local_files_only = TRUE) }
                        }, error = function(e) NULL)
                }

                if (!is.null(connector_path) && file.exists(connector_path)) {
                    if (verbose) { message("Loading text connectors...") }
                    connectors <- load_ltx2_connectors(weights_path = connector_path,
                                                       text_proj_path = text_proj_path,
                                                       device = "cpu",
                                                       dtype = "float32",
                                                       verbose = verbose)
                } else {
                    if (verbose) { message("Text connectors not found - using embeddings directly") }
                }
            }

            if (!is.null(connectors)) {
                # Run connectors to get video/audio conditioning
                if (verbose) { message("Applying text connectors...") }
                connector_result <- connectors(prompt_embeds,
                                               prompt_attention_mask)
                video_embeds <- connector_result[[1]]
                audio_embeds <- connector_result[[2]]

                if (use_cfg) {
                    neg_connector_result <- connectors(negative_prompt_embeds,
                                                       negative_attention_mask)
                    neg_video_embeds <- neg_connector_result[[1]]
                    neg_audio_embeds <- neg_connector_result[[2]]
                }
            } else {
                # Connectors not available - generate random projected embeddings
                # for testing, or error for real backends
                if (text_backend == "random") {
                    if (verbose) message("Generating random projected embeddings (no connectors)")
                    emb_shape <- c(prompt_embeds$shape[1],
                                   prompt_embeds$shape[2], 3840L)
                    video_embeds <- torch::torch_randn(emb_shape,
                                                       device = prompt_embeds$device,
                                                       dtype = prompt_embeds$dtype)
                    audio_embeds <- torch::torch_randn(emb_shape,
                                                       device = prompt_embeds$device,
                                                       dtype = prompt_embeds$dtype)
                    if (use_cfg) {
                        neg_video_embeds <- torch::torch_randn(emb_shape,
                                                               device = prompt_embeds$device,
                                                               dtype = prompt_embeds$dtype)
                        neg_audio_embeds <- torch::torch_randn(emb_shape,
                                                               device = prompt_embeds$device,
                                                               dtype = prompt_embeds$dtype)
                    }
                } else {
                    stop("Text connectors required but not found.\n",
                         "Provide model_dir with connector weights, or download with:\n",
                         "  huggingface-cli download Lightricks/LTX-2")
                }
            }

            # Phase cleanup: free text encoder and connectors before DiT
            rm(prompt_embeds, prompt_attention_mask, text_result)
            if (use_cfg) rm(negative_prompt_embeds, negative_attention_mask,
                           neg_result)
            if (!is.null(connectors)) {
                rm(connectors)
            }
            gc()
            if (torch::cuda_is_available()) torch::cuda_empty_cache()
            if (verbose) message("Text encoding complete, freed memory.")

            # Move embeddings to GPU with correct dtype
            video_embeds <- video_embeds$to(device = dit_device,
                                            dtype = latent_dtype)
            audio_embeds <- audio_embeds$to(device = dit_device,
                                            dtype = latent_dtype)
            if (use_cfg) {
                neg_video_embeds <- neg_video_embeds$to(device = dit_device,
                                                        dtype = latent_dtype)
                neg_audio_embeds <- neg_audio_embeds$to(device = dit_device,
                                                        dtype = latent_dtype)
            }

            # ---- Step 2: Initialize Latents ----
            if (verbose) { message("Initializing latents...") }

            # LTX-2 has 128 latent channels
            latent_channels <- 128L
            batch_size <- 1L

            # Random noise
            latents <- torch::torch_randn(c(batch_size, latent_channels,
                                            latent_frames, latent_height,
                                            latent_width),
                                          device = dit_device,
                                          dtype = latent_dtype)

            # Flatten spatial dims for transformer: [B, C, T, H, W] -> [B, T*H*W, C]
            num_patches <- latent_frames * latent_height * latent_width
            latents <- latents$permute(c(1, 3, 4, 5, 2)) # [B, T, H, W, C]
            latents <- latents$reshape(c(batch_size, num_patches,
                                         latent_channels))

            # ---- Step 3: Create Scheduler ----
            if (verbose) { message("Setting up FlowMatch scheduler...") }

            if (distilled) {
                # WanGP distilled sigma schedule (8 steps, no CFG)
                distilled_sigmas <- c(1.0, 0.99375, 0.9875, 0.98125, 0.975,
                                      0.909375, 0.725, 0.421875, 0.0)
                num_inference_steps <- length(distilled_sigmas) - 1L
                sigmas_tensor <- torch::torch_tensor(distilled_sigmas,
                                                     dtype = latent_dtype,
                                                     device = dit_device)
                # Use sigmas directly as timesteps (FlowMatch convention)
                schedule <- list(
                    sigmas = distilled_sigmas,
                    timesteps = distilled_sigmas[-length(distilled_sigmas)]
                )
            } else {
                schedule <- flowmatch_set_timesteps(
                    flowmatch_scheduler_create(shift = 9.0),
                    num_inference_steps = num_inference_steps,
                    device = dit_device)
            }

            # ---- Step 4: Load/Create DiT if needed ----
            if (is.null(dit)) {
                # Try to load INT4 quantized weights from R_user_dir cache
                cache_dir <- tools::R_user_dir("diffuseR", "cache")
                int4_path <- file.path(cache_dir,
                                       "ltx2_transformer_int4.safetensors")
                # Check for single file or sharded files
                int4_shards <- list.files(cache_dir,
                                          pattern = "ltx2_transformer_int4.*\\.safetensors$")
                if (file.exists(int4_path) || length(int4_shards) > 0) {
                    if (verbose) { message("Loading INT4 quantized DiT...") }

                    # Enable INT4-native model creation
                    old_use_int4 <- getOption("diffuseR.use_int4", FALSE)
                    old_int4_device <- getOption("diffuseR.int4_device", "cuda")
                    old_int4_dtype <- getOption("diffuseR.int4_dtype",
                                                torch::torch_float16())

                    options(diffuseR.use_int4 = TRUE)
                    options(diffuseR.int4_device = dit_device)
                    options(diffuseR.int4_dtype = latent_dtype)

                    # Create model with int4_linear layers via make_linear()
                    dit <- ltx2_video_transformer_3d_model(in_channels = latent_channels,
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
                                                           vae_scale_factors = c(temporal_ratio,
                                                                                 spatial_ratio,
                                                                                 spatial_ratio))

                    # Load INT4 weights (keeps them compressed, dequantizes during forward)
                    int4_weights <- load_int4_weights(int4_path,
                                                      verbose = verbose)
                    load_int4_weights_into_model(dit, int4_weights,
                                                 verbose = verbose)

                    # Move model to GPU (non-INT4 params like biases, norms)
                    dit <- dit$to(device = dit_device, dtype = latent_dtype)

                    # Restore options
                    options(diffuseR.use_int4 = old_use_int4)
                    options(diffuseR.int4_device = old_int4_device)
                    options(diffuseR.int4_dtype = old_int4_dtype)
                } else {
                    if (verbose) { message("NOTE: INT4 weights not found - run quantize_ltx2_transformer() first") }
                    stop("DiT model required. Run: quantize_ltx2_transformer()")
                }
            }

            # ---- Step 5: Denoising Loop ----
            if (verbose) { message(sprintf("Denoising (%d steps)...", num_inference_steps)) }

            # Audio placeholder (zeros for video-only generation)
            audio_latents <- torch::torch_zeros(
                c(batch_size, 50L,
                  128L), # Placeholder audio: [B, seq, audio_channels=128]
                device = dit_device,
                dtype = latent_dtype
            )

            timesteps_vec <- schedule$timesteps
            sigmas <- schedule$sigmas

            for (i in seq_len(num_inference_steps)) {
                sigma <- sigmas[i]
                sigma_next <- sigmas[i + 1L]

                if (verbose) {
                    message(sprintf("  Step %d/%d (sigma=%.4f -> %.4f)", i,
                                    num_inference_steps, as.numeric(sigma),
                                    as.numeric(sigma_next)))
                }

                # Prepare timestep tensor (sigma value as timestep)
                timestep <- torch::torch_tensor(c(as.numeric(sigma)))$unsqueeze(2L)
                timestep <- timestep$to(device = dit_device,
                                        dtype = latent_dtype)

                if (distilled || !use_cfg) {
                    # Distilled: single forward pass, no CFG
                    output <- dit(hidden_states = latents,
                                  audio_hidden_states = audio_latents,
                                  encoder_hidden_states = video_embeds,
                                  audio_encoder_hidden_states = audio_embeds,
                                  timestep = timestep,
                                  num_frames = latent_frames,
                                  height = latent_height, width = latent_width,
                                  fps = fps, audio_num_frames = 50L)
                    denoised <- output$sample
                } else if (memory_profile$cfg_mode == "sequential") {
                    # Sequential CFG (memory efficient)
                    denoised <- sequential_cfg_forward(model = dit,
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
                                                       audio_num_frames = 50L)
                } else {
                    # Batched CFG
                    latents_input <- torch::torch_cat(list(latents, latents),
                                                      dim = 1L)
                    video_input <- torch::torch_cat(list(neg_video_embeds,
                                                         video_embeds),
                                                    dim = 1L)
                    audio_input <- torch::torch_cat(list(neg_audio_embeds,
                                                         audio_embeds),
                                                    dim = 1L)
                    timestep_input <- torch::torch_cat(list(timestep, timestep),
                                                       dim = 1L)

                    output <- dit(hidden_states = latents_input,
                                  audio_hidden_states = torch::torch_cat(list(audio_latents,
                                                                              audio_latents),
                                                                         dim = 1L),
                                  encoder_hidden_states = video_input,
                                  audio_encoder_hidden_states = audio_input,
                                  timestep = timestep_input,
                                  num_frames = latent_frames,
                                  height = latent_height, width = latent_width,
                                  fps = fps, audio_num_frames = 50L)

                    denoised_all <- output$sample
                    denoised_uncond <- denoised_all[1,,]$unsqueeze(1L)
                    denoised_cond <- denoised_all[2,,]$unsqueeze(1L)
                    denoised <- denoised_uncond + (denoised_cond - denoised_uncond)$mul(guidance_scale)
                }

                # Debug: show model output and latent statistics
                if (verbose) {
                    d_f <- denoised$to(torch::torch_float32())
                    l_f <- latents$to(torch::torch_float32())
                    message(sprintf("    latents:  mean=%.4f std=%.4f min=%.4f max=%.4f",
                            as.numeric(l_f$mean()), as.numeric(l_f$std()),
                            as.numeric(l_f$min()), as.numeric(l_f$max())))
                    message(sprintf("    model_out: mean=%.4f std=%.4f min=%.4f max=%.4f",
                            as.numeric(d_f$mean()), as.numeric(d_f$std()),
                            as.numeric(d_f$min()), as.numeric(d_f$max())))
                }

                # FlowMatch Euler step (LTX-2 / WanGP convention):
                # Model predicts x_0 (denoised sample).
                # velocity = (z_t - x_0) / sigma; z_{t+1} = z_t + velocity * dt
                sigma_t <- torch::torch_tensor(as.numeric(sigma),
                                               dtype = torch::torch_float32(),
                                               device = dit_device)
                dt <- torch::torch_tensor(sigma_next - sigma,
                                          dtype = torch::torch_float32(),
                                          device = dit_device)
                velocity <- (latents$to(torch::torch_float32()) - denoised$to(torch::torch_float32())) / sigma_t
                latents <- (latents$to(torch::torch_float32()) + velocity * dt)$to(dtype = latent_dtype)

                if (verbose) {
                    l_f <- latents$to(torch::torch_float32())
                    message(sprintf("    after_step: mean=%.4f std=%.4f", as.numeric(l_f$mean()), as.numeric(l_f$std())))
                }

                # Cleanup for low memory
                if (memory_profile$name %in% c("low", "very_low") &&
                    i %% 2 == 0) {
                    clear_vram()
                }
            }

            # ---- Phase cleanup: free DiT before VAE ----
            # Move latents to CPU, delete DiT and embeddings to free VRAM
            latents <- latents$cpu()
            rm(dit, video_embeds, audio_embeds, audio_latents, timestep)
            if (use_cfg) rm(neg_video_embeds, neg_audio_embeds)
            gc()
            if (torch::cuda_is_available()) torch::cuda_empty_cache()
            if (verbose) message("Denoising complete, freed DiT VRAM.")

            # ---- Step 6: Decode Latents ----
            if (verbose) { message("Decoding video...") }

            # Reshape latents back to spatial: [B, T*H*W, C] -> [B, C, T, H, W]
            latents <- latents$reshape(c(batch_size, latent_frames,
                                         latent_height, latent_width,
                                         latent_channels))
            latents <- latents$permute(c(1, 5, 2, 3, 4)) # [B, C, T, H, W]

            # Load/create VAE if needed
            if (is.null(vae)) {
                if (verbose) { message("Loading VAE...") }

                vae_weights_path <- NULL
                if (!is.null(model_dir)) {
                    vae_cand <- file.path(path.expand(model_dir),
                                          "ltx-2-19b_vae.safetensors")
                    if (file.exists(vae_cand)) { vae_weights_path <- vae_cand }
                }

                if (is.null(vae_weights_path)) {
                    # Fall back to HuggingFace cache
                    vae_weights_path <- tryCatch({
                            if (requireNamespace("hfhub", quietly = TRUE)) {
                                config_path <- hfhub::hub_download("Lightricks/LTX-2",
                                                                   "vae/config.json",
                                                                   local_files_only = TRUE)
                                dirname(config_path)
                            } else {
                                NULL
                            }
                        }, error = function(e) NULL)
                }

                if (is.null(vae_weights_path)) {
                    if (verbose) { message("NOTE: VAE not found - skipping decode") }
                    video_tensor <- latents
                } else {
                    # VAE uses float32 for quality (matching WanGP)
                    vae <- load_ltx2_vae(weights_path = vae_weights_path,
                                         device = vae_device,
                                         dtype = "float32", verbose = verbose)
                }
            }

            if (!is.null(vae)) {
                # Configure VAE for memory profile
                configure_vae_for_profile(vae, memory_profile)

                # Move VAE and latents to decode device
                vae <- vae$to(device = vae_device)
                latents <- latents$to(device = vae_device,
                                      dtype = torch::torch_float32())

                # Save latents before decode if requested
                if (return_latents) {
                    saved_latents <- latents$cpu()$clone()
                }

                # Denormalize latents before decoding (diffusers _denormalize_latents)
                # latents = latents * latents_std / scaling_factor + latents_mean
                lat_mean <- vae$latents_mean$view(c(1, -1, 1, 1, 1))$to(
                    device = latents$device, dtype = latents$dtype)
                lat_std <- vae$latents_std$view(c(1, -1, 1, 1, 1))$to(
                    device = latents$device, dtype = latents$dtype)
                latents <- latents * lat_std / vae$scaling_factor + lat_mean

                # Decode
                video_tensor <- vae$decode(latents)

                # Free VAE immediately
                rm(vae, latents)
                gc()
                if (torch::cuda_is_available()) torch::cuda_empty_cache()
            }

            # Prepare tensor for conversion to R array
            video_cpu <- video_tensor$squeeze(1L)$permute(c(2, 3, 4, 1))$cpu()

        }) # end with_no_grad

    # Convert to R array
    video_array <- as.array(video_cpu)

    # Denormalize VAE output: [-1, 1] -> [0, 1] (diffusers VaeImageProcessor)
    video_array <- video_array * 0.5 + 0.5
    video_array <- pmax(pmin(video_array, 1), 0)

    # ---- Step 7: Save Output ----
    if (!is.null(output_file)) {
        if (verbose) { message(sprintf("Saving to %s...", output_file)) }
        save_video_frames(video_array, output_file, fps = fps,
                          verbose = verbose)
    }

    # Build result
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    if (verbose) {
        message(sprintf("Generation complete in %.1f seconds", elapsed))
    }

    result <- list(video = video_array,
                   metadata = list(prompt = prompt,
                                   negative_prompt = negative_prompt,
                                   width = width, height = height,
                                   num_frames = num_frames, fps = fps,
                                   num_inference_steps = num_inference_steps,
                                   guidance_scale = guidance_scale,
                                   distilled = distilled,
                                   seed = seed,
                                   memory_profile = memory_profile$name,
                                   elapsed_seconds = elapsed))

    if (return_latents && exists("saved_latents")) {
        result$latents <- saved_latents
    }

    result
}

#' Save video frames to file
#'
#' Saves an array of video frames to an MP4 file using ffmpeg via the av package.
#'
#' @param video_array Numeric array with dimensions [frames, height, width, channels].
#'   Values should be in range [0, 1].
#' @param output_file Character. Path to output video file.
#' @param fps Numeric. Frames per second. Default: 24.
#' @param verbose Logical. Print progress messages. Default: TRUE.
#' @return Invisibly returns the output file path.
#' @keywords internal
save_video_frames <- function(
    video_array,
    output_file,
    fps = 24,
    verbose = TRUE
) {
    if (!requireNamespace("av", quietly = TRUE)) {
        stop("Package 'av' is required for video saving. Install with: install.packages('av')")
    }

    # video_array should be [frames, height, width, channels]
    dims <- dim(video_array)
    if (length(dims) != 4) {
        stop("video_array must have 4 dimensions: [frames, height, width, channels]")
    }

    num_frames <- dims[1]
    height <- dims[2]
    width <- dims[3]
    channels <- dims[4]

    if (channels != 3) {
        stop("Expected 3 color channels, got ", channels)
    }

    # Create temporary directory for frames

    temp_dir <- tempfile("video_frames_")
    dir.create(temp_dir)
    on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE)

    # Save each frame as PNG
    if (verbose) { message(sprintf("  Writing %d frames...", num_frames)) }

    for (i in seq_len(num_frames)) {
        # Extract frame and convert to [0, 255] uint8
        frame <- video_array[i,,,]
        frame <- pmax(pmin(frame, 1), 0) * 255

        # Convert to integer matrix for png
        # frame is [height, width, channels]
        frame_int <- array(as.integer(round(frame)), dim = dim(frame))

        # Save as PNG
        frame_path <- file.path(temp_dir, sprintf("frame_%05d.png", i))
        png::writePNG(frame_int / 255, frame_path)
    }

    # Encode video using av
    if (verbose) { message("  Encoding video...") }

    # Get list of frame files
    frame_files <- list.files(temp_dir, pattern = "frame_.*\\.png$",
                              full.names = TRUE)
    frame_files <- sort(frame_files)

    # Use av to encode
    av::av_encode_video(input = frame_files, output = output_file,
                        framerate = fps, codec = "libx264", verbose = FALSE)

    if (verbose) { message(sprintf("  Saved: %s", output_file)) }

    invisible(output_file)
}

