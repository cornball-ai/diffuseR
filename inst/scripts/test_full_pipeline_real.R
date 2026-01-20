#!/usr/bin/env r
#
# Full LTX-2 Pipeline Test with Real Text Encoding and VAE
#

library(torch)
library(diffuseR)

cat("=== LTX-2 Full Pipeline (Real Text + VAE) ===\n\n")

get_gpu_memory <- function() {
  out <- system("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", intern = TRUE)
  as.numeric(out[1]) / 1024
}

gc(full = TRUE)
torch::cuda_empty_cache()

# Test parameters
prompt <- "A Bigfoot dj transforming into a humanoid robot with lifelike realistic human facial features and hair with rubbery synthetic skin, uncanny valley style exaggerated features"
width <- 512L
height <- 320L
num_frames <- 17L
num_steps <- 25L  # Proper quality test

cat(sprintf("Prompt: %s\n", prompt))
cat(sprintf("Resolution: %dx%d, %d frames, %d steps\n\n", width, height, num_frames, num_steps))

# LTX-2 compression ratios
spatial_ratio <- 32L
temporal_ratio <- 8L
latent_height <- height %/% spatial_ratio
latent_width <- width %/% spatial_ratio
latent_frames <- (num_frames - 1L) %/% temporal_ratio + 1L

cat(sprintf("Latent dims: %dx%dx%d\n\n", latent_width, latent_height, latent_frames))

torch::with_no_grad({

  # ---- Step 1: Text Encoding with Gemma3 ----
  cat("=== Step 1: Text Encoding (Gemma3) ===\n")
  t0 <- Sys.time()

  # Find Gemma3 model
  gemma_path <- Sys.glob("~/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/*/")[1]
  if (is.na(gemma_path)) {
    stop("Gemma3 model not found. Download with: huggingface-cli download google/gemma-3-12b-it")
  }
  cat(sprintf("Using Gemma3 from: %s\n", gemma_path))

  text_result <- encode_text_ltx2(
    prompt = prompt,
    backend = "gemma3",
    model_path = gemma_path,
    tokenizer_path = gemma_path,
    max_sequence_length = 128L,
    caption_channels = 3840L,
    device = "cpu",
    dtype = torch::torch_float32()
  )

  # Negative prompt (empty)
  neg_result <- encode_text_ltx2(
    prompt = "",
    backend = "gemma3",
    model_path = gemma_path,
    tokenizer_path = gemma_path,
    max_sequence_length = 128L,
    caption_channels = 3840L,
    device = "cpu",
    dtype = torch::torch_float32()
  )

  text_time <- as.numeric(Sys.time() - t0)
  cat(sprintf("Text encoding: %.1f seconds\n", text_time))
  cat(sprintf("Prompt embeds shape: %s\n", paste(dim(text_result$prompt_embeds), collapse = "x")))

  # Save embeddings for future use
  embed_cache <- list(
    prompt_embeds = text_result$prompt_embeds,
    prompt_attention_mask = text_result$prompt_attention_mask,
    neg_prompt_embeds = neg_result$prompt_embeds,
    neg_attention_mask = neg_result$prompt_attention_mask,
    prompt = prompt
  )
  cache_path <- path.expand("~/.cache/diffuseR/text_embeds_test.rds")
  saveRDS(embed_cache, cache_path)
  cat(sprintf("Cached embeddings to: %s\n\n", cache_path))

  # ---- Step 2: Apply Connectors ----
  cat("=== Step 2: Text Connectors ===\n")
  t0 <- Sys.time()

  connector_path <- Sys.glob("~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/*/connectors/diffusion_pytorch_model.safetensors")[1]
  connectors <- load_ltx2_connectors(connector_path, device = "cpu", dtype = "float32", verbose = FALSE)

  conn_result <- connectors(text_result$prompt_embeds, text_result$prompt_attention_mask)
  video_embeds <- conn_result[[1]]
  audio_embeds <- conn_result[[2]]

  neg_conn_result <- connectors(neg_result$prompt_embeds, neg_result$prompt_attention_mask)
  neg_video_embeds <- neg_conn_result[[1]]
  neg_audio_embeds <- neg_conn_result[[2]]

  conn_time <- as.numeric(Sys.time() - t0)
  cat(sprintf("Connectors: %.1f seconds\n", conn_time))
  cat(sprintf("Video embeds shape: %s\n\n", paste(dim(video_embeds), collapse = "x")))

  # Move to GPU
  latent_dtype <- torch::torch_float16()
  video_embeds <- video_embeds$to(device = "cuda", dtype = latent_dtype)
  audio_embeds <- audio_embeds$to(device = "cuda", dtype = latent_dtype)
  neg_video_embeds <- neg_video_embeds$to(device = "cuda", dtype = latent_dtype)
  neg_audio_embeds <- neg_audio_embeds$to(device = "cuda", dtype = latent_dtype)

  # ---- Step 3: Load INT4 DiT ----
  cat("=== Step 3: INT4 DiT Loading ===\n")
  t0 <- Sys.time()
  mem_before <- get_gpu_memory()

  options(diffuseR.use_int4 = TRUE)
  options(diffuseR.int4_device = "cuda")
  options(diffuseR.int4_dtype = latent_dtype)

  dit <- ltx2_video_transformer_3d_model(
    in_channels = 128L,
    out_channels = 128L,
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

  int4_weights <- load_int4_weights("~/.cache/diffuseR/ltx2_transformer_int4.safetensors", verbose = FALSE)
  load_int4_weights_into_model(dit, int4_weights, verbose = FALSE)
  dit <- dit$to(device = "cuda", dtype = latent_dtype)

  dit_time <- as.numeric(Sys.time() - t0)
  mem_after <- get_gpu_memory()
  cat(sprintf("DiT loading: %.1f seconds\n", dit_time))
  cat(sprintf("GPU memory: %.2f GB -> %.2f GB\n\n", mem_before, mem_after))

  # ---- Step 4: Initialize and Denoise ----
  cat("=== Step 4: Denoising ===\n")
  t0 <- Sys.time()

  # Initialize latents
  latents <- torch::torch_randn(
    c(1L, 128L, latent_frames, latent_height, latent_width),
    device = "cuda",
    dtype = latent_dtype
  )

  # Pack for transformer
  num_patches <- latent_frames * latent_height * latent_width
  latents <- latents$permute(c(1L, 3L, 4L, 5L, 2L))
  latents <- latents$reshape(c(1L, num_patches, 128L))

  # Audio placeholder
  audio_latents <- torch::torch_zeros(c(1L, 50L, 128L), device = "cuda", dtype = latent_dtype)

  # Scheduler
  schedule <- flowmatch_set_timesteps(
    flowmatch_scheduler_create(shift = 9.0),
    num_inference_steps = num_steps,
    device = "cuda"
  )

  guidance_scale <- 4.0

  for (i in seq_len(num_steps)) {
    sigma <- schedule$sigmas[i]
    sigma_next <- if (i < num_steps) schedule$sigmas[i + 1L] else 0

    timestep <- torch::torch_tensor(c(as.numeric(schedule$timesteps[i])))$unsqueeze(2L)
    timestep <- timestep$to(device = "cuda", dtype = latent_dtype)

    # Unconditional forward
    out_uncond <- dit(
      hidden_states = latents,
      audio_hidden_states = audio_latents,
      encoder_hidden_states = neg_video_embeds,
      audio_encoder_hidden_states = neg_audio_embeds,
      timestep = timestep,
      num_frames = latent_frames,
      height = latent_height,
      width = latent_width,
      fps = 24.0,
      audio_num_frames = 50L
    )

    # Conditional forward
    out_cond <- dit(
      hidden_states = latents,
      audio_hidden_states = audio_latents,
      encoder_hidden_states = video_embeds,
      audio_encoder_hidden_states = audio_embeds,
      timestep = timestep,
      num_frames = latent_frames,
      height = latent_height,
      width = latent_width,
      fps = 24.0,
      audio_num_frames = 50L
    )

    # CFG
    noise_uncond <- out_uncond$sample
    noise_cond <- out_cond$sample
    noise_pred <- noise_uncond + (noise_cond - noise_uncond)$mul(guidance_scale)

    # FlowMatch step
    dt <- torch::torch_tensor(sigma_next - as.numeric(sigma), dtype = latent_dtype, device = "cuda")
    latents <- latents + dt * noise_pred

    cat(sprintf("  Step %d/%d done\n", i, num_steps))
  }

  denoise_time <- as.numeric(Sys.time() - t0)
  mem_peak <- get_gpu_memory()
  cat(sprintf("Denoising: %.1f seconds (%.1f s/step)\n", denoise_time, denoise_time / num_steps))
  cat(sprintf("Peak GPU memory: %.2f GB\n\n", mem_peak))

  # Unpack latents
  latents <- latents$reshape(c(1L, latent_frames, latent_height, latent_width, 128L))
  latents <- latents$permute(c(1L, 5L, 2L, 3L, 4L))

  # ---- Step 5: VAE Decode on CPU ----
  cat("=== Step 5: VAE Decode (CPU) ===\n")
  t0 <- Sys.time()

  # Clear GPU for VAE
  rm(dit, int4_weights)
  gc(full = TRUE)
  torch::cuda_empty_cache()

  # Move latents to CPU
  latents_cpu <- latents$to(device = "cpu", dtype = torch::torch_float32())
  rm(latents)
  gc()

  cat(sprintf("Latents moved to CPU, shape: %s\n", paste(dim(latents_cpu), collapse = "x")))

  # Load VAE
  vae_path <- Sys.glob("~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/*/vae/")[1]
  if (!is.na(vae_path) && dir.exists(vae_path)) {
    cat("Loading VAE...\n")
    vae <- load_ltx2_vae(vae_path, device = "cpu", dtype = "float32", verbose = TRUE)

    cat("Decoding latents...\n")
    video_tensor <- vae$decode(latents_cpu)

    vae_time <- as.numeric(Sys.time() - t0)
    cat(sprintf("VAE decode: %.1f seconds\n", vae_time))
    cat(sprintf("Video tensor shape: %s\n", paste(dim(video_tensor), collapse = "x")))

    # Convert to array
    video_array <- as.array(video_tensor$squeeze(1L)$permute(c(2L, 3L, 4L, 1L))$cpu())
    video_array <- pmax(pmin(video_array, 1), 0)
    cat(sprintf("Video array shape: %s\n", paste(dim(video_array), collapse = "x")))
    cat(sprintf("Value range: [%.3f, %.3f]\n", min(video_array), max(video_array)))

    # Save video
    output_path <- path.expand("~/cornball_media/bigfoot_dj_25steps.mp4")
    dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
    cat(sprintf("\nSaving video to: %s\n", output_path))
    save_video(video_array, output_path, fps = 24)
    cat("Video saved.\n")
  } else {
    cat("VAE not found - skipping decode\n")
    vae_time <- 0
  }

})

# ---- Summary ----
cat("\n=== TIMING SUMMARY ===\n")
cat(sprintf("Text encoding (Gemma3):  %.1f s\n", text_time))
cat(sprintf("Connectors:              %.1f s\n", conn_time))
cat(sprintf("DiT loading:             %.1f s\n", dit_time))
cat(sprintf("Denoising (%d steps):    %.1f s\n", num_steps, denoise_time))
cat(sprintf("VAE decode (CPU):        %.1f s\n", vae_time))
cat(sprintf("TOTAL:                   %.1f s\n", text_time + conn_time + dit_time + denoise_time + vae_time))
cat(sprintf("\nPeak GPU memory: %.2f GB\n", mem_peak))
