#' LTX-2.3 Text-to-Video Pipeline
#'
#' Fresh R port of the LTX-2 text-to-video flow from the diffusers
#' reference (Apache-2.0, pipelines/ltx2/pipeline_ltx2.py), specialized
#' for the distilled LTX 2.3 checkpoints: 8-step official sigma schedule,
#' no classifier-free guidance, joint audio-video denoising with an Euler
#' velocity step, and audio decoding through the audio VAE and BWE
#' vocoder to 48 kHz stereo.
#'
#' @name txt2vid_ltx23
NULL

#' Official distilled sigma schedule
#'
#' The distilled LTX sigma values (with terminal zero appended), as
#' published in the Apache-2.0 diffusers reference
#' (pipelines/ltx2/utils.py).
#'
#' @return Numeric vector of length 9.
#'
#' @export
ltx23_distilled_sigmas <- function() {
  c(1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
}

#' Stage-2 distilled sigma schedule (two-stage refinement)
#'
#' @return Numeric vector of length 4.
#'
#' @export
ltx23_stage2_distilled_sigmas <- function() {
  c(0.909375, 0.725, 0.421875, 0.0)
}

# Pack video latents [B, C, F, H, W] -> [B, F*H*W, C] (patch sizes are 1
# for the LTX transformer; kept general per the reference)
ltx23_pack_video_latents <- function(latents, patch_size = 1L, patch_size_t = 1L) {
  d <- latents$shape
  pf <- d[3] %/% patch_size_t
  ph <- d[4] %/% patch_size
  pw <- d[5] %/% patch_size
  latents <- latents$reshape(c(d[1], -1L, pf, patch_size_t, ph, patch_size, pw, patch_size))
  # [B, C, F', pt, H', p, W', p] -> [B, F', H', W', C, pt, p, p]
  latents <- latents$permute(c(1L, 3L, 5L, 7L, 2L, 4L, 6L, 8L))
  latents$flatten(start_dim = 5L, end_dim = 8L)$flatten(start_dim = 2L, end_dim = 4L)
}

# Unpack video latents [B, S, D] -> [B, C, F, H, W]
ltx23_unpack_video_latents <- function(latents, num_frames, height, width,
  patch_size = 1L, patch_size_t = 1L) {
  b <- latents$shape[1]
  latents <- latents$reshape(c(b, num_frames, height, width, -1L,
    patch_size_t, patch_size, patch_size))
  latents <- latents$permute(c(1L, 5L, 2L, 6L, 3L, 7L, 4L, 8L))
  latents$flatten(start_dim = 7L, end_dim = 8L)$
    flatten(start_dim = 5L, end_dim = 6L)$
    flatten(start_dim = 3L, end_dim = 4L)
}

# Pack audio latents [B, C, L, M] -> [B, L, C * M]
ltx23_pack_audio_latents <- function(latents) {
  latents$transpose(2L, 3L)$flatten(start_dim = 3L, end_dim = 4L)
}

# Unpack audio latents [B, L, C * M] -> [B, C, L, M]
ltx23_unpack_audio_latents <- function(latents, num_mel_bins) {
  latents$unflatten(3L, c(-1L, num_mel_bins))$transpose(2L, 3L)
}

# Audio statistics apply to the PACKED [B, L, C*M] representation
# (broadcast over the trailing feature dim)
.ltx23_denormalize_audio <- function(latents, latents_mean, latents_std) {
  mean <- latents_mean$to(device = latents$device, dtype = latents$dtype)
  std <- latents_std$to(device = latents$device, dtype = latents$dtype)
  latents * std + mean
}

# Joint audio/video Euler denoise loop over a sigma schedule (CFG-free)
.ltx23_denoise <- function(
  transformer, latents, audio_latents, sigmas,
  video_text_embeds, audio_text_embeds, text_mask,
  latent_frames, latent_height, latent_width,
  audio_num_frames, frame_rate,
  device, compute_dtype, verbose = TRUE, stage = ""
) {
  f32 <- torch::torch_float32()
  video_coords <- transformer$rope$prepare_video_coords(
    latents$shape[1], latent_frames, latent_height, latent_width,
    device, fps = frame_rate
  )
  audio_coords <- transformer$audio_rope$prepare_audio_coords(
    audio_latents$shape[1], audio_num_frames, device
  )

  n_steps <- length(sigmas) - 1L
  scale_mult <- transformer$timestep_scale_multiplier
  torch::with_no_grad({
    for (i in seq_len(n_steps)) {
      sigma <- sigmas[i]
      t <- torch::torch_tensor(sigma * scale_mult, device = device, dtype = f32)

      out <- transformer(
        hidden_states = latents$to(dtype = compute_dtype),
        audio_hidden_states = audio_latents$to(dtype = compute_dtype),
        encoder_hidden_states = video_text_embeds,
        audio_encoder_hidden_states = audio_text_embeds,
        timestep = t,
        sigma = t,
        encoder_attention_mask = text_mask,
        audio_encoder_attention_mask = text_mask,
        num_frames = latent_frames,
        height = latent_height,
        width = latent_width,
        fps = frame_rate,
        audio_num_frames = audio_num_frames,
        video_coords = video_coords,
        audio_coords = audio_coords,
        use_cross_timestep = TRUE
      )

      # Euler velocity step in float32; dt is negative (sigma decreasing)
      dt <- torch::torch_tensor(sigmas[i + 1L] - sigma, device = device, dtype = f32)
      latents <- latents + dt * out$sample$to(dtype = f32)
      audio_latents <- audio_latents + dt * out$audio_sample$to(dtype = f32)
      rm(out)
      gc(verbose = FALSE)
      if (verbose) {
        message(sprintf("  %sstep %d/%d (sigma %.4f)", stage, i, n_steps, sigma))
      }
    }
  })
  list(latents = latents, audio_latents = audio_latents)
}

#' Load the LTX-2.3 generation components from a single-file checkpoint
#'
#' Builds the transformer, connectors, video VAE, audio VAE, and vocoder
#' with the LTX 2.3 configuration and streams the checkpoint weights into
#' them. The Gemma3 text encoder ships separately (see
#' \code{\link{load_gemma3_text_encoder}}).
#'
#' @param checkpoint_path Path to the single-file checkpoint (e.g.
#'   \code{ltx-2.3-22b-distilled-1.1.safetensors}) or to an fp8 artifact
#'   directory produced by \code{\link{ltx23_quantize_fp8}}. With the fp8
#'   artifact, the transformer loads with CPU-resident fp8 weights that
#'   stream to \code{device} during the forward pass.
#' @param device Character. Device for the small components (VAEs,
#'   vocoder, connectors) and, with fp8, the transformer residents.
#' @param dtype Character. "bfloat16" (checkpoint native) or "float32".
#' @param transformer_device Character. Device for the transformer
#'   weights when loading the plain (non-fp8) checkpoint.
#' @param components Character vector. Which components to load.
#' @param pin Logical. Pin fp8 host memory (fp8 artifact only).
#' @param attn_chunk Integer or NULL. Query-chunk size for attention
#'   (see \code{\link{ltx23_set_attn_chunk}}).
#' @param verbose Logical.
#'
#' @return A list with the loaded modules and the checkpoint config,
#'   class \code{ltx23_pipeline}.
#'
#' @export
ltx23_load_pipeline <- function(
  checkpoint_path,
  device = "cuda",
  dtype = "bfloat16",
  transformer_device = "cpu",
  components = c("dit", "connectors", "vae", "audio_vae", "vocoder"),
  pin = TRUE,
  attn_chunk = NULL,
  verbose = TRUE
) {
  fp8 <- dir.exists(checkpoint_path)
  ckpt <- if (fp8) {
    ltx23_open_fp8_checkpoint(checkpoint_path)
  } else {
    ltx23_open_checkpoint(checkpoint_path)
  }
  groups <- ltx23_split_keys(ckpt$keys)
  torch_dtype <- switch(dtype,
    bfloat16 = torch::torch_bfloat16(),
    float16 = torch::torch_float16(),
    float32 = torch::torch_float32(),
    stop("Unsupported dtype: ", dtype)
  )

  pipe <- list(config = ckpt$config, checkpoint_path = ckpt$path)

  load_component <- function(name, module, map_key, dev) {
    if (verbose) message("Loading ", name, " (", length(groups[[name]]), " tensors)...")
    module$to(dtype = torch_dtype)
    res <- ltx23_load_group(ckpt, groups[[name]], module,
      map_key = map_key, verbose = verbose)
    if (length(res$unmapped) || length(res$unfilled)) {
      stop(name, ": incomplete load (", length(res$unmapped), " unmapped, ",
        length(res$unfilled), " unfilled)")
    }
    module$to(device = dev)
    module$eval()
    module
  }

  if ("dit" %in% components) {
    if (fp8) {
      pipe$transformer <- ltx23_load_transformer_fp8(
        ckpt, device = device, pin = pin, verbose = verbose
      )
    } else {
      pipe$transformer <- load_component(
        "dit", ltx23_transformer(), ltx23_map_dit_key, transformer_device
      )
    }
    if (!is.null(attn_chunk)) {
      ltx23_set_attn_chunk(pipe$transformer, as.integer(attn_chunk))
    }
  }
  if ("connectors" %in% components) {
    pipe$connectors <- load_component(
      "connectors", ltx23_text_connectors(), ltx23_map_connector_key, device
    )
  }
  if ("vae" %in% components) {
    pipe$vae <- load_component(
      "vae", ltx23_video_vae(), ltx23_map_vae_key, device
    )
  }
  if ("audio_vae" %in% components) {
    pipe$audio_vae <- load_component(
      "audio_vae", ltx23_audio_vae(), ltx23_map_audio_vae_key, device
    )
  }
  if ("vocoder" %in% components) {
    # The vocoder is small and precision-sensitive: run in float32
    voc <- ltx23_vocoder_with_bwe()
    voc$to(dtype = torch_dtype)
    res <- ltx23_load_group(ckpt, groups$vocoder, voc,
      map_key = ltx23_map_vocoder_key, verbose = verbose)
    if (length(res$unmapped) || length(res$unfilled)) {
      stop("vocoder: incomplete load")
    }
    voc$to(device = device, dtype = torch::torch_float32())
    voc$eval()
    pipe$vocoder <- voc
  }

  structure(pipe, class = "ltx23_pipeline")
}

#' Generate video (and audio) with LTX-2.3
#'
#' Distilled text-to-video generation: encodes the prompt with Gemma3 +
#' connectors, denoises joint audio/video latents over the official
#' 8-step distilled schedule (no classifier-free guidance), decodes the
#' video with the causal VAE and the audio with the audio VAE + BWE
#' vocoder, and optionally muxes both into an MP4.
#'
#' @param prompt Character. The prompt.
#' @param pipeline An \code{ltx23_pipeline} from
#'   \code{\link{ltx23_load_pipeline}}.
#' @param text_encoder,tokenizer Gemma3 model and tokenizer (or paths;
#'   see \code{\link{load_gemma3_text_encoder}} and
#'   \code{\link{gemma3_tokenizer}}). Ignored when \code{prompt_embeds}
#'   is supplied.
#' @param prompt_embeds Optional precomputed list with
#'   \code{prompt_embeds} (raw stacked Gemma3 states) and
#'   \code{prompt_attention_mask}; bypasses the text encoder.
#' @param width,height Integers. Output resolution (multiples of 32).
#' @param num_frames Integer. 8k + 1 frames (e.g. 121).
#' @param frame_rate Numeric. Frames per second.
#' @param sigmas Numeric vector. Denoising schedule (default: official
#'   distilled schedule; must end in 0).
#' @param guidance_scale Numeric. Only 1 (no CFG) is supported; the
#'   distilled checkpoints are trained for CFG-free sampling.
#' @param seed Integer or NULL.
#' @param device Character. Compute device for the denoising loop.
#' @param dtype Character. Model compute dtype ("bfloat16" or "float32").
#' @param filename Character or NULL. Output video path (.mp4). Audio is
#'   muxed in when the av package is available.
#' @param max_sequence_length Integer. Text token length (multiple of 128).
#' @param decode_video,decode_audio Logicals. Decode the respective
#'   latents (disable for latent-space work).
#' @param two_stage Logical. Generate at half resolution, upsample the
#'   latents 2x spatially, and refine over the stage-2 schedule
#'   (resolution must then be a multiple of 64; requires
#'   \code{upsampler}).
#' @param upsampler An \code{\link{ltx23_latent_upsampler}} (see
#'   \code{\link{ltx23_load_upsampler}}).
#' @param adain_factor Numeric. AdaIN blend of the upsampled latents
#'   toward the stage-1 statistics (0 disables).
#' @param tone_map_compression Numeric in [0, 1]. Optional latent tone
#'   mapping before stage 2.
#' @param verbose Logical.
#'
#' @return Invisibly, a list with \code{video} (array
#'   [frames, height, width, 3] in [0, 1]), \code{audio} (matrix
#'   [2, samples] in [-1, 1]), \code{sample_rate}, and the raw latents.
#'
#' @export
txt2vid_ltx2 <- function(
  prompt,
  pipeline,
  text_encoder = NULL,
  tokenizer = NULL,
  prompt_embeds = NULL,
  width = 768L,
  height = 512L,
  num_frames = 121L,
  frame_rate = 24,
  sigmas = ltx23_distilled_sigmas(),
  guidance_scale = 1,
  seed = NULL,
  device = "cuda",
  dtype = "bfloat16",
  filename = NULL,
  max_sequence_length = 1024L,
  decode_video = TRUE,
  decode_audio = TRUE,
  two_stage = FALSE,
  upsampler = NULL,
  adain_factor = 1.0,
  tone_map_compression = 0,
  verbose = TRUE
) {
  stopifnot(inherits(pipeline, "ltx23_pipeline"))
  if (guidance_scale != 1) {
    stop("Only guidance_scale = 1 is supported (distilled checkpoints).")
  }
  if (two_stage) {
    if (is.null(upsampler)) {
      stop("two_stage = TRUE requires an upsampler (see ltx23_load_upsampler).")
    }
    if (width %% 64L != 0L || height %% 64L != 0L) {
      stop("width and height must be multiples of 64 for two-stage generation")
    }
  }
  if (width %% 32L != 0L || height %% 32L != 0L) {
    stop("width and height must be multiples of 32")
  }
  if ((num_frames - 1L) %% 8L != 0L) {
    stop("num_frames must be 8k + 1 (e.g. 121)")
  }
  if (utils::tail(sigmas, 1) != 0) {
    stop("sigmas must end with 0")
  }
  if (!is.null(seed)) torch::torch_manual_seed(seed)

  compute_dtype <- switch(dtype,
    bfloat16 = torch::torch_bfloat16(),
    float16 = torch::torch_float16(),
    float32 = torch::torch_float32(),
    stop("Unsupported dtype: ", dtype)
  )
  f32 <- torch::torch_float32()

  # --- Phase 1: text encoding -------------------------------------------------
  if (is.null(prompt_embeds)) {
    if (is.null(text_encoder) || is.null(tokenizer)) {
      stop("Supply text_encoder + tokenizer, or precomputed prompt_embeds.")
    }
    if (verbose) message("Encoding prompt...")
    prompt_embeds <- encode_with_gemma3(
      prompt,
      model = text_encoder, tokenizer = tokenizer,
      max_sequence_length = max_sequence_length,
      device = if (is.character(text_encoder)) device else "cpu",
      verbose = verbose
    )
  }

  connectors_device <- pipeline$connectors$video_text_proj_in$weight$device
  torch::with_no_grad({
    conn <- pipeline$connectors(
      prompt_embeds$prompt_embeds$to(device = connectors_device, dtype = compute_dtype),
      prompt_embeds$prompt_attention_mask$to(device = connectors_device)
    )
  })
  video_text_embeds <- conn$video_text_embedding
  audio_text_embeds <- conn$audio_text_embedding
  text_mask <- conn$attention_mask
  rm(conn)
  gc(verbose = FALSE)

  # --- Phase 2: latent preparation ---------------------------------------------
  latent_frames <- (num_frames - 1L) %/% 8L + 1L
  latent_height <- height %/% 32L
  latent_width <- width %/% 32L
  # Two-stage: stage 1 generates at half resolution
  s1_height <- if (two_stage) latent_height %/% 2L else latent_height
  s1_width <- if (two_stage) latent_width %/% 2L else latent_width

  # 25 audio latent frames per second: sampling_rate / hop / downsample
  audio_num_frames <- as.integer(round(num_frames / frame_rate * 25))
  latent_mel_bins <- 16L# 64 mel bins / 4

  latents <- torch::torch_randn(
    c(1L, 128L, latent_frames, s1_height, s1_width),
    device = device, dtype = f32
  )
  latents <- ltx23_pack_video_latents(latents)

  audio_latents <- torch::torch_randn(
    c(1L, 8L, audio_num_frames, latent_mel_bins),
    device = device, dtype = f32
  )
  audio_latents <- ltx23_pack_audio_latents(audio_latents)

  # --- Phase 3: denoising -------------------------------------------------------
  transformer <- pipeline$transformer
  if (verbose) message(sprintf("Denoising: %d steps at %dx%dx%d...",
    length(sigmas) - 1L, width %/% (if (two_stage) 2L else 1L),
    height %/% (if (two_stage) 2L else 1L), num_frames))

  denoised <- .ltx23_denoise(
    transformer, latents, audio_latents, sigmas,
    video_text_embeds, audio_text_embeds, text_mask,
    latent_frames, s1_height, s1_width,
    audio_num_frames, frame_rate,
    device, compute_dtype, verbose = verbose,
    stage = if (two_stage) "stage 1 " else ""
  )
  latents <- denoised$latents
  audio_latents <- denoised$audio_latents

  if (two_stage) {
    vae <- pipeline$vae
    if (verbose) message("Upsampling latents 2x...")
    torch::with_no_grad({
      up_device <- upsampler$final_conv$weight$device
      up_dtype <- upsampler$final_conv$weight$dtype
      s1_latents <- ltx23_unpack_video_latents(
        latents, latent_frames, s1_height, s1_width
      )$to(device = up_device)
      # The upsampler operates on unnormalized latents
      s1_latents <- ltx23_denormalize_latents(
        s1_latents, vae$latents_mean, vae$latents_std
      )
      up_latents <- upsampler(s1_latents$to(dtype = up_dtype))$to(dtype = f32)
      if (adain_factor != 0) {
        up_latents <- ltx23_adain_filter_latent(
          up_latents, s1_latents$to(dtype = f32), adain_factor
        )
      }
      if (tone_map_compression > 0) {
        up_latents <- ltx23_tone_map_latents(up_latents, tone_map_compression)
      }
      up_latents <- ltx23_normalize_latents(
        up_latents, vae$latents_mean, vae$latents_std
      )
      latents <- ltx23_pack_video_latents(up_latents$to(device = device))
      rm(s1_latents, up_latents)
    })
    gc(verbose = FALSE)

    # Re-noise BOTH modalities at the stage-2 entry sigma
    s2_sigmas <- ltx23_stage2_distilled_sigmas()
    noise_scale <- s2_sigmas[1]
    torch::with_no_grad({
      latents <- torch::torch_randn_like(latents)$mul(noise_scale) +
        latents$mul(1 - noise_scale)
      audio_latents <- torch::torch_randn_like(audio_latents)$mul(noise_scale) +
        audio_latents$mul(1 - noise_scale)
    })

    if (verbose) message(sprintf("Refining: %d steps at %dx%d...",
      length(s2_sigmas) - 1L, width, height))
    denoised <- .ltx23_denoise(
      transformer, latents, audio_latents, s2_sigmas,
      video_text_embeds, audio_text_embeds, text_mask,
      latent_frames, latent_height, latent_width,
      audio_num_frames, frame_rate,
      device, compute_dtype, verbose = verbose, stage = "stage 2 "
    )
    latents <- denoised$latents
    audio_latents <- denoised$audio_latents
  }

  result <- list(
    latents = latents,
    audio_latents = audio_latents,
    sample_rate = 48000L
  )

  # --- Phase 4: decoding -----------------------------------------------------------
  if (decode_video) {
    if (verbose) message("Decoding video...")
    vae <- pipeline$vae
    vae_device <- vae$latents_mean$device
    vae_dtype <- vae$decoder$conv_in$conv$weight$dtype
    torch::with_no_grad({
      video_latents <- ltx23_unpack_video_latents(
        latents, latent_frames, latent_height, latent_width
      )$to(device = vae_device)
      video_latents <- ltx23_denormalize_latents(
        video_latents, vae$latents_mean, vae$latents_std
      )
      video <- vae$decode(video_latents$to(dtype = vae_dtype))
      # [-1, 1] -> [0, 1], [B, 3, F, H, W] -> [F, H, W, 3]
      video <- ((video$to(dtype = f32) / 2 + 0.5)$clamp(0, 1))[1, , , , ]
      video <- video$permute(c(2L, 3L, 4L, 1L))$cpu()
    })
    result$video <- as.array(video)
    rm(video, video_latents)
    gc(verbose = FALSE)
  }

  if (decode_audio) {
    if (verbose) message("Decoding audio...")
    audio_vae <- pipeline$audio_vae
    av_device <- audio_vae$latents_mean$device
    av_dtype <- audio_vae$decoder$conv_in$conv$weight$dtype
    torch::with_no_grad({
      audio_packed <- .ltx23_denormalize_audio(
        audio_latents$to(device = av_device),
        audio_vae$latents_mean, audio_vae$latents_std
      )
      audio_lat <- ltx23_unpack_audio_latents(audio_packed, latent_mel_bins)
      mel <- audio_vae$decode(audio_lat$to(dtype = av_dtype))
      waveform <- pipeline$vocoder(mel$to(dtype = torch::torch_float32()))
      waveform <- waveform[1, , ]$cpu()
    })
    result$audio <- as.matrix(as.array(waveform))
    rm(waveform, mel, audio_lat, audio_packed)
    gc(verbose = FALSE)
  }

  if (!is.null(filename) && decode_video) {
    save_video_ltx23(result$video, filename,
      fps = frame_rate,
      audio = if (decode_audio) result$audio else NULL,
      sample_rate = result$sample_rate,
      verbose = verbose
    )
    result$filename <- filename
  }

  invisible(result)
}

#' Write a 16-bit PCM WAV file
#'
#' Minimal RIFF writer in base R.
#'
#' @param audio Numeric matrix [channels, samples] in [-1, 1].
#' @param path Output path.
#' @param sample_rate Integer.
#'
#' @return Invisibly, the path.
#'
#' @export
write_wav <- function(audio, path, sample_rate = 48000L) {
  if (is.null(dim(audio))) audio <- matrix(audio, nrow = 1L)
  n_channels <- nrow(audio)
  n_samples <- ncol(audio)

  # Interleave channels, scale to int16
  pcm <- as.integer(round(pmax(pmin(as.vector(audio), 1), -1) * 32767))
  byte_rate <- sample_rate * n_channels * 2L
  data_size <- n_samples * n_channels * 2L

  con <- file(path, "wb")
  on.exit(close(con), add = TRUE)
  writeChar("RIFF", con, eos = NULL)
  writeBin(as.integer(36L + data_size), con, size = 4, endian = "little")
  writeChar("WAVEfmt ", con, eos = NULL)
  writeBin(16L, con, size = 4, endian = "little")
  writeBin(1L, con, size = 2, endian = "little")# PCM
  writeBin(n_channels, con, size = 2, endian = "little")
  writeBin(as.integer(sample_rate), con, size = 4, endian = "little")
  writeBin(byte_rate, con, size = 4, endian = "little")
  writeBin(n_channels * 2L, con, size = 2, endian = "little")# block align
  writeBin(16L, con, size = 2, endian = "little")# bits per sample
  writeChar("data", con, eos = NULL)
  writeBin(data_size, con, size = 4, endian = "little")
  writeBin(pcm, con, size = 2, endian = "little")
  invisible(path)
}

#' Save an LTX video (optionally with audio) to MP4
#'
#' Uses the av package (Suggests) to encode frames and mux the audio
#' track.
#'
#' @param video Array [frames, height, width, 3] in [0, 1].
#' @param filename Output path (.mp4).
#' @param fps Numeric.
#' @param audio Optional numeric matrix [channels, samples] in [-1, 1].
#' @param sample_rate Integer.
#' @param verbose Logical.
#'
#' @return Invisibly, the filename.
#'
#' @export
save_video_ltx23 <- function(video, filename, fps = 24, audio = NULL,
  sample_rate = 48000L, verbose = TRUE) {
  if (!requireNamespace("av", quietly = TRUE)) {
    stop("The av package is required to write video files.")
  }
  tmp_dir <- tempfile("ltx23_frames_")
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  n_frames <- dim(video)[1]
  frame_paths <- character(n_frames)
  for (i in seq_len(n_frames)) {
    frame_paths[i] <- file.path(tmp_dir, sprintf("frame_%05d.png", i))
    png::writePNG(video[i, , , ], frame_paths[i])
  }

  audio_path <- NULL
  if (!is.null(audio)) {
    audio_path <- file.path(tmp_dir, "audio.wav")
    write_wav(audio, audio_path, sample_rate = sample_rate)
  }

  av::av_encode_video(frame_paths, output = filename, framerate = fps,
    audio = audio_path, verbose = verbose)
  if (verbose) message("Wrote ", filename)
  invisible(filename)
}
