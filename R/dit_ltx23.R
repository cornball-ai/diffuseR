#' LTX-2.3 Audio-Video Diffusion Transformer
#'
#' Fresh R port of the LTX-2 transformer from the diffusers reference
#' (Apache-2.0, src/diffusers/models/transformers/transformer_ltx2.py),
#' configured for LTX 2.3: gated attention, cross-attention modulation,
#' prompt AdaLN, split RoPE, and connector-projected text embeddings
#' (no in-model caption projection).
#'
#' @name dit_ltx23
NULL

#' LTX-2.3 video transformer model
#'
#' Dual-stream audio/video DiT. Text embeddings arrive already projected
#' to the video (\code{inner_dim}) and audio (\code{audio_inner_dim})
#' dimensions by the connector modules.
#'
#' @param in_channels,out_channels Integers. Video latent channels.
#' @param patch_size,patch_size_t Integers. Video patch sizes.
#' @param num_attention_heads,attention_head_dim Video attention shape.
#' @param cross_attention_dim Integer. Video text embedding dimension.
#' @param vae_scale_factors Integer vector. VAE (time, height, width) scales.
#' @param pos_embed_max_pos,base_height,base_width RoPE base grid.
#' @param audio_in_channels,audio_out_channels Integers. Audio latent channels.
#' @param audio_patch_size,audio_patch_size_t Integers. Audio patch sizes.
#' @param audio_num_attention_heads,audio_attention_head_dim Audio attention shape.
#' @param audio_cross_attention_dim Integer. Audio text embedding dimension.
#' @param audio_scale_factor,audio_pos_embed_max_pos,audio_sampling_rate,audio_hop_length
#'   Audio latent grid parameters.
#' @param num_layers Integer. Transformer block count.
#' @param norm_eps Numeric. Norm epsilon.
#' @param rope_theta,rope_double_precision,causal_offset RoPE parameters.
#' @param timestep_scale_multiplier,cross_attn_timestep_scale_multiplier
#'   Timestep scaling (inputs arrive already scaled; the ratio modulates
#'   the a2v/v2a gates).
#' @param rope_type "split" (LTX-2.3) or "interleaved".
#' @param gated_attn,cross_attn_mod,audio_gated_attn,audio_cross_attn_mod,perturbed_attn
#'   LTX-2.3 feature flags (all TRUE for the 2.3 checkpoints).
#'
#' @export
ltx23_transformer <- torch::nn_module(
  "ltx23_transformer",
  initialize = function(
    in_channels = 128L,
    out_channels = 128L,
    patch_size = 1L,
    patch_size_t = 1L,
    num_attention_heads = 32L,
    attention_head_dim = 128L,
    cross_attention_dim = 4096L,
    vae_scale_factors = c(8L, 32L, 32L),
    pos_embed_max_pos = 20L,
    base_height = 2048L,
    base_width = 2048L,
    gated_attn = TRUE,
    cross_attn_mod = TRUE,
    audio_in_channels = 128L,
    audio_out_channels = 128L,
    audio_patch_size = 1L,
    audio_patch_size_t = 1L,
    audio_num_attention_heads = 32L,
    audio_attention_head_dim = 64L,
    audio_cross_attention_dim = 2048L,
    audio_scale_factor = 4L,
    audio_pos_embed_max_pos = 20L,
    audio_sampling_rate = 16000L,
    audio_hop_length = 160L,
    audio_gated_attn = TRUE,
    audio_cross_attn_mod = TRUE,
    num_layers = 48L,
    norm_eps = 1e-6,
    rope_theta = 10000.0,
    rope_double_precision = TRUE,
    causal_offset = 1L,
    timestep_scale_multiplier = 1000,
    cross_attn_timestep_scale_multiplier = 1000,
    rope_type = "split",
    perturbed_attn = TRUE
  ) {
    inner_dim <- num_attention_heads * attention_head_dim
    audio_inner_dim <- audio_num_attention_heads * audio_attention_head_dim
    self$inner_dim <- inner_dim
    self$audio_inner_dim <- audio_inner_dim
    self$timestep_scale_multiplier <- timestep_scale_multiplier
    self$cross_attn_timestep_scale_multiplier <- cross_attn_timestep_scale_multiplier
    self$prompt_modulation <- cross_attn_mod || audio_cross_attn_mod

    # 1. Patchified input projections
    self$proj_in <- torch::nn_linear(in_channels, inner_dim)
    self$audio_proj_in <- torch::nn_linear(audio_in_channels, audio_inner_dim)

    # 2. Global timestep modulation (9 params each with cross-attn mod)
    video_mod_params <- if (cross_attn_mod) 9L else 6L
    audio_mod_params <- if (audio_cross_attn_mod) 9L else 6L
    self$time_embed <- ltx23_ada_layer_norm_single(inner_dim, video_mod_params)
    self$audio_time_embed <- ltx23_ada_layer_norm_single(audio_inner_dim, audio_mod_params)

    # Global a2v/v2a cross-attention modulation
    self$av_cross_attn_video_scale_shift <- ltx23_ada_layer_norm_single(inner_dim, 4L)
    self$av_cross_attn_audio_scale_shift <- ltx23_ada_layer_norm_single(audio_inner_dim, 4L)
    self$av_cross_attn_video_a2v_gate <- ltx23_ada_layer_norm_single(inner_dim, 1L)
    self$av_cross_attn_audio_v2a_gate <- ltx23_ada_layer_norm_single(audio_inner_dim, 1L)

    # Output layer modulation
    self$scale_shift_table <- torch::nn_parameter(
      torch::torch_randn(2L, inner_dim) / sqrt(inner_dim)
    )
    self$audio_scale_shift_table <- torch::nn_parameter(
      torch::torch_randn(2L, audio_inner_dim) / sqrt(audio_inner_dim)
    )

    # Prompt modulation (LTX-2.3)
    if (self$prompt_modulation) {
      self$prompt_adaln <- ltx23_ada_layer_norm_single(inner_dim, 2L)
      self$audio_prompt_adaln <- ltx23_ada_layer_norm_single(audio_inner_dim, 2L)
    }

    # 3. RoPE embedders (self-attention per modality + a2v/v2a cross)
    self$rope <- ltx23_rotary_pos_embed(
      dim = inner_dim, patch_size = patch_size, patch_size_t = patch_size_t,
      base_num_frames = pos_embed_max_pos, base_height = base_height,
      base_width = base_width, scale_factors = vae_scale_factors,
      theta = rope_theta, causal_offset = causal_offset, modality = "video",
      double_precision = rope_double_precision, rope_type = rope_type,
      num_attention_heads = num_attention_heads
    )
    self$audio_rope <- ltx23_rotary_pos_embed(
      dim = audio_inner_dim, patch_size = audio_patch_size,
      patch_size_t = audio_patch_size_t, base_num_frames = audio_pos_embed_max_pos,
      sampling_rate = audio_sampling_rate, hop_length = audio_hop_length,
      scale_factors = audio_scale_factor,
      theta = rope_theta, causal_offset = causal_offset, modality = "audio",
      double_precision = rope_double_precision, rope_type = rope_type,
      num_attention_heads = audio_num_attention_heads
    )
    ca_max_pos <- max(pos_embed_max_pos, audio_pos_embed_max_pos)
    self$cross_attn_rope <- ltx23_rotary_pos_embed(
      dim = audio_cross_attention_dim, patch_size = patch_size,
      patch_size_t = patch_size_t, base_num_frames = ca_max_pos,
      base_height = base_height, base_width = base_width,
      theta = rope_theta, causal_offset = causal_offset, modality = "video",
      double_precision = rope_double_precision, rope_type = rope_type,
      num_attention_heads = num_attention_heads
    )
    self$cross_attn_audio_rope <- ltx23_rotary_pos_embed(
      dim = audio_cross_attention_dim, patch_size = audio_patch_size,
      patch_size_t = audio_patch_size_t, base_num_frames = ca_max_pos,
      sampling_rate = audio_sampling_rate, hop_length = audio_hop_length,
      theta = rope_theta, causal_offset = causal_offset, modality = "audio",
      double_precision = rope_double_precision, rope_type = rope_type,
      num_attention_heads = audio_num_attention_heads
    )

    # 4. Transformer blocks
    self$transformer_blocks <- torch::nn_module_list(lapply(seq_len(num_layers), function(i) {
      ltx23_transformer_block(
        dim = inner_dim,
        num_attention_heads = num_attention_heads,
        attention_head_dim = attention_head_dim,
        cross_attention_dim = cross_attention_dim,
        audio_dim = audio_inner_dim,
        audio_num_attention_heads = audio_num_attention_heads,
        audio_attention_head_dim = audio_attention_head_dim,
        audio_cross_attention_dim = audio_cross_attention_dim,
        video_gated_attn = gated_attn,
        video_cross_attn_adaln = cross_attn_mod,
        audio_gated_attn = audio_gated_attn,
        audio_cross_attn_adaln = audio_cross_attn_mod,
        eps = norm_eps,
        elementwise_affine = FALSE,
        rope_type = rope_type,
        perturbed_attn = perturbed_attn
      )
    }))

    # 5. Output layers
    self$norm_out <- torch::nn_layer_norm(inner_dim, eps = 1e-6, elementwise_affine = FALSE)
    self$proj_out <- torch::nn_linear(inner_dim, out_channels)
    self$audio_norm_out <- torch::nn_layer_norm(audio_inner_dim, eps = 1e-6,
      elementwise_affine = FALSE)
    self$audio_proj_out <- torch::nn_linear(audio_inner_dim, audio_out_channels)
  },
  forward = function(
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    audio_encoder_hidden_states,
    timestep,
    audio_timestep = NULL,
    sigma = NULL,
    audio_sigma = NULL,
    encoder_attention_mask = NULL,
    audio_encoder_attention_mask = NULL,
    num_frames = NULL,
    height = NULL,
    width = NULL,
    fps = 24.0,
    audio_num_frames = NULL,
    video_coords = NULL,
    audio_coords = NULL,
    isolate_modalities = FALSE,
    spatio_temporal_guidance_blocks = NULL,
    perturbation_mask = NULL,
    use_cross_timestep = FALSE,
    video_self_attention_mask = NULL
  ) {
    audio_timestep <- audio_timestep %||% timestep
    audio_sigma <- audio_sigma %||% sigma
    if (self$prompt_modulation && is.null(sigma)) {
      stop("sigma is required for LTX-2.3 prompt modulation")
    }

    # Multiplicative [B, S] masks -> additive bias [B, 1, S]
    if (!is.null(encoder_attention_mask) && encoder_attention_mask$ndim == 2L) {
      encoder_attention_mask <- ((1 - encoder_attention_mask$to(
        dtype = hidden_states$dtype
      )) * -10000.0)$unsqueeze(2L)
    }
    if (!is.null(audio_encoder_attention_mask) && audio_encoder_attention_mask$ndim == 2L) {
      audio_encoder_attention_mask <- ((1 - audio_encoder_attention_mask$to(
        dtype = audio_hidden_states$dtype
      )) * -10000.0)$unsqueeze(2L)
    }
    if (!is.null(video_self_attention_mask)) {
      video_self_attention_mask <- (1 - video_self_attention_mask$to(
        dtype = hidden_states$dtype
      )) * -10000.0
    }

    batch_size <- hidden_states$shape[1]

    # 1. RoPE frequencies
    if (is.null(video_coords)) {
      video_coords <- self$rope$prepare_video_coords(
        batch_size, num_frames, height, width, hidden_states$device, fps = fps
      )
    }
    if (is.null(audio_coords)) {
      audio_coords <- self$audio_rope$prepare_audio_coords(
        batch_size, audio_num_frames, audio_hidden_states$device
      )
    }
    video_rotary_emb <- self$rope(video_coords, device = hidden_states$device)
    audio_rotary_emb <- self$audio_rope(audio_coords, device = audio_hidden_states$device)
    # Cross-modal attention uses the temporal axis only
    video_ca_rotary_emb <- self$cross_attn_rope(
      video_coords$narrow(2L, 1L, 1L), device = hidden_states$device
    )
    audio_ca_rotary_emb <- self$cross_attn_audio_rope(
      audio_coords$narrow(2L, 1L, 1L), device = audio_hidden_states$device
    )

    # 2. Input projections
    hidden_states <- self$proj_in(hidden_states)
    audio_hidden_states <- self$audio_proj_in(audio_hidden_states)

    # 3. Timestep embeddings and global modulation parameters
    gate_scale_factor <- self$cross_attn_timestep_scale_multiplier /
      self$timestep_scale_multiplier

    te <- self$time_embed(timestep$flatten(), hidden_dtype = hidden_states$dtype)
    temb <- te[[1]]$view(c(batch_size, -1L, te[[1]]$shape[2]))
    embedded_timestep <- te[[2]]$view(c(batch_size, -1L, te[[2]]$shape[2]))

    ate <- self$audio_time_embed(audio_timestep$flatten(),
      hidden_dtype = audio_hidden_states$dtype)
    temb_audio <- ate[[1]]$view(c(batch_size, -1L, ate[[1]]$shape[2]))
    audio_embedded_timestep <- ate[[2]]$view(c(batch_size, -1L, ate[[2]]$shape[2]))

    if (self$prompt_modulation) {
      pe <- self$prompt_adaln(sigma$flatten(), hidden_dtype = hidden_states$dtype)
      temb_prompt <- pe[[1]]$view(c(batch_size, -1L, pe[[1]]$shape[2]))
      ape <- self$audio_prompt_adaln(audio_sigma$flatten(),
        hidden_dtype = audio_hidden_states$dtype)
      temb_prompt_audio <- ape[[1]]$view(c(batch_size, -1L, ape[[1]]$shape[2]))
    } else {
      temb_prompt <- NULL
      temb_prompt_audio <- NULL
    }

    # a2v/v2a modulation; 2.3 modulates each modality by the *other*
    # modality's sigma (use_cross_timestep)
    video_ca_timestep <- if (use_cross_timestep) audio_sigma$flatten() else timestep$flatten()
    vcss <- self$av_cross_attn_video_scale_shift(
      video_ca_timestep, hidden_dtype = hidden_states$dtype
    )[[1]]
    vcg <- self$av_cross_attn_video_a2v_gate(
      video_ca_timestep * gate_scale_factor, hidden_dtype = hidden_states$dtype
    )[[1]]
    video_ca_scale_shift <- vcss$view(c(batch_size, -1L, vcss$shape[2]))
    video_ca_a2v_gate <- vcg$view(c(batch_size, -1L, vcg$shape[2]))

    audio_ca_timestep <- if (use_cross_timestep) sigma$flatten() else audio_timestep$flatten()
    acss <- self$av_cross_attn_audio_scale_shift(
      audio_ca_timestep, hidden_dtype = audio_hidden_states$dtype
    )[[1]]
    acg <- self$av_cross_attn_audio_v2a_gate(
      audio_ca_timestep * gate_scale_factor, hidden_dtype = audio_hidden_states$dtype
    )[[1]]
    audio_ca_scale_shift <- acss$view(c(batch_size, -1L, acss$shape[2]))
    audio_ca_v2a_gate <- acg$view(c(batch_size, -1L, acg$shape[2]))

    # 4. STG perturbation setup
    stg_blocks <- spatio_temporal_guidance_blocks %||% integer(0)
    if (length(stg_blocks) > 0L && is.null(perturbation_mask)) {
      perturbation_mask <- torch::torch_zeros(batch_size, device = hidden_states$device)
    }
    if (!is.null(perturbation_mask) && perturbation_mask$ndim == 1L) {
      perturbation_mask <- perturbation_mask$unsqueeze(2L)$unsqueeze(3L)
    }
    all_perturbed <- if (!is.null(perturbation_mask)) {
      as.logical((perturbation_mask == 0)$all()$item())
    } else {
      FALSE
    }

    # 5. Transformer blocks
    fp8_mode <- isTRUE(getOption("diffuseR.use_fp8"))
    for (block_idx in seq_along(self$transformer_blocks)) {
      is_stg_block <- (block_idx - 1L) %in% stg_blocks# reference indices are 0-based
      res <- self$transformer_blocks[[block_idx]](
        hidden_states = hidden_states,
        audio_hidden_states = audio_hidden_states,
        encoder_hidden_states = encoder_hidden_states,
        audio_encoder_hidden_states = audio_encoder_hidden_states,
        temb = temb,
        temb_audio = temb_audio,
        temb_ca_scale_shift = video_ca_scale_shift,
        temb_ca_audio_scale_shift = audio_ca_scale_shift,
        temb_ca_gate = video_ca_a2v_gate,
        temb_ca_audio_gate = audio_ca_v2a_gate,
        temb_prompt = temb_prompt,
        temb_prompt_audio = temb_prompt_audio,
        video_rotary_emb = video_rotary_emb,
        audio_rotary_emb = audio_rotary_emb,
        ca_video_rotary_emb = video_ca_rotary_emb,
        ca_audio_rotary_emb = audio_ca_rotary_emb,
        encoder_attention_mask = encoder_attention_mask,
        audio_encoder_attention_mask = audio_encoder_attention_mask,
        self_attention_mask = video_self_attention_mask,
        use_a2v_cross_attention = !isolate_modalities,
        use_v2a_cross_attention = !isolate_modalities,
        perturbation_mask = if (is_stg_block) perturbation_mask else NULL,
        all_perturbed = if (is_stg_block) all_perturbed else FALSE
      )
      hidden_states <- res[[1]]
      audio_hidden_states <- res[[2]]
      if (fp8_mode) {
        # Dequantized fp8 temporaries only free once R's GC finalizes them
        gc(verbose = FALSE)
      }
    }

    # 6. Output layers
    scale_shift_values <- self$scale_shift_table$unsqueeze(1L)$unsqueeze(1L) +
      embedded_timestep$unsqueeze(3L)
    shift <- scale_shift_values[, , 1, ]
    scale <- scale_shift_values[, , 2, ]
    hidden_states <- self$norm_out(hidden_states)
    hidden_states <- hidden_states * (1 + scale) + shift
    output <- self$proj_out(hidden_states)

    audio_scale_shift_values <- self$audio_scale_shift_table$unsqueeze(1L)$unsqueeze(1L) +
      audio_embedded_timestep$unsqueeze(3L)
    audio_shift <- audio_scale_shift_values[, , 1, ]
    audio_scale <- audio_scale_shift_values[, , 2, ]
    audio_hidden_states <- self$audio_norm_out(audio_hidden_states)
    audio_hidden_states <- audio_hidden_states * (1 + audio_scale) + audio_shift
    audio_output <- self$audio_proj_out(audio_hidden_states)

    list(sample = output, audio_sample = audio_output)
  }
)

#' Map an official DiT checkpoint key to the R module name
#'
#' Applies the official-to-diffusers renames for the LTX-2.3 transformer
#' (cf. diffusers scripts/convert_ltx2_to_diffusers.py). Our module tree
#' matches the diffusers names, so this is the full mapping.
#'
#' @param key Character. Checkpoint key (with or without the
#'   \code{model.diffusion_model.} prefix).
#'
#' @return Character. Module parameter/buffer name.
#'
#' @export
ltx23_map_dit_key <- function(key) {
  key <- sub("^model\\.diffusion_model\\.", "", key)

  # Substring renames; *_adaln_single variants must precede the bare
  # adaln_single prefix renames below
  key <- gsub("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift", key, fixed = TRUE)
  key <- gsub("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate", key, fixed = TRUE)
  key <- gsub("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift", key, fixed = TRUE)
  key <- gsub("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate", key, fixed = TRUE)
  key <- gsub("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table", key, fixed = TRUE)
  key <- gsub("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table", key, fixed = TRUE)
  key <- gsub("prompt_adaln_single", "prompt_adaln", key, fixed = TRUE)
  key <- sub("^audio_adaln_single\\.", "audio_time_embed.", key)
  key <- sub("^adaln_single\\.", "time_embed.", key)
  key <- gsub("patchify_proj", "proj_in", key, fixed = TRUE)
  key <- gsub("q_norm", "norm_q", key, fixed = TRUE)
  key <- gsub("k_norm", "norm_k", key, fixed = TRUE)
  key
}
