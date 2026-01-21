# LTX2 DiT Transformer Model
# Full audio-video transformer matching HuggingFace diffusers implementation

# ------------------------------------------------------------------------------
# LTX2 Audio-Video Rotary Positional Embeddings
# ------------------------------------------------------------------------------

#' LTX2 Audio-Video Rotary Positional Embeddings
#' @keywords internal
ltx2_audio_video_rotary_pos_embed <- torch::nn_module(
  "LTX2AudioVideoRotaryPosEmbed",
  initialize = function(
    dim,
    patch_size = 1L,
    patch_size_t = 1L,
    base_num_frames = 20L,
    base_height = 2048L,
    base_width = 2048L,
    sampling_rate = 16000L,
    hop_length = 160L,
    scale_factors = c(8L, 32L, 32L),
    theta = 10000.0,
    causal_offset = 1L,
    modality = "video",
    double_precision = TRUE,
    rope_type = "interleaved",
    num_attention_heads = 32L
  ) {

    self$dim <- dim
    self$patch_size <- patch_size
    self$patch_size_t <- patch_size_t

    if (!rope_type %in% c("interleaved", "split")) {
      stop("rope_type must be 'interleaved' or 'split'")
    }
    self$rope_type <- rope_type

    self$base_num_frames <- base_num_frames
    self$num_attention_heads <- num_attention_heads

    # Video-specific
    self$base_height <- base_height
    self$base_width <- base_width

    # Audio-specific
    self$sampling_rate <- sampling_rate
    self$hop_length <- hop_length
    self$audio_latents_per_second <- as.numeric(sampling_rate) / as.numeric(hop_length) / as.numeric(scale_factors[1])

    self$scale_factors <- scale_factors
    self$theta <- theta
    self$causal_offset <- causal_offset

    self$modality <- modality
    if (!modality %in% c("video", "audio")) {
      stop("modality must be 'video' or 'audio'")
    }
    self$double_precision <- double_precision
  },

  prepare_video_coords = function(
    batch_size,
    num_frames,
    height,
    width,
    device,
    fps = 24.0
  ) {
    # Generate grid coordinates for each spatiotemporal dimension
    grid_f <- torch::torch_arange(start = 0, end = num_frames - 1L, step = self$patch_size_t,
      dtype = torch::torch_float32(), device = device)
    grid_h <- torch::torch_arange(start = 0, end = height - 1L, step = self$patch_size,
      dtype = torch::torch_float32(), device = device)
    grid_w <- torch::torch_arange(start = 0, end = width - 1L, step = self$patch_size,
      dtype = torch::torch_float32(), device = device)

    # Create meshgrid (F, H, W order)
    grids <- torch::torch_meshgrid(list(grid_f, grid_h, grid_w), indexing = "ij")
    grid <- torch::torch_stack(grids, dim = 1L) # [3, N_F, N_H, N_W]

    # Get patch boundaries
    patch_size_vec <- c(self$patch_size_t, self$patch_size, self$patch_size)
    patch_size_delta <- torch::torch_tensor(patch_size_vec, dtype = grid$dtype, device = grid$device)
    patch_ends <- grid + patch_size_delta$view(c(3L, 1L, 1L, 1L))

    # Combine start and end coordinates
    latent_coords <- torch::torch_stack(list(grid, patch_ends), dim = - 1L) # [3, N_F, N_H, N_W, 2]

    # Reshape to [batch_size, 3, num_patches, 2]
    latent_coords <- latent_coords$flatten(start_dim = 2L, end_dim = 4L)
    latent_coords <- latent_coords$unsqueeze(1L)$`repeat`(c(batch_size, 1L, 1L, 1L))

    # Scale to pixel space
    scale_tensor <- torch::torch_tensor(self$scale_factors, device = latent_coords$device)
    pixel_coords <- latent_coords * scale_tensor$view(c(1L, 3L, 1L, 1L))

    # Handle causal offset for temporal dimension
    pixel_coords[, 1,,] <- (pixel_coords[, 1,,] + self$causal_offset - self$scale_factors[1])$clamp(min = 0)

    # Scale temporal by FPS
    pixel_coords[, 1,,] <- pixel_coords[, 1,,] / fps

    pixel_coords
  },

  prepare_audio_coords = function(
    batch_size,
    num_frames,
    device,
    shift = 0L
  ) {
    # Generate coordinates in the frame (time) dimension
    grid_f <- torch::torch_arange(start = shift, end = num_frames + shift - 1L, step = self$patch_size_t,
      dtype = torch::torch_float32(), device = device)

    audio_scale_factor <- self$scale_factors[1]

    # Calculate start timestamps
    grid_start_mel <- grid_f * audio_scale_factor
    grid_start_mel <- (grid_start_mel + self$causal_offset - audio_scale_factor)$clamp(min = 0)
    grid_start_s <- grid_start_mel * self$hop_length / self$sampling_rate

    # Calculate end timestamps
    grid_end_mel <- (grid_f + self$patch_size_t) * audio_scale_factor
    grid_end_mel <- (grid_end_mel + self$causal_offset - audio_scale_factor)$clamp(min = 0)
    grid_end_s <- grid_end_mel * self$hop_length / self$sampling_rate

    audio_coords <- torch::torch_stack(list(grid_start_s, grid_end_s), dim = - 1L) # [num_patches, 2]
    audio_coords <- audio_coords$unsqueeze(1L)$expand(c(batch_size, - 1L, - 1L)) # [batch_size, num_patches, 2]
    audio_coords <- audio_coords$unsqueeze(2L) # [batch_size, 1, num_patches, 2]

    audio_coords
  },

  prepare_coords = function(...) {
    if (self$modality == "video") {
      self$prepare_video_coords(...)
    } else {
      self$prepare_audio_coords(...)
    }
  },

  forward = function(
    coords,
    device = NULL,
    dtype = NULL
  ) {
    if (is.null(device)) device <- coords$device
    # Store target dtype for conversion at end (RoPE computed in float32 for precision)

    num_pos_dims <- coords$shape[2]

    # If coords are patch boundaries, use midpoint
    if (coords$ndim == 4L) {
      coords_chunks <- coords$chunk(2L, dim = - 1L)
      coords_start <- coords_chunks[[1]]
      coords_end <- coords_chunks[[2]]
      coords <- (coords_start + coords_end) / 2.0
      coords <- coords$squeeze(- 1L) # [B, num_pos_dims, num_patches]
    }

    # Get coordinates as fraction of base data shape
    if (self$modality == "video") {
      max_positions <- c(self$base_num_frames, self$base_height, self$base_width)
    } else {
      max_positions <- c(self$base_num_frames)
    }

    # [B, num_pos_dims, num_patches] -> [B, num_patches, num_pos_dims]
    grid_parts <- lapply(seq_len(num_pos_dims), function(i) {
        coords[, i,] / max_positions[i]
      })
    grid <- torch::torch_stack(grid_parts, dim = - 1L)$to(device = device)

    num_rope_elems <- num_pos_dims * 2L

    # Create frequency grid
    if (self$double_precision) {
      freqs_dtype <- torch::torch_float64()
    } else {
      freqs_dtype <- torch::torch_float32()
    }
    pow_indices <- torch::torch_pow(
      self$theta,
      torch::torch_linspace(start = 0.0, end = 1.0, steps = self$dim %/% num_rope_elems,
        dtype = freqs_dtype, device = device)
    )
    freqs <- (pow_indices * pi / 2.0)$to(dtype = torch::torch_float32())

    # Outer product
    freqs <- (grid$unsqueeze(- 1L) * 2 - 1) * freqs# [B, num_patches, num_pos_dims, dim/num_elems]
    freqs <- freqs$transpose(- 1L, - 2L)$flatten(start_dim = 3L) # [B, num_patches, dim/2]

    # Get cos/sin frequencies
    if (self$rope_type == "interleaved") {
      cos_freqs <- freqs$cos()$repeat_interleave(2L, dim = - 1L)
      sin_freqs <- freqs$sin()$repeat_interleave(2L, dim = - 1L)

      if (self$dim %% num_rope_elems != 0L) {
        pad_size <- self$dim %% num_rope_elems
        cos_padding <- torch::torch_ones_like(cos_freqs[,, 1:pad_size])
        sin_padding <- torch::torch_zeros_like(sin_freqs[,, 1:pad_size])
        cos_freqs <- torch::torch_cat(list(cos_padding, cos_freqs), dim = - 1L)
        sin_freqs <- torch::torch_cat(list(sin_padding, sin_freqs), dim = - 1L)
      }
    } else {
      # split type
      expected_freqs <- self$dim %/% 2L
      current_freqs <- freqs$shape[length(freqs$shape)]
      pad_size <- expected_freqs - current_freqs

      cos_freq <- freqs$cos()
      sin_freq <- freqs$sin()

      if (pad_size != 0L) {
        cos_padding <- torch::torch_ones_like(cos_freq[,, 1:pad_size])
        sin_padding <- torch::torch_zeros_like(sin_freq[,, 1:pad_size])
        cos_freq <- torch::torch_cat(list(cos_padding, cos_freq), dim = - 1L)
        sin_freq <- torch::torch_cat(list(sin_padding, sin_freq), dim = - 1L)
      }

      # Reshape for multi-head attention
      b <- cos_freq$shape[1]
      t <- cos_freq$shape[2]

      cos_freq <- cos_freq$reshape(c(b, t, self$num_attention_heads, - 1L))
      sin_freq <- sin_freq$reshape(c(b, t, self$num_attention_heads, - 1L))

      cos_freqs <- cos_freq$transpose(2L, 3L) # (B, H, T, D//2)
      sin_freqs <- sin_freq$transpose(2L, 3L)
    }

    # Convert to target dtype if specified (for mixed precision)
    if (!is.null(dtype)) {
      cos_freqs <- cos_freqs$to(dtype = dtype)
      sin_freqs <- sin_freqs$to(dtype = dtype)
    }

    list(cos_freqs, sin_freqs)
  }
)

# ------------------------------------------------------------------------------
# LTX2 Video Transformer 3D Model
# ------------------------------------------------------------------------------

#' LTX2 Video Transformer 3D Model (Audio-Video)
#'
#' Full audio-video transformer matching HuggingFace diffusers implementation.
#'
#' @param in_channels Integer. Video input channels (default: 128).
#' @param out_channels Integer. Video output channels (default: 128).
#' @param patch_size Integer. Spatial patch size (default: 1).
#' @param patch_size_t Integer. Temporal patch size (default: 1).
#' @param num_attention_heads Integer. Video attention heads (default: 32).
#' @param attention_head_dim Integer. Video attention head dimension (default: 128).
#' @param cross_attention_dim Integer. Video cross-attention dimension (default: 4096).
#' @param vae_scale_factors Integer vector. VAE scale factors (default: c(8, 32, 32)).
#' @param pos_embed_max_pos Integer. Max position for RoPE (default: 20).
#' @param base_height Integer. Base height for RoPE (default: 2048).
#' @param base_width Integer. Base width for RoPE (default: 2048).
#' @param audio_in_channels Integer. Audio input channels (default: 128).
#' @param audio_out_channels Integer. Audio output channels (default: 128).
#' @param audio_patch_size Integer. Audio patch size (default: 1).
#' @param audio_patch_size_t Integer. Audio temporal patch size (default: 1).
#' @param audio_num_attention_heads Integer. Audio attention heads (default: 32).
#' @param audio_attention_head_dim Integer. Audio head dimension (default: 64).
#' @param audio_cross_attention_dim Integer. Audio cross-attention dim (default: 2048).
#' @param audio_scale_factor Integer. Audio scale factor (default: 4).
#' @param audio_pos_embed_max_pos Integer. Audio max position (default: 20).
#' @param audio_sampling_rate Integer. Audio sampling rate (default: 16000).
#' @param audio_hop_length Integer. Audio hop length (default: 160).
#' @param num_layers Integer. Number of transformer layers (default: 48).
#' @param activation_fn Character. Activation function (default: "gelu-approximate").
#' @param qk_norm Character. QK normalization type (default: "rms_norm_across_heads").
#' @param norm_elementwise_affine Logical. Use elementwise affine in norms (default: FALSE).
#' @param norm_eps Numeric. Epsilon for normalization (default: 1e-6).
#' @param caption_channels Integer. Caption embedding channels (default: 3840).
#' @param attention_bias Logical. Use bias in attention (default: TRUE).
#' @param attention_out_bias Logical. Use bias in attention output (default: TRUE).
#' @param rope_theta Numeric. Theta for RoPE (default: 10000).
#' @param rope_double_precision Logical. Use double precision for RoPE (default: TRUE).
#' @param causal_offset Integer. Causal offset for RoPE (default: 1).
#' @param timestep_scale_multiplier Numeric. Timestep scale (default: 1000).
#' @param cross_attn_timestep_scale_multiplier Numeric. Cross-attn timestep scale (default: 1000).
#' @param rope_type Character. RoPE type: "interleaved" or "split" (default: "interleaved").
#'
#' @return An nn_module representing the LTX2 video transformer.
#' @export
ltx2_video_transformer_3d_model <- torch::nn_module(
  "LTX2VideoTransformer3DModel",
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
    num_layers = 48L,
    activation_fn = "gelu-approximate",
    qk_norm = "rms_norm_across_heads",
    norm_elementwise_affine = FALSE,
    norm_eps = 1e-6,
    caption_channels = 3840L,
    attention_bias = TRUE,
    attention_out_bias = TRUE,
    rope_theta = 10000.0,
    rope_double_precision = TRUE,
    causal_offset = 1L,
    timestep_scale_multiplier = 1000L,
    cross_attn_timestep_scale_multiplier = 1000L,
    rope_type = "interleaved"
  ) {

    if (is.null(out_channels)) out_channels <- in_channels
    if (is.null(audio_out_channels)) audio_out_channels <- audio_in_channels

    self$timestep_scale_multiplier <- timestep_scale_multiplier
    self$cross_attn_timestep_scale_multiplier <- cross_attn_timestep_scale_multiplier

    inner_dim <- num_attention_heads * attention_head_dim
    audio_inner_dim <- audio_num_attention_heads * audio_attention_head_dim

    # 1. Patchification input projections
    self$proj_in <- make_linear(in_channels, inner_dim)
    self$audio_proj_in <- make_linear(audio_in_channels, audio_inner_dim)

    # 2. Prompt embeddings
    self$caption_projection <- pixart_alpha_text_projection(
      in_features = caption_channels, hidden_size = inner_dim
    )
    self$audio_caption_projection <- pixart_alpha_text_projection(
      in_features = caption_channels, hidden_size = audio_inner_dim
    )

    # 3. Timestep modulation
    # 3.1 Global timestep embedding and modulation
    self$time_embed <- ltx2_ada_layer_norm_single(inner_dim, num_mod_params = 6L, use_additional_conditions = FALSE)
    self$audio_time_embed <- ltx2_ada_layer_norm_single(audio_inner_dim, num_mod_params = 6L, use_additional_conditions = FALSE)

    # 3.2 Global cross-attention modulation
    self$av_cross_attn_video_scale_shift <- ltx2_ada_layer_norm_single(inner_dim, num_mod_params = 4L, use_additional_conditions = FALSE)
    self$av_cross_attn_audio_scale_shift <- ltx2_ada_layer_norm_single(audio_inner_dim, num_mod_params = 4L, use_additional_conditions = FALSE)
    self$av_cross_attn_video_a2v_gate <- ltx2_ada_layer_norm_single(inner_dim, num_mod_params = 1L, use_additional_conditions = FALSE)
    self$av_cross_attn_audio_v2a_gate <- ltx2_ada_layer_norm_single(audio_inner_dim, num_mod_params = 1L, use_additional_conditions = FALSE)

    # 3.3 Output layer modulation
    self$scale_shift_table <- torch::nn_parameter(torch::torch_randn(2L, inner_dim) / sqrt(inner_dim))
    self$audio_scale_shift_table <- torch::nn_parameter(torch::torch_randn(2L, audio_inner_dim) / sqrt(audio_inner_dim))

    # 4. Rotary Positional Embeddings
    # Video self-attention RoPE
    self$rope <- ltx2_audio_video_rotary_pos_embed(
      dim = inner_dim,
      patch_size = patch_size,
      patch_size_t = patch_size_t,
      base_num_frames = pos_embed_max_pos,
      base_height = base_height,
      base_width = base_width,
      scale_factors = vae_scale_factors,
      theta = rope_theta,
      causal_offset = causal_offset,
      modality = "video",
      double_precision = rope_double_precision,
      rope_type = rope_type,
      num_attention_heads = num_attention_heads
    )

    # Audio self-attention RoPE
    self$audio_rope <- ltx2_audio_video_rotary_pos_embed(
      dim = audio_inner_dim,
      patch_size = audio_patch_size,
      patch_size_t = audio_patch_size_t,
      base_num_frames = audio_pos_embed_max_pos,
      sampling_rate = audio_sampling_rate,
      hop_length = audio_hop_length,
      scale_factors = c(audio_scale_factor),
      theta = rope_theta,
      causal_offset = causal_offset,
      modality = "audio",
      double_precision = rope_double_precision,
      rope_type = rope_type,
      num_attention_heads = audio_num_attention_heads
    )

    # Cross-attention RoPE
    cross_attn_pos_embed_max_pos <- max(pos_embed_max_pos, audio_pos_embed_max_pos)

    self$cross_attn_rope <- ltx2_audio_video_rotary_pos_embed(
      dim = audio_cross_attention_dim,
      patch_size = patch_size,
      patch_size_t = patch_size_t,
      base_num_frames = cross_attn_pos_embed_max_pos,
      base_height = base_height,
      base_width = base_width,
      theta = rope_theta,
      causal_offset = causal_offset,
      modality = "video",
      double_precision = rope_double_precision,
      rope_type = rope_type,
      num_attention_heads = num_attention_heads
    )

    self$cross_attn_audio_rope <- ltx2_audio_video_rotary_pos_embed(
      dim = audio_cross_attention_dim,
      patch_size = audio_patch_size,
      patch_size_t = audio_patch_size_t,
      base_num_frames = cross_attn_pos_embed_max_pos,
      sampling_rate = audio_sampling_rate,
      hop_length = audio_hop_length,
      theta = rope_theta,
      causal_offset = causal_offset,
      modality = "audio",
      double_precision = rope_double_precision,
      rope_type = rope_type,
      num_attention_heads = audio_num_attention_heads
    )

    # 5. Transformer blocks
    self$transformer_blocks <- torch::nn_module_list(lapply(seq_len(num_layers), function(i) {
          ltx2_video_transformer_block(
            dim = inner_dim,
            num_attention_heads = num_attention_heads,
            attention_head_dim = attention_head_dim,
            cross_attention_dim = cross_attention_dim,
            audio_dim = audio_inner_dim,
            audio_num_attention_heads = audio_num_attention_heads,
            audio_attention_head_dim = audio_attention_head_dim,
            audio_cross_attention_dim = audio_cross_attention_dim,
            qk_norm = qk_norm,
            activation_fn = activation_fn,
            attention_bias = attention_bias,
            attention_out_bias = attention_out_bias,
            eps = norm_eps,
            elementwise_affine = norm_elementwise_affine,
            rope_type = rope_type
          )
        }))

    # 6. Output layers
    self$norm_out <- torch::nn_layer_norm(inner_dim, eps = 1e-6, elementwise_affine = FALSE)
    self$proj_out <- make_linear(inner_dim, out_channels)

    self$audio_norm_out <- torch::nn_layer_norm(audio_inner_dim, eps = 1e-6, elementwise_affine = FALSE)
    self$audio_proj_out <- make_linear(audio_inner_dim, audio_out_channels)

    self$gradient_checkpointing <- FALSE
  },

  forward = function(
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    audio_encoder_hidden_states,
    timestep,
    audio_timestep = NULL,
    encoder_attention_mask = NULL,
    audio_encoder_attention_mask = NULL,
    num_frames = NULL,
    height = NULL,
    width = NULL,
    fps = 24.0,
    audio_num_frames = NULL,
    video_coords = NULL,
    audio_coords = NULL
  ) {

    if (is.null(audio_timestep)) audio_timestep <- timestep

    # Convert attention masks to bias (use tensor ops to preserve dtype)
    # Formula: (1 - mask) * -10000.0  ->  (mask - 1) * 10000.0
    if (!is.null(encoder_attention_mask) && encoder_attention_mask$ndim == 2L) {
      mask_dtype <- hidden_states$dtype
      encoder_attention_mask <- encoder_attention_mask$to(dtype = mask_dtype)
      # Use tensor scalar to preserve dtype
      scale <- torch::torch_tensor(- 10000.0, dtype = mask_dtype, device = encoder_attention_mask$device)
      encoder_attention_mask <- encoder_attention_mask$sub(1)$neg()$mul(scale)
      encoder_attention_mask <- encoder_attention_mask$unsqueeze(2L)
    }

    if (!is.null(audio_encoder_attention_mask) && audio_encoder_attention_mask$ndim == 2L) {
      mask_dtype <- audio_hidden_states$dtype
      audio_encoder_attention_mask <- audio_encoder_attention_mask$to(dtype = mask_dtype)
      scale <- torch::torch_tensor(- 10000.0, dtype = mask_dtype, device = audio_encoder_attention_mask$device)
      audio_encoder_attention_mask <- audio_encoder_attention_mask$sub(1)$neg()$mul(scale)
      audio_encoder_attention_mask <- audio_encoder_attention_mask$unsqueeze(2L)
    }

    batch_size <- hidden_states$shape[1]

    # 1. Prepare RoPE positional embeddings
    if (is.null(video_coords)) {
      video_coords <- self$rope$prepare_video_coords(batch_size, num_frames, height, width, hidden_states$device, fps = fps)
    }
    if (is.null(audio_coords)) {
      audio_coords <- self$audio_rope$prepare_audio_coords(batch_size, audio_num_frames, audio_hidden_states$device)
    }

    video_rotary_emb <- self$rope(video_coords, device = hidden_states$device, dtype = hidden_states$dtype)
    audio_rotary_emb <- self$audio_rope(audio_coords, device = audio_hidden_states$device, dtype = audio_hidden_states$dtype)

    video_cross_attn_rotary_emb <- self$cross_attn_rope(video_coords[, 1:1,,], device = hidden_states$device, dtype = hidden_states$dtype)
    audio_cross_attn_rotary_emb <- self$cross_attn_audio_rope(audio_coords[, 1:1,,], device = audio_hidden_states$device, dtype = audio_hidden_states$dtype)

    # 2. Patchify input projections
    hidden_states <- self$proj_in(hidden_states)
    audio_hidden_states <- self$audio_proj_in(audio_hidden_states)

    # 3. Prepare timestep embeddings
    timestep_cross_attn_gate_scale_factor <- self$cross_attn_timestep_scale_multiplier / self$timestep_scale_multiplier

    # 3.1 Global timestep embedding
    temb_result <- self$time_embed(timestep$flatten(), batch_size = batch_size, hidden_dtype = hidden_states$dtype)
    temb <- temb_result[[1]]$view(c(batch_size, - 1L, temb_result[[1]]$shape[length(temb_result[[1]]$shape)]))
    embedded_timestep <- temb_result[[2]]$view(c(batch_size, - 1L, temb_result[[2]]$shape[length(temb_result[[2]]$shape)]))

    temb_audio_result <- self$audio_time_embed(audio_timestep$flatten(), batch_size = batch_size, hidden_dtype = audio_hidden_states$dtype)
    temb_audio <- temb_audio_result[[1]]$view(c(batch_size, - 1L, temb_audio_result[[1]]$shape[length(temb_audio_result[[1]]$shape)]))
    audio_embedded_timestep <- temb_audio_result[[2]]$view(c(batch_size, - 1L, temb_audio_result[[2]]$shape[length(temb_audio_result[[2]]$shape)]))

    # 3.2 Cross-attention modulation
    video_cross_attn_scale_shift_result <- self$av_cross_attn_video_scale_shift(timestep$flatten(), batch_size = batch_size, hidden_dtype = hidden_states$dtype)
    video_cross_attn_a2v_gate_result <- self$av_cross_attn_video_a2v_gate(timestep$flatten() * timestep_cross_attn_gate_scale_factor, batch_size = batch_size, hidden_dtype = hidden_states$dtype)

    video_cross_attn_scale_shift <- video_cross_attn_scale_shift_result[[1]]$view(c(batch_size, - 1L, video_cross_attn_scale_shift_result[[1]]$shape[length(video_cross_attn_scale_shift_result[[1]]$shape)]))
    video_cross_attn_a2v_gate <- video_cross_attn_a2v_gate_result[[1]]$view(c(batch_size, - 1L, video_cross_attn_a2v_gate_result[[1]]$shape[length(video_cross_attn_a2v_gate_result[[1]]$shape)]))

    audio_cross_attn_scale_shift_result <- self$av_cross_attn_audio_scale_shift(audio_timestep$flatten(), batch_size = batch_size, hidden_dtype = audio_hidden_states$dtype)
    audio_cross_attn_v2a_gate_result <- self$av_cross_attn_audio_v2a_gate(audio_timestep$flatten() * timestep_cross_attn_gate_scale_factor, batch_size = batch_size, hidden_dtype = audio_hidden_states$dtype)

    audio_cross_attn_scale_shift <- audio_cross_attn_scale_shift_result[[1]]$view(c(batch_size, - 1L, audio_cross_attn_scale_shift_result[[1]]$shape[length(audio_cross_attn_scale_shift_result[[1]]$shape)]))
    audio_cross_attn_v2a_gate <- audio_cross_attn_v2a_gate_result[[1]]$view(c(batch_size, - 1L, audio_cross_attn_v2a_gate_result[[1]]$shape[length(audio_cross_attn_v2a_gate_result[[1]]$shape)]))

    # 4. Prepare prompt embeddings
    encoder_hidden_states <- self$caption_projection(encoder_hidden_states)
    encoder_hidden_states <- encoder_hidden_states$view(c(batch_size, - 1L, hidden_states$shape[length(hidden_states$shape)]))

    audio_encoder_hidden_states <- self$audio_caption_projection(audio_encoder_hidden_states)
    audio_encoder_hidden_states <- audio_encoder_hidden_states$view(c(batch_size, - 1L, audio_hidden_states$shape[length(audio_hidden_states$shape)]))

    # 5. Run transformer blocks
    for (i in seq_along(self$transformer_blocks)) {
      block <- self$transformer_blocks[[i]]
      result <- block(
        hidden_states = hidden_states,
        audio_hidden_states = audio_hidden_states,
        encoder_hidden_states = encoder_hidden_states,
        audio_encoder_hidden_states = audio_encoder_hidden_states,
        temb = temb,
        temb_audio = temb_audio,
        temb_ca_scale_shift = video_cross_attn_scale_shift,
        temb_ca_audio_scale_shift = audio_cross_attn_scale_shift,
        temb_ca_gate = video_cross_attn_a2v_gate,
        temb_ca_audio_gate = audio_cross_attn_v2a_gate,
        video_rotary_emb = video_rotary_emb,
        audio_rotary_emb = audio_rotary_emb,
        ca_video_rotary_emb = video_cross_attn_rotary_emb,
        ca_audio_rotary_emb = audio_cross_attn_rotary_emb,
        encoder_attention_mask = encoder_attention_mask,
        audio_encoder_attention_mask = audio_encoder_attention_mask
      )
      hidden_states <- result[[1]]
      audio_hidden_states <- result[[2]]
    }

    # 6. Output layers
    scale_shift_values <- self$scale_shift_table$unsqueeze(1)$unsqueeze(1)$to(dtype = hidden_states$dtype) + embedded_timestep$unsqueeze(3)
    shift <- scale_shift_values[,, 1,]
    scale <- scale_shift_values[,, 2,]

    hidden_states <- self$norm_out(hidden_states)
    hidden_states <- hidden_states * scale$add(1) + shift
    output <- self$proj_out(hidden_states)

    audio_scale_shift_values <- self$audio_scale_shift_table$unsqueeze(1)$unsqueeze(1)$to(dtype = audio_hidden_states$dtype) + audio_embedded_timestep$unsqueeze(3)
    audio_shift <- audio_scale_shift_values[,, 1,]
    audio_scale <- audio_scale_shift_values[,, 2,]

    audio_hidden_states <- self$audio_norm_out(audio_hidden_states)
    audio_hidden_states <- audio_hidden_states * audio_scale$add(1) + audio_shift
    audio_output <- self$audio_proj_out(audio_hidden_states)

    list(sample = output, audio_sample = audio_output)
  }
)

# ------------------------------------------------------------------------------
# Weight Loading Functions
# ------------------------------------------------------------------------------

#' Load LTX2 DiT Transformer from safetensors
#'
#' Load pre-trained LTX2 transformer weights from HuggingFace safetensors files.
#' Supports both single file and sharded multi-file loading.
#'
#' @param weights_dir Character. Directory containing safetensors files.
#' @param config_path Character. Optional path to config.json. If NULL, uses default config.
#' @param device Character. Device to load weights to. Default: "cpu"
#' @param dtype Character. Data type ("float32", "float16", "bfloat16"). Default: "float16"
#' @param verbose Logical. Print loading progress. Default: TRUE
#' @return Initialized ltx2_video_transformer3d module
#' @export
load_ltx2_transformer <- function(
  weights_dir,
  config_path = NULL,
  device = "cpu",
  dtype = "float16",
  verbose = TRUE
) {

  # Look for config
  if (is.null(config_path)) {
    config_path <- file.path(weights_dir, "config.json")
  }

  config <- NULL
  if (file.exists(config_path)) {
    config <- jsonlite::fromJSON(config_path)
    if (verbose) message("Loaded config from: ", config_path)
  }

  # Create transformer with config or defaults
  if (!is.null(config)) {
    transformer <- ltx2_video_transformer_3d_model(
      in_channels = config$in_channels %||% 128L,
      out_channels = config$out_channels %||% 128L,
      patch_size = config$patch_size %||% 1L,
      patch_size_t = config$patch_size_t %||% 1L,
      num_attention_heads = config$num_attention_heads %||% 32L,
      attention_head_dim = config$attention_head_dim %||% 128L,
      cross_attention_dim = config$cross_attention_dim %||% 4096L,
      num_layers = config$num_layers %||% 48L,
      caption_channels = config$caption_channels %||% 3840L,
      qk_norm = config$qk_norm %||% "rms_norm_across_heads",
      norm_elementwise_affine = config$norm_elementwise_affine %||% FALSE,
      norm_eps = config$norm_eps %||% 1e-6,
      timestep_scale_multiplier = config$timestep_scale_multiplier %||% 1000.0,
      cross_attn_timestep_scale_multiplier = config$cross_attn_timestep_scale_multiplier %||% 1000.0,
      base_height = config$base_height %||% 2048L,
      base_width = config$base_width %||% 2048L,
      pos_embed_max_pos = config$pos_embed_max_pos %||% 20L,
      rope_theta = config$rope_theta %||% 10000.0,
      rope_type = config$rope_type %||% "split",
      causal_offset = config$causal_offset %||% 1L,
      audio_in_channels = config$audio_in_channels %||% 128L,
      audio_out_channels = config$audio_out_channels %||% 128L,
      audio_num_attention_heads = config$audio_num_attention_heads %||% 32L,
      audio_attention_head_dim = config$audio_attention_head_dim %||% 64L,
      audio_cross_attention_dim = config$audio_cross_attention_dim %||% 2048L,
      audio_sampling_rate = config$audio_sampling_rate %||% 16000L,
      audio_hop_length = config$audio_hop_length %||% 160L,
      audio_scale_factor = config$audio_scale_factor %||% 4L,
      audio_pos_embed_max_pos = config$audio_pos_embed_max_pos %||% 20L
    )
  } else {
    transformer <- ltx2_video_transformer_3d_model()
  }

  # Find weight files
  index_path <- file.path(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
  single_path <- file.path(weights_dir, "diffusion_pytorch_model.safetensors")

  if (file.exists(index_path)) {
    # Sharded weights
    if (verbose) message("Loading sharded weights from: ", weights_dir)
    load_ltx2_transformer_sharded(transformer, weights_dir, index_path, verbose = verbose)
  } else if (file.exists(single_path)) {
    # Single file
    if (verbose) message("Loading weights from: ", single_path)
    weights <- safetensors::safe_load_file(single_path, framework = "torch")
    load_ltx2_transformer_weights(transformer, weights, verbose = verbose)
  } else {
    stop("No weights found in: ", weights_dir)
  }

  # Convert dtype and move to device
  torch_dtype <- switch(dtype,
    "float32" = torch::torch_float32(),
    "float16" = torch::torch_float16(),
    "bfloat16" = torch::torch_bfloat16(),
    torch::torch_float16()
  )

  transformer$to(device = device, dtype = torch_dtype)

  if (verbose) message("Transformer loaded successfully on device: ", device, " dtype: ", dtype)
  transformer
}

#' Load sharded transformer weights
#'
#' @param transformer LTX2 transformer module
#' @param weights_dir Directory containing sharded safetensors
#' @param index_path Path to index.json
#' @param verbose Print progress
#' @keywords internal
load_ltx2_transformer_sharded <- function(
  transformer,
  weights_dir,
  index_path,
  verbose = TRUE
) {
  # Load index
  index <- jsonlite::fromJSON(index_path)
  weight_map <- index$weight_map

  # Get unique shard files
  shard_files <- unique(unlist(weight_map))
  if (verbose) message(sprintf("Loading %d shards...", length(shard_files)))

  total_loaded <- 0L
  total_skipped <- 0L

  for (i in seq_along(shard_files)) {
    shard_file <- shard_files[i]
    shard_path <- file.path(weights_dir, shard_file)

    if (!file.exists(shard_path)) {
      warning("Shard not found: ", shard_path)
      next
    }

    if (verbose) message(sprintf("[%d/%d] Loading %s...", i, length(shard_files), shard_file))

    weights <- safetensors::safe_load_file(shard_path, framework = "torch")
    result <- load_ltx2_transformer_weights(transformer, weights, verbose = FALSE)
    total_loaded <- total_loaded + result$loaded
    total_skipped <- total_skipped + result$skipped

    # Free memory
    rm(weights)
    gc()
  }

  if (verbose) {
    message(sprintf("Total: %d loaded, %d skipped", total_loaded, total_skipped))
  }

  invisible(list(loaded = total_loaded, skipped = total_skipped))
}

#' Load weights into LTX2 transformer module
#'
#' @param transformer LTX2 transformer module
#' @param weights Named list of weight tensors
#' @param verbose Print progress
#' @keywords internal
load_ltx2_transformer_weights <- function(
  transformer,
  weights,
  verbose = TRUE
) {
  native_params <- names(transformer$parameters)

  # Remap HuggingFace names to R module names
  remap_transformer_key <- function(key) {
    # HuggingFace uses nn.ModuleList for FeedForward:
    #   ff.net.0.proj.weight -> ff.act_fn.proj.weight
    #   ff.net.2.weight -> ff.proj_out.weight
    key <- gsub("\\.ff\\.net\\.0\\.", ".ff.act_fn.", key)
    key <- gsub("\\.ff\\.net\\.2\\.", ".ff.proj_out.", key)
    # Same for audio_ff
    key <- gsub("\\.audio_ff\\.net\\.0\\.", ".audio_ff.act_fn.", key)
    key <- gsub("\\.audio_ff\\.net\\.2\\.", ".audio_ff.proj_out.", key)

    # to_out.0 is used in both HF and our module (ModuleList)
    # No remapping needed for to_out
    key
  }

  loaded <- 0L
  skipped <- 0L
  unmapped <- character(0)

  torch::with_no_grad({
      for (hf_name in names(weights)) {
        native_name <- remap_transformer_key(hf_name)

        if (native_name %in% native_params) {
          hf_tensor <- weights[[hf_name]]
          native_tensor <- transformer$parameters[[native_name]]

          if (all(as.integer(hf_tensor$shape) == as.integer(native_tensor$shape))) {
            native_tensor$copy_(hf_tensor)
            loaded <- loaded + 1L
          } else {
            if (verbose) {
              message("Shape mismatch: ", native_name,
                " (HF: ", paste(as.integer(hf_tensor$shape), collapse = "x"),
                " vs R: ", paste(as.integer(native_tensor$shape), collapse = "x"), ")")
            }
            skipped <- skipped + 1L
          }
        } else {
          skipped <- skipped + 1L
          unmapped <- c(unmapped, paste0(hf_name, " -> ", native_name))
        }
      }
    })

  if (verbose) {
    message(sprintf("Transformer weights: %d loaded, %d skipped", loaded, skipped))
    if (length(unmapped) > 0 && length(unmapped) <= 20) {
      message("Unmapped parameters:")
      for (u in unmapped[1:min(20, length(unmapped))]) {
        message("  ", u)
      }
    }
    if (length(unmapped) > 20) {
      message("  ... and ", length(unmapped) - 20, " more")
    }
  }

  invisible(list(loaded = loaded, skipped = skipped, unmapped = unmapped))
}

# ------------------------------------------------------------------------------
# Latent Packing/Unpacking for DiT
# ------------------------------------------------------------------------------

#' Pack Video Latents for DiT
#'
#' Transforms 5D video latents \[B, C, F, H, W\] to 3D sequence \[B, F*H*W, C\]
#' for input to the DiT transformer.
#'
#' @param latents Tensor of shape \[B, C, F, H, W\].
#' @param patch_size Integer. Spatial patch size (default 1).
#' @param patch_size_t Integer. Temporal patch size (default 1).
#'
#' @return Packed tensor of shape \[B, num_patches, C*patch_size_t*patch_size^2\].
#'
#' @export
pack_video_latents <- function(
  latents,
  patch_size = 1L,
  patch_size_t = 1L
) {
  batch_size <- latents$shape[1]
  num_channels <- latents$shape[2]
  num_frames <- latents$shape[3]
  height <- latents$shape[4]
  width <- latents$shape[5]

  post_patch_frames <- num_frames %/% patch_size_t
  post_patch_height <- height %/% patch_size
  post_patch_width <- width %/% patch_size

  # Reshape: [B, C, F, H, W] -> [B, C, F//pt, pt, H//p, p, W//p, p]
  latents <- latents$reshape(c(
      batch_size, num_channels,
      post_patch_frames, patch_size_t,
      post_patch_height, patch_size,
      post_patch_width, patch_size
    ))

  # Permute: [B, C, F//pt, pt, H//p, p, W//p, p] -> [B, F//pt, H//p, W//p, C, pt, p, p]
  latents <- latents$permute(c(1L, 3L, 5L, 7L, 2L, 4L, 6L, 8L))

  # Flatten patches: [B, F//pt, H//p, W//p, C*pt*p*p]
  latents <- latents$flatten(start_dim = 5L, end_dim = 8L)

  # Flatten sequence: [B, F//pt * H//p * W//p, C*pt*p*p]
  latents <- latents$flatten(start_dim = 2L, end_dim = 4L)

  latents
}

#' Unpack Video Latents from DiT
#'
#' Transforms 3D sequence \[B, num_patches, D\] back to 5D video latents \[B, C, F, H, W\].
#'
#' @param latents Tensor of shape \[B, num_patches, D\].
#' @param num_frames Integer. Target number of latent frames.
#' @param height Integer. Target latent height.
#' @param width Integer. Target latent width.
#' @param patch_size Integer. Spatial patch size (default 1).
#' @param patch_size_t Integer. Temporal patch size (default 1).
#'
#' @return Unpacked tensor of shape \[B, C, F, H, W\].
#'
#' @export
unpack_video_latents <- function(
  latents,
  num_frames,
  height,
  width,
  patch_size = 1L,
  patch_size_t = 1L
) {
  batch_size <- latents$shape[1]

  post_patch_frames <- num_frames %/% patch_size_t
  post_patch_height <- height %/% patch_size
  post_patch_width <- width %/% patch_size

  # Unflatten sequence: [B, S, D] -> [B, F//pt, H//p, W//p, D]
  latents <- latents$reshape(c(batch_size, post_patch_frames, post_patch_height, post_patch_width, - 1L))

  # Calculate channel dimension
  d <- latents$shape[5]
  num_channels <- d %/% (patch_size_t * patch_size * patch_size)

  # Unflatten patches: [B, F//pt, H//p, W//p, C, pt, p, p]
  latents <- latents$reshape(c(
      batch_size, post_patch_frames, post_patch_height, post_patch_width,
      num_channels, patch_size_t, patch_size, patch_size
    ))

  # Permute back: [B, C, F//pt, pt, H//p, p, W//p, p]
  latents <- latents$permute(c(1L, 5L, 2L, 6L, 3L, 7L, 4L, 8L))

  # Flatten to video: [B, C, F, H, W]
  latents <- latents$flatten(start_dim = 7L, end_dim = 8L)
  latents <- latents$flatten(start_dim = 5L, end_dim = 6L)
  latents <- latents$flatten(start_dim = 3L, end_dim = 4L)

  latents
}

