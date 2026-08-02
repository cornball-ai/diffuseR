#' LTX-2.3 Rotary Positional Embeddings
#'
#' Fresh R port of the LTX rotary positional embedding scheme from the
#' diffusers reference implementation (Apache-2.0,
#' src/diffusers/models/transformers/transformer_ltx2.py). LTX 2.3 uses
#' the "split" RoPE layout everywhere; "interleaved" is kept for
#' completeness. Frequencies are computed in float64 per the checkpoint
#' config (frequencies_precision) and applied in float32.
#'
#' @name rope_ltx23
NULL

#' Apply interleaved rotary embeddings
#'
#' Rotates adjacent element pairs of the last dimension:
#' \code{out = x * cos + rotate_half(x) * sin} with pairs interleaved
#' (elements 1,2 form the first complex pair).
#'
#' @param x Tensor of shape [B, S, C].
#' @param freqs List of two tensors (cos, sin), each [B, S, C].
#'
#' @return Tensor with the same shape and dtype as \code{x}.
#'
#' @export
ltx23_apply_interleaved_rotary_emb <- function(x, freqs) {
    cos <- freqs[[1]]
    sin <- freqs[[2]]

    # Split last dim into (C/2, 2) pairs: real = even positions, imag = odd
    pairs <- x$unflatten(3L, c(-1L, 2L))
    x_real <- pairs[,,, 1] # [B, S, C/2]
    x_imag <- pairs[,,, 2]
    x_rotated <- torch::torch_stack(list(-x_imag, x_real), dim = -1L)$flatten(start_dim = 3L)

    out <- x$to(dtype = torch::torch_float32()) * cos +
    x_rotated$to(dtype = torch::torch_float32()) * sin
    out$to(dtype = x$dtype)
}

#' Apply split rotary embeddings
#'
#' Rotates element pairs formed by splitting the last dimension in half:
#' element i pairs with element i + d/2. The cos/sin tensors carry half
#' the head dimension.
#'
#' @param x Tensor of shape [B, H, T, D] (per-head layout), or [B, T, H*D]
#'   which is reshaped per-head when \code{freqs} is 4D.
#' @param freqs List of two tensors (cos, sin), each [B, H, T, D/2].
#'
#' @return Tensor with the same shape and dtype as \code{x}.
#'
#' @export
ltx23_apply_split_rotary_emb <- function(x, freqs) {
    cos <- freqs[[1]]
    sin <- freqs[[2]]

    x_dtype <- x$dtype
    needs_reshape <- FALSE
    if (x$ndim != 4L && cos$ndim == 4L) {
        # cos is [B, H, T, r] -> reshape x [B, T, H*D] to [B, H, T, D]
        b <- cos$shape[1]
        h <- cos$shape[2]
        t <- cos$shape[3]
        x <- x$reshape(c(b, t, h, -1L))$transpose(2L, 3L)
        needs_reshape <- TRUE
    }

    d <- x$shape[length(x$shape)]
    if (d %% 2L != 0L) {
        stop("Expected the last dimension of x to be even for split rotary, got ",
             d)
    }
    r <- d %/% 2L

    # First/second halves of the last dim form the rotation pairs
    x_first <- x$narrow(-1L, 1L, r)$to(dtype = torch::torch_float32())
    x_second <- x$narrow(-1L, r + 1L, r)$to(dtype = torch::torch_float32())

    out_first <- x_first * cos - x_second * sin
    out_second <- x_second * cos + x_first * sin
    out <- torch::torch_cat(list(out_first, out_second), dim = -1L)

    if (needs_reshape) {
        out <- out$transpose(2L, 3L)$reshape(c(b, t, -1L))
    }

    out$to(dtype = x_dtype)
}

#' LTX-2.3 audio/video rotary position embedder
#'
#' Computes RoPE cos/sin frequency tensors from spatiotemporal patch
#' coordinates. Video coordinates are 3D (frames scaled to seconds via
#' fps, height, width in pixel space); audio coordinates are 1D
#' (seconds). Coordinates are patch boundaries [start, end); the midpoint
#' is used as the position.
#'
#' @param dim Integer. Rotary dimension (attention head dim x heads for
#'   split type at model level; see reference).
#' @param patch_size,patch_size_t Integers. Spatial/temporal patch sizes.
#' @param base_num_frames,base_height,base_width Integers. Base grid the
#'   coordinates are normalized against.
#' @param sampling_rate,hop_length Integers. Audio spectrogram params.
#' @param scale_factors Integer vector. VAE (time, height, width) scale
#'   factors.
#' @param theta Numeric. RoPE theta.
#' @param causal_offset Integer. Temporal offset for the causal VAE
#'   (first frame has stride 1).
#' @param modality "video" or "audio".
#' @param double_precision Logical. Compute base frequencies in float64.
#' @param rope_type "split" (LTX 2.3) or "interleaved".
#' @param num_attention_heads Integer. Needed for the split layout.
#'
#' @return Module whose forward(coords, device) returns
#'   \code{list(cos_freqs, sin_freqs)}, the two rotary tables to apply
#'   to queries and keys.
#'
#' @export
ltx23_rotary_pos_embed <- torch::nn_module(
    "ltx23_rotary_pos_embed",
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
                          rope_type = "split",
                          num_attention_heads = 32L
    ) {
    if (!rope_type %in% c("interleaved", "split")) {
        stop("rope_type must be 'interleaved' or 'split', got: ", rope_type)
    }
    if (!modality %in% c("video", "audio")) {
        stop("modality must be 'video' or 'audio', got: ", modality)
    }
    self$dim <- as.integer(dim)
    self$patch_size <- as.integer(patch_size)
    self$patch_size_t <- as.integer(patch_size_t)
    self$base_num_frames <- base_num_frames
    self$base_height <- base_height
    self$base_width <- base_width
    self$sampling_rate <- sampling_rate
    self$hop_length <- hop_length
    self$scale_factors <- as.integer(scale_factors)
    self$theta <- theta
    self$causal_offset <- causal_offset
    self$modality <- modality
    self$double_precision <- double_precision
    self$rope_type <- rope_type
    self$num_attention_heads <- as.integer(num_attention_heads)
},

    # Patch boundaries [B, 3, num_patches, 2] in pixel/second space
    prepare_video_coords = function(batch_size, num_frames, height, width,
                                    device, fps = 24.0) {
    f32 <- torch::torch_float32()
    # torch_arange has an inclusive end; end = n - 1 matches Python's
    # exclusive arange for integer grids
    grid_f <- torch::torch_arange(start = 0, end = num_frames - 1,
                                  step = self$patch_size_t, dtype = f32,
                                  device = device)
    grid_h <- torch::torch_arange(
                                  start = 0, end = height - 1, step = self$patch_size,
                                  dtype = f32, device = device
    )
    grid_w <- torch::torch_arange(
                                  start = 0, end = width - 1, step = self$patch_size,
                                  dtype = f32, device = device
    )
    grid <- torch::torch_meshgrid(list(grid_f, grid_h, grid_w), indexing = "ij")
    grid <- torch::torch_stack(grid, dim = 1L) # [3, NF, NH, NW]

    patch_size <- c(self$patch_size_t, self$patch_size, self$patch_size)
    patch_delta <- torch::torch_tensor(patch_size, dtype = grid$dtype, device = grid$device)
    patch_ends <- grid + patch_delta$view(c(3L, 1L, 1L, 1L))

    latent_coords <- torch::torch_stack(list(grid, patch_ends), dim = -1L) # [3,NF,NH,NW,2]
    latent_coords <- latent_coords$flatten(start_dim = 2L, end_dim = 4L) # [3,N,2]
    latent_coords <- latent_coords$unsqueeze(1L)$`repeat`(c(batch_size, 1L, 1L, 1L))

    scale_tensor <- torch::torch_tensor(self$scale_factors, device = latent_coords$device)
    pixel_coords <- latent_coords * scale_tensor$view(c(1L, -1L, 1L, 1L))

    # First latent frame has temporal stride 1 in the causal VAE: shift and
    # clamp so timestamps stay causal and non-negative
    pixel_coords[, 1,,] <- (pixel_coords[, 1,,] + self$causal_offset -
        self$scale_factors[1])$clamp(min = 0)
    # Temporal coordinates in seconds
    pixel_coords[, 1,,] <- pixel_coords[, 1,,] / fps

    pixel_coords
},

    # Patch boundaries [B, 1, num_patches, 2] in seconds
    prepare_audio_coords = function(batch_size, num_frames, device, shift = 0L) {
    f32 <- torch::torch_float32()
    grid_f <- torch::torch_arange(
                                  start = shift, end = num_frames + shift - 1, step = self$patch_size_t,
                                  dtype = f32, device = device
    )

    audio_scale_factor <- self$scale_factors[1]
    grid_start_mel <- grid_f * audio_scale_factor
    grid_start_mel <- (grid_start_mel + self$causal_offset - audio_scale_factor)$clamp(min = 0)
    grid_start_s <- grid_start_mel * self$hop_length / self$sampling_rate

    grid_end_mel <- (grid_f + self$patch_size_t) * audio_scale_factor
    grid_end_mel <- (grid_end_mel + self$causal_offset - audio_scale_factor)$clamp(min = 0)
    grid_end_s <- grid_end_mel * self$hop_length / self$sampling_rate

    audio_coords <- torch::torch_stack(list(grid_start_s, grid_end_s), dim = -1L) # [N, 2]
    audio_coords <- audio_coords$unsqueeze(1L)$expand(c(batch_size, -1L, -1L))
    audio_coords$unsqueeze(2L) # [B, 1, N, 2]
},

    forward = function(coords, device = NULL) {
    device <- device %||% coords$device
    num_pos_dims <- coords$shape[2]

    # Patch boundaries [start, end) -> midpoint position
    if (coords$ndim == 4L) {
        chunks <- torch::torch_chunk(coords, 2L, dim = -1L)
        coords <- ((chunks[[1]] + chunks[[2]]) / 2.0)$squeeze(-1L) # [B, dims, N]
    }

    max_positions <- if (self$modality == "video") {
        c(self$base_num_frames, self$base_height, self$base_width)
    } else {
        self$base_num_frames
    }
    # [B, dims, N] -> [B, N, dims], each dim normalized to its base size
    grid <- torch::torch_stack(
                               lapply(seq_len(num_pos_dims), function(i) coords[, i,] / max_positions[i]),
                               dim = -1L
    )$to(device = device)

    num_rope_elems <- num_pos_dims * 2L

    freqs_dtype <- if (self$double_precision) torch::torch_float64() else torch::torch_float32()
    pow_indices <- torch::torch_pow(
                                    self$theta,
                                    torch::torch_linspace(
            start = 0.0, end = 1.0, steps = self$dim %/% num_rope_elems,
            dtype = freqs_dtype, device = device
        )
    )
    freqs <- (pow_indices * pi / 2.0)$to(dtype = torch::torch_float32())

    # Outer product of normalized positions (mapped to [-1, 1]) and freqs:
    # [B, N, dims, dim / num_rope_elems]
    freqs <- (grid$unsqueeze(-1L) * 2 - 1) * freqs
    freqs <- freqs$transpose(-1L, -2L)$flatten(start_dim = 3L) # [B, N, dim/2]

    if (self$rope_type == "interleaved") {
        cos_freqs <- freqs$cos()$repeat_interleave(2L, dim = -1L)
        sin_freqs <- freqs$sin()$repeat_interleave(2L, dim = -1L)

        pad <- self$dim %% num_rope_elems
        if (pad != 0L) {
            cos_padding <- torch::torch_ones_like(cos_freqs[,, 1:pad])
            sin_padding <- torch::torch_zeros_like(cos_freqs[,, 1:pad])
            cos_freqs <- torch::torch_cat(list(cos_padding, cos_freqs), dim = -1L)
            sin_freqs <- torch::torch_cat(list(sin_padding, sin_freqs), dim = -1L)
        }
    } else {
        expected_freqs <- self$dim %/% 2L
        current_freqs <- freqs$shape[length(freqs$shape)]
        pad_size <- expected_freqs - current_freqs
        cos_freqs <- freqs$cos()
        sin_freqs <- freqs$sin()

        if (pad_size != 0L) {
            cos_padding <- torch::torch_ones_like(cos_freqs[,, 1:pad_size])
            sin_padding <- torch::torch_zeros_like(sin_freqs[,, 1:pad_size])
            cos_freqs <- torch::torch_cat(list(cos_padding, cos_freqs), dim = -1L)
            sin_freqs <- torch::torch_cat(list(sin_padding, sin_freqs), dim = -1L)
        }

        # Per-head layout for split application: [B, H, N, dim/(2H)]
        b <- cos_freqs$shape[1]
        t <- cos_freqs$shape[2]
        cos_freqs <- cos_freqs$reshape(c(b, t, self$num_attention_heads, -1L))$transpose(2L, 3L)
        sin_freqs <- sin_freqs$reshape(c(b, t, self$num_attention_heads, -1L))$transpose(2L, 3L)
    }

    list(cos_freqs, sin_freqs)
}
)
