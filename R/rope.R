#' Rotary Position Embeddings (RoPE) for LTX-2 Video Models
#'
#' Implementation of rotary positional embeddings for 3D video (spatiotemporal)
#' coordinates as used in LTX-2. RoPE encodes position information directly
#' into the attention queries and keys without adding position embeddings.
#'
#' @name rope
#' @references
#' Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
#' "RoFormer: Enhanced Transformer with Rotary Position Embedding."
#' \url{https://arxiv.org/abs/2104.09864}
NULL

#' Create RoPE position embedder for video
#'
#' Creates a RoPE embedder configured for video generation, handling
#' spatiotemporal coordinates (frames, height, width).
#'
#' @param dim Integer. Dimension for RoPE (typically attention head dim * num_heads).
#' @param patch_size Integer. Spatial patch size. Default: 1
#' @param patch_size_t Integer. Temporal patch size. Default: 1
#' @param base_num_frames Integer. Base number of frames for normalization. Default: 20
#' @param base_height Integer. Base height for normalization. Default: 2048
#' @param base_width Integer. Base width for normalization. Default: 2048
#' @param scale_factors Numeric vector of length 3. VAE scale factors for
#'   (temporal, height, width). Default: c(8, 32, 32)
#' @param theta Numeric. Base frequency for RoPE. Default: 10000.0
#' @param causal_offset Integer. Offset for causal VAE modeling. Default: 1
#' @param double_precision Logical. Whether to use float64 for frequency
#'   computation. Default: TRUE
#' @param rope_type Character. Type of RoPE: "interleaved" or "split". Default: "interleaved"
#' @param num_attention_heads Integer. Number of attention heads (for split RoPE). Default: 32
#'
#' @return A list containing RoPE configuration and methods.
#' @export
rope_embedder_create <- function(
    dim,
    patch_size = 1L,
    patch_size_t = 1L,
    base_num_frames = 20L,
    base_height = 2048L,
    base_width = 2048L,
    scale_factors = c(8, 32, 32),
    theta = 10000.0,
    causal_offset = 1L,
    double_precision = TRUE,
    rope_type = c("interleaved", "split"),
    num_attention_heads = 32L
) {
  rope_type <- match.arg(rope_type)

  list(
    dim = dim,
    patch_size = patch_size,
    patch_size_t = patch_size_t,
    base_num_frames = base_num_frames,
    base_height = base_height,
    base_width = base_width,
    scale_factors = scale_factors,
    theta = theta,
    causal_offset = causal_offset,
    double_precision = double_precision,
    rope_type = rope_type,
    num_attention_heads = num_attention_heads
  )
}

#' Prepare video coordinates for RoPE
#'
#' Creates per-dimension patch boundaries for video coordinates in pixel space.
#' Returns tensor of shape (batch_size, 3, num_patches, 2) where dimension 1
#' represents (frame, height, width) and dimension 3 represents (start, end).
#'
#' @param embedder List. RoPE embedder configuration.
#' @param batch_size Integer. Batch size.
#' @param num_frames Integer. Number of latent frames.
#' @param height Integer. Latent height.
#' @param width Integer. Latent width.
#' @param device Character or torch device. Device for tensors.
#' @param fps Numeric. Video frames per second. Default: 24.0
#'
#' @return torch tensor of shape (batch_size, 3, num_patches, 2).
#' @export
rope_prepare_video_coords <- function(
    embedder,
    batch_size,
    num_frames,
    height,
    width,
    device = "cpu",
    fps = 24.0
) {
  patch_size <- embedder$patch_size
  patch_size_t <- embedder$patch_size_t
  scale_factors <- embedder$scale_factors
  causal_offset <- embedder$causal_offset

  # 1. Generate grid coordinates for each spatiotemporal dimension
  # Note: R's torch_arange is inclusive, so use end-step to match Python's exclusive behavior
  grid_f <- torch::torch_arange(
    start = 0, end = num_frames - patch_size_t, step = patch_size_t,
    dtype = torch::torch_float32(), device = device
  )
  grid_h <- torch::torch_arange(
    start = 0, end = height - patch_size, step = patch_size,
    dtype = torch::torch_float32(), device = device
  )
  grid_w <- torch::torch_arange(
    start = 0, end = width - patch_size, step = patch_size,
    dtype = torch::torch_float32(), device = device
  )

  # Create meshgrid (ij indexing keeps order as frames, height, width)
  grid <- torch::torch_meshgrid(list(grid_f, grid_h, grid_w), indexing = "ij")
  grid <- torch::torch_stack(grid, dim = 1)  # [3, N_F, N_H, N_W]

  # 2. Get patch boundaries with respect to latent grid
  patch_size_vec <- c(patch_size_t, patch_size, patch_size)
  patch_size_delta <- torch::torch_tensor(
    patch_size_vec, dtype = torch::torch_float32(), device = device
  )$view(c(3, 1, 1, 1))

  patch_ends <- grid + patch_size_delta

  # Combine start and end coordinates [3, N_F, N_H, N_W, 2]
  latent_coords <- torch::torch_stack(list(grid, patch_ends), dim = -1)

  # Reshape to (1, 3, num_patches, 2)
  latent_coords <- torch::torch_flatten(latent_coords, start_dim = 2, end_dim = 4)  # [3, num_patches, 2]
  latent_coords <- latent_coords$unsqueeze(1)   # [1, 3, num_patches, 2]
  latent_coords <- latent_coords$`repeat`(c(batch_size, 1, 1, 1))

  # 3. Convert to pixel space using VAE scale factors
  scale_tensor <- torch::torch_tensor(
    scale_factors, dtype = torch::torch_float32(), device = device
  )$view(c(1, 3, 1, 1))

  pixel_coords <- latent_coords * scale_tensor

  # Handle causal offset for first frame
  # Frame coordinates need special handling for causal VAE
  frame_coords <- pixel_coords[, 1, , ]
  frame_coords <- (frame_coords + causal_offset - scale_factors[1])$clamp(min = 0)
  pixel_coords[, 1, , ] <- frame_coords

  # Scale temporal coordinates by FPS (convert to seconds)
  pixel_coords[, 1, , ] <- pixel_coords[, 1, , ] / fps

  pixel_coords
}

#' Compute RoPE frequencies from coordinates
#'
#' Converts spatiotemporal coordinates to (cos, sin) frequency tensors
#' for applying rotary embeddings.
#'
#' @param embedder List. RoPE embedder configuration.
#' @param coords torch tensor. Coordinate tensor from rope_prepare_video_coords().
#' @param device Character or torch device. Device for output tensors.
#'
#' @return A list with:
#'   \describe{
#'     \item{cos_freqs}{Cosine frequencies tensor}
#'     \item{sin_freqs}{Sine frequencies tensor}
#'   }
#' @export
rope_forward <- function(embedder, coords, device = NULL) {
  if (is.null(device)) {
    device <- coords$device
  }

  dim <- embedder$dim
  theta <- embedder$theta
  base_num_frames <- embedder$base_num_frames
  base_height <- embedder$base_height
  base_width <- embedder$base_width
  double_precision <- embedder$double_precision
  rope_type <- embedder$rope_type
  num_attention_heads <- embedder$num_attention_heads

  # Number of spatiotemporal dimensions (3 for video)
  num_pos_dims <- as.numeric(coords$shape[2])

  # 1. If coords are patch boundaries [start, end), use midpoint
  if (length(coords$shape) == 4) {
    # Split into start and end
    coords_split <- coords$chunk(2, dim = -1)
    coords_start <- coords_split[[1]]
    coords_end <- coords_split[[2]]
    coords <- (coords_start + coords_end) / 2.0
    coords <- coords$squeeze(-1)  # [B, num_pos_dims, num_patches]
  }

  # 2. Get coordinates as fraction of base shape
  max_positions <- c(base_num_frames, base_height, base_width)

  # Create grid tensor [B, num_patches, num_pos_dims]
  grid_list <- list()
  for (i in seq_len(num_pos_dims)) {
    grid_list[[i]] <- coords[, i, ] / max_positions[i]
  }
  grid <- torch::torch_stack(grid_list, dim = -1)$to(device = device)

  # Number of RoPE elements (3 dims * 2 for cos/sin)
  num_rope_elems <- num_pos_dims * 2L

  # 3. Create 1D grid of frequencies
  freqs_dtype <- if (double_precision) torch::torch_float64() else torch::torch_float32()

  pow_indices <- torch::torch_pow(
    theta,
    torch::torch_linspace(
      start = 0.0, end = 1.0,
      steps = as.integer(dim %/% num_rope_elems),
      dtype = freqs_dtype, device = device
    )
  )
  freqs <- (pow_indices * pi / 2.0)$to(dtype = torch::torch_float32())

  # 4. Compute position-specific frequencies
  # [B, num_patches, num_pos_dims, dim // num_rope_elems]
  freqs <- (grid$unsqueeze(-1) * 2 - 1) * freqs

  # Transpose and flatten [B, num_patches, dim // 2]
  freqs <- freqs$transpose(-1, -2)$flatten(start_dim = 3)

  # 5. Get interleaved (cos, sin) frequencies
  if (rope_type == "interleaved") {
    cos_freqs <- freqs$cos()$repeat_interleave(2L, dim = -1L)
    sin_freqs <- freqs$sin()$repeat_interleave(2L, dim = -1L)

    # Handle padding if dim not divisible by num_rope_elems
    if (dim %% num_rope_elems != 0) {
      pad_size <- dim %% num_rope_elems
      cos_padding <- torch::torch_ones_like(cos_freqs[, , 1:pad_size])
      sin_padding <- torch::torch_zeros_like(sin_freqs[, , 1:pad_size])
      cos_freqs <- torch::torch_cat(list(cos_padding, cos_freqs), dim = -1)
      sin_freqs <- torch::torch_cat(list(sin_padding, sin_freqs), dim = -1)
    }
  } else if (rope_type == "split") {
    expected_freqs <- dim %/% 2L
    current_freqs <- as.numeric(freqs$shape[length(freqs$shape)])
    pad_size <- expected_freqs - current_freqs

    cos_freq <- freqs$cos()
    sin_freq <- freqs$sin()

    if (pad_size > 0) {
      cos_padding <- torch::torch_ones_like(cos_freq[, , 1:pad_size])
      sin_padding <- torch::torch_zeros_like(sin_freq[, , 1:pad_size])
      cos_freq <- torch::torch_cat(list(cos_padding, cos_freq), dim = -1)
      sin_freq <- torch::torch_cat(list(sin_padding, sin_freq), dim = -1)
    }

    # Reshape for multi-head attention
    b <- as.numeric(cos_freq$shape[1])
    t <- as.numeric(cos_freq$shape[2])

    cos_freq <- cos_freq$reshape(c(b, t, num_attention_heads, -1))
    sin_freq <- sin_freq$reshape(c(b, t, num_attention_heads, -1))

    cos_freqs <- cos_freq$swapaxes(2, 3)  # (B, H, T, D//2)
    sin_freqs <- sin_freq$swapaxes(2, 3)  # (B, H, T, D//2)
  }

  list(cos_freqs = cos_freqs, sin_freqs = sin_freqs)
}

#' Apply interleaved rotary embeddings
#'
#' Applies rotary position embeddings to query or key tensors using the
#' interleaved format where real and imaginary components alternate.
#'
#' @param x torch tensor. Query or key tensor of shape (B, S, C).
#' @param freqs List. Contains cos_freqs and sin_freqs from rope_forward().
#'
#' @return torch tensor. Rotated tensor with same shape as input.
#' @export
apply_interleaved_rotary_emb <- function(x, freqs) {
  cos_freqs <- freqs$cos_freqs
  sin_freqs <- freqs$sin_freqs

  # Split x into interleaved real/imaginary pairs
  # x has shape [B, S, C], unflatten to [B, S, C//2, 2]
  x_unflat <- x$unflatten(3, c(-1, 2))
  x_split <- x_unflat$unbind(-1)
  x_real <- x_split[[1]]  # [B, S, C // 2]
  x_imag <- x_split[[2]]  # [B, S, C // 2]

  # Rotate: stack [-x_imag, x_real] and flatten back
  x_rotated <- torch::torch_stack(list(-x_imag, x_real), dim = -1)$flatten(start_dim = 3)

  # Apply rotation formula: x * cos + rotate(x) * sin
  out <- (x$to(dtype = torch::torch_float32()) * cos_freqs +
            x_rotated$to(dtype = torch::torch_float32()) * sin_freqs)
  out <- out$to(dtype = x$dtype)

  out
}

#' Apply split rotary embeddings
#'
#' Applies rotary position embeddings to query or key tensors using the
#' split format where first half is real and second half is imaginary.
#'
#' @param x torch tensor. Query or key tensor.
#' @param freqs List. Contains cos_freqs and sin_freqs from rope_forward().
#'
#' @return torch tensor. Rotated tensor with same shape as input.
#' @export
apply_split_rotary_emb <- function(x, freqs) {
  cos_freqs <- freqs$cos_freqs
  sin_freqs <- freqs$sin_freqs

  x_dtype <- x$dtype
  needs_reshape <- FALSE

  # Handle dimension mismatch
  if (length(x$shape) != 4 && length(cos_freqs$shape) == 4) {
    b <- as.numeric(x$shape[1])
    cos_shape <- as.numeric(cos_freqs$shape)
    h <- cos_shape[2]
    t <- cos_shape[3]
    x <- x$reshape(c(b, t, h, -1))$swapaxes(2, 3)
    needs_reshape <- TRUE
  }

  # Split into real and imaginary halves
  half_dim <- as.numeric(x$shape[length(x$shape)]) %/% 2L
  x_real <- x[, , , 1:half_dim]
  x_imag <- x[, , , (half_dim + 1):(half_dim * 2)]

  # Rotate: [-x_imag, x_real]
  x_rotated <- torch::torch_cat(list(-x_imag, x_real), dim = -1)

  # Apply rotation
  out <- (x$to(dtype = torch::torch_float32()) * cos_freqs +
            x_rotated$to(dtype = torch::torch_float32()) * sin_freqs)

  if (needs_reshape) {
    out <- out$swapaxes(2, 3)$flatten(start_dim = 2, end_dim = 3)
  }

  out$to(dtype = x_dtype)
}
