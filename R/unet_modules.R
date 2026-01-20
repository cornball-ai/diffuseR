#' UNet Modules for Diffusion Models
#'
#' Native R torch modules for UNet architecture.
#' @keywords internal
#' @name unet_modules
NULL

#' Group Normalization (32 groups)
#' @keywords internal
group_norm_32 <- function(channels) {
  torch::nn_group_norm(32, channels, eps = 1e-5)
}

#' Sinusoidal Timestep Embedding
#'
#' @param timesteps Tensor of timesteps (batch_size,)
#' @param dim Embedding dimension
#' @param flip_sin_to_cos If TRUE, output [cos, sin] instead of [sin, cos].
#'   SDXL uses TRUE (default), SD21 uses FALSE.
#' @param downscale_freq_shift Frequency shift parameter. SDXL uses 0 (default),
#'   SD21 uses 1. With 0: exponent = log(10000) / half_dim. With 1: exponent = log(10000) / (half_dim - 1).
#' @return Tensor (batch_size, dim)
#' @keywords internal
timestep_embedding <- function(
  timesteps,
  dim,
  flip_sin_to_cos = TRUE,
  downscale_freq_shift = 0L
) {
  half_dim <- dim %/% 2L

  # SDXL uses downscale_freq_shift=0, SD21 uses downscale_freq_shift=1
  emb_scale <- log(10000) / (half_dim - downscale_freq_shift)

  # Create frequency bands: [0, 1, ..., half_dim-1]
  freqs <- torch::torch_exp(torch::torch_arange(0, half_dim - 1L) * - emb_scale)
  freqs <- freqs$to(device = timesteps$device, dtype = torch::torch_float32())

  # Expand and compute embeddings
  # timesteps: [batch] -> [batch, 1]
  # freqs: [half_dim] -> [1, half_dim]
  timesteps_float <- timesteps$to(dtype = torch::torch_float32()) $unsqueeze(2L)
  args <- timesteps_float * freqs$unsqueeze(1L)

  # SDXL uses flip_sin_to_cos=True -> [cos, sin]
  # SD21 uses flip_sin_to_cos=False -> [sin, cos]
  if (flip_sin_to_cos) {
    embedding <- torch::torch_cat(list(torch::torch_cos(args), torch::torch_sin(args)), dim = 2L)
  } else {
    embedding <- torch::torch_cat(list(torch::torch_sin(args), torch::torch_cos(args)), dim = 2L)
  }

  # Pad if odd dimension
  if (dim %% 2L == 1L) {
    embedding <- torch::nnf_pad(embedding, c(0L, 1L, 0L, 0L))
  }

  embedding
}

#' ResNet Block for UNet
#'
#' @keywords internal
UNetResBlock <- torch::nn_module(
  "UNetResBlock",

  initialize = function(
    in_channels,
    out_channels,
    time_embed_dim
  ) {
    self$in_channels <- in_channels
    self$out_channels <- out_channels

    # First conv block
    self$norm1 <- group_norm_32(in_channels)
    self$conv1 <- torch::nn_conv2d(in_channels, out_channels, 3L, padding = 1L)

    # Time embedding projection
    self$time_emb_proj <- torch::nn_linear(time_embed_dim, out_channels)

    # Second conv block
    self$norm2 <- group_norm_32(out_channels)
    self$conv2 <- torch::nn_conv2d(out_channels, out_channels, 3L, padding = 1L)

    # Skip connection (if channels differ)
    if (in_channels != out_channels) {
      self$conv_shortcut <- torch::nn_conv2d(in_channels, out_channels, 1L)
    } else {
      self$conv_shortcut <- NULL
    }
  },

  forward = function(
    x,
    temb
  ) {
    h <- x

    # First block
    h <- self$norm1(h)
    h <- torch::nnf_silu(h)
    h <- self$conv1(h)

    # Add time embedding
    temb_out <- torch::nnf_silu(temb)
    temb_out <- self$time_emb_proj(temb_out)
    # Expand to spatial dimensions: (B, C) -> (B, C, 1, 1)
    temb_out <- temb_out$unsqueeze(- 1L) $unsqueeze(- 1L)
    h <- h + temb_out

    # Second block
    h <- self$norm2(h)
    h <- torch::nnf_silu(h)
    h <- self$conv2(h)

    # Skip connection
    if (!is.null(self$conv_shortcut)) {
      x <- self$conv_shortcut(x)
    }

    h + x
  }
)

#' Downsample Block
#' @keywords internal
Downsample2D <- torch::nn_module(
  "Downsample2D",

  initialize = function(channels) {
    self$conv <- torch::nn_conv2d(channels, channels, 3L, stride = 2L, padding = 1L)
  },

  forward = function(x) {
    self$conv(x)
  }
)

#' Upsample Block
#' @keywords internal
Upsample2D <- torch::nn_module(
  "Upsample2D",

  initialize = function(channels) {
    self$conv <- torch::nn_conv2d(channels, channels, 3L, padding = 1L)
  },

  forward = function(x) {
    # Upsample 2x then conv
    x <- torch::nnf_interpolate(x, scale_factor = 2, mode = "nearest")
    self$conv(x)
  }
)

#' Cross-Attention for UNet
#' @keywords internal
UNetCrossAttention <- torch::nn_module(
  "UNetCrossAttention",

  initialize = function(
    query_dim,
    context_dim = NULL,
    heads = 8L,
    dim_head = 64L
  ) {
    self$heads <- heads
    self$dim_head <- dim_head
    inner_dim <- heads * dim_head
    context_dim <- context_dim %||% query_dim

    self$scale <- dim_head ^ (- 0.5)

    # Query, Key, Value projections
    self$to_q <- torch::nn_linear(query_dim, inner_dim, bias = FALSE)
    self$to_k <- torch::nn_linear(context_dim, inner_dim, bias = FALSE)
    self$to_v <- torch::nn_linear(context_dim, inner_dim, bias = FALSE)

    # Output projection
    self$to_out <- torch::nn_sequential(
      torch::nn_linear(inner_dim, query_dim),
      torch::nn_dropout(0)
    )
  },

  forward = function(
    x,
    context = NULL
  ) {
    if (is.null(context)) context <- x

    b <- x$shape[1]
    seq_len <- x$shape[2]
    context_len <- context$shape[2]
    h <- self$heads

    # Project
    q <- self$to_q(x)
    k <- self$to_k(context)
    v <- self$to_v(context)

    # Reshape for multi-head attention: (B, S, H*D) -> (B, H, S, D)
    q <- q$view(c(b, seq_len, h, self$dim_head)) $transpose(2L, 3L)
    k <- k$view(c(b, context_len, h, self$dim_head)) $transpose(2L, 3L)
    v <- v$view(c(b, context_len, h, self$dim_head)) $transpose(2L, 3L)

    # Scaled dot-product attention
    attn <- torch::torch_matmul(q, k$transpose(3L, 4L)) * self$scale
    attn <- torch::nnf_softmax(attn, dim = - 1L)

    # Apply attention to values
    out <- torch::torch_matmul(attn, v)

    # Reshape back: (B, H, S, D) -> (B, S, H*D)
    out <- out$transpose(2L, 3L) $contiguous() $view(c(b, seq_len, - 1L))

    self$to_out(out)
  }
)

#' GEGLU Feedforward
#' @keywords internal
GEGLU <- torch::nn_module(
  "GEGLU",

  initialize = function(
    dim_in,
    dim_out
  ) {
    self$proj <- torch::nn_linear(dim_in, dim_out * 2L)
  },

  forward = function(x) {
    x_proj <- self$proj(x)
    chunks <- torch::torch_chunk(x_proj, 2L, dim = - 1L)
    chunks[[1]] * torch::nnf_gelu(chunks[[2]])
  }
)

#' FeedForward Network
#' @keywords internal
FeedForward <- torch::nn_module(
  "FeedForward",

  initialize = function(
    dim,
    mult = 4L
  ) {
    inner_dim <- dim * mult
    self$net <- torch::nn_sequential(
      GEGLU(dim, inner_dim),
      torch::nn_dropout(0),
      torch::nn_linear(inner_dim, dim)
    )
  },

  forward = function(x) {
    self$net(x)
  }
)

#' Basic Transformer Block
#' @keywords internal
BasicTransformerBlock <- torch::nn_module(
  "BasicTransformerBlock",

  initialize = function(
    dim,
    n_heads,
    d_head,
    context_dim = NULL
  ) {
    # Self-attention
    self$attn1 <- UNetCrossAttention(dim, heads = n_heads, dim_head = d_head)

    # Cross-attention
    self$attn2 <- UNetCrossAttention(dim, context_dim = context_dim,
      heads = n_heads, dim_head = d_head)

    # Feedforward
    self$ff <- FeedForward(dim)

    # Layer norms
    self$norm1 <- torch::nn_layer_norm(dim)
    self$norm2 <- torch::nn_layer_norm(dim)
    self$norm3 <- torch::nn_layer_norm(dim)
  },

  forward = function(
    x,
    context = NULL
  ) {
    # Self-attention
    x <- x + self$attn1(self$norm1(x))

    # Cross-attention
    x <- x + self$attn2(self$norm2(x), context)

    # Feedforward
    x <- x + self$ff(self$norm3(x))

    x
  }
)

#' Spatial Transformer (Attention Block)
#' @keywords internal
SpatialTransformer <- torch::nn_module(
  "SpatialTransformer",

  initialize = function(
    in_channels,
    n_heads,
    d_head,
    depth = 1L,
    context_dim = NULL
  ) {
    inner_dim <- n_heads * d_head
    self$in_channels <- in_channels

    # Normalization
    self$norm <- group_norm_32(in_channels)

    # Projection in/out
    self$proj_in <- torch::nn_linear(in_channels, inner_dim)
    self$proj_out <- torch::nn_linear(inner_dim, in_channels)

    # Transformer blocks
    self$transformer_blocks <- torch::nn_module_list()
    for (i in seq_len(depth)) {
      self$transformer_blocks$append(
        BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim)
      )
    }
  },

  forward = function(
    x,
    context = NULL
  ) {
    b <- x$shape[1]
    c <- x$shape[2]
    h <- x$shape[3]
    w <- x$shape[4]

    x_in <- x

    # Normalize
    x <- self$norm(x)

    # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
    x <- x$permute(c(1L, 3L, 4L, 2L)) $reshape(c(b, h * w, c))

    # Project in
    x <- self$proj_in(x)

    # Transformer blocks
    for (i in seq_along(self$transformer_blocks)) {
      x <- self$transformer_blocks[[i]](x, context)
    }

    # Project out
    x <- self$proj_out(x)

    # Reshape back: (B, H*W, C) -> (B, C, H, W)
    x <- x$reshape(c(b, h, w, c)) $permute(c(1L, 4L, 2L, 3L))

    # Residual
    x + x_in
  }
)

