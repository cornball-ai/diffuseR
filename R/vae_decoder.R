#' VAE ResNet Block
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @keywords internal
VAEResnetBlock <- torch::nn_module(
  "VAEResnetBlock",

  initialize = function(
    in_channels,
    out_channels
  ) {
    self$norm1 <- torch::nn_group_norm(32, in_channels, eps = 1e-6)
    self$conv1 <- torch::nn_conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
    self$norm2 <- torch::nn_group_norm(32, out_channels, eps = 1e-6)
    self$conv2 <- torch::nn_conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

    # Shortcut if dimensions change
    if (in_channels != out_channels) {
      self$conv_shortcut <- torch::nn_conv2d(in_channels, out_channels, kernel_size = 1)
    }
  },

  forward = function(x) {
    h <- x
    h <- self$norm1(h)
    h <- torch::nnf_silu(h)
    h <- self$conv1(h)

    h <- self$norm2(h)
    h <- torch::nnf_silu(h)
    h <- self$conv2(h)

    # Apply shortcut if it exists
    if (!is.null(self$conv_shortcut)) {
      x <- self$conv_shortcut(x)
    }

    h + x
  }
)

#' VAE Attention Block
#'
#' Self-attention for VAE mid block
#' @param channels Number of channels
#' @keywords internal
VAEAttentionBlock <- torch::nn_module(
  "VAEAttentionBlock",

  initialize = function(channels) {
    self$group_norm <- torch::nn_group_norm(32, channels, eps = 1e-6)
    self$to_q <- torch::nn_linear(channels, channels)
    self$to_k <- torch::nn_linear(channels, channels)
    self$to_v <- torch::nn_linear(channels, channels)
    self$to_out <- torch::nn_module_list(list(
        torch::nn_linear(channels, channels)
      ))
    self$channels <- channels
  },

  forward = function(x) {
    residual <- x
    batch <- x$shape[1]
    channels <- x$shape[2]
    height <- x$shape[3]
    width <- x$shape[4]

    # Normalize
    x <- self$group_norm(x)

    # Reshape to (batch, h*w, channels)
    x <- x$permute(c(1, 3, 4, 2)) $reshape(c(batch, height * width, channels))

    # QKV projections
    q <- self$to_q(x)
    k <- self$to_k(x)
    v <- self$to_v(x)

    # Scaled dot-product attention
    scale <- 1.0 / sqrt(channels)
    attn <- torch::torch_bmm(q, k$transpose(2, 3)) * scale
    attn <- torch::nnf_softmax(attn, dim = - 1)
    out <- torch::torch_bmm(attn, v)

    # Project out
    out <- self$to_out[[1]](out)

    # Reshape back to (batch, channels, h, w)
    out <- out$reshape(c(batch, height, width, channels)) $permute(c(1, 4, 2, 3))

    out + residual
  }
)

#' VAE Up Block
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @param num_resnets Number of resnet blocks (default 3)
#' @param add_upsample Whether to add upsampler
#' @keywords internal
VAEUpBlock <- torch::nn_module(
  "VAEUpBlock",

  initialize = function(
    in_channels,
    out_channels,
    num_resnets = 3,
    add_upsample = TRUE
  ) {
    self$resnets <- torch::nn_module_list()

    for (i in seq_len(num_resnets)) {
      if (i == 1) {
        res_in <- in_channels
      } else {
        res_in <- out_channels
      }
      self$resnets$append(VAEResnetBlock(res_in, out_channels))
    }

    if (add_upsample) {
      self$upsamplers <- torch::nn_module_list(list(
          torch::nn_module(
            "Upsampler",
            initialize = function(channels) {
              self$conv <- torch::nn_conv2d(channels, channels, kernel_size = 3, padding = 1)
            },
            forward = function(x) {
              x <- torch::nnf_interpolate(x, scale_factor = 2, mode = "nearest")
              self$conv(x)
            }
          )(out_channels)
        ))
    }
  },

  forward = function(x) {
    for (i in seq_along(self$resnets)) {
      x <- self$resnets[[i]](x)
    }
    if (!is.null(self$upsamplers)) {
      x <- self$upsamplers[[1]](x)
    }
    x
  }
)

#' VAE Mid Block
#'
#' @param channels Number of channels
#' @keywords internal
VAEMidBlock <- torch::nn_module(
  "VAEMidBlock",

  initialize = function(channels) {
    self$resnets <- torch::nn_module_list(list(
        VAEResnetBlock(channels, channels),
        VAEResnetBlock(channels, channels)
      ))
    self$attentions <- torch::nn_module_list(list(
        VAEAttentionBlock(channels)
      ))
  },

  forward = function(x) {
    x <- self$resnets[[1]](x)
    x <- self$attentions[[1]](x)
    x <- self$resnets[[2]](x)
    x
  }
)

#' Load weights from TorchScript decoder into native decoder
#'
#' @param native_decoder Native VAE decoder module
#' @param torchscript_path Path to TorchScript decoder .pt file
#' @param verbose Print loading progress
#'
#' @return The native decoder with loaded weights (invisibly)
#' @export
load_decoder_weights <- function(
  native_decoder,
  torchscript_path,
  verbose = TRUE
) {
  ts_decoder <- torch::jit_load(torchscript_path)
  ts_params <- ts_decoder$parameters

  loaded <- 0
  torch::with_no_grad({
      for (ts_name in names(ts_params)) {
        # Strip dec. prefix
        native_name <- sub("^dec\\.", "", ts_name)

        if (native_name %in% names(native_decoder$parameters)) {
          ts_tensor <- ts_params[[ts_name]]
          native_tensor <- native_decoder$parameters[[native_name]]

          if (all(ts_tensor$shape == native_tensor$shape)) {
            native_tensor$copy_(ts_tensor)
            loaded <- loaded + 1
          } else if (verbose) {
            cat("Shape mismatch:", native_name, "\n")
          }
        } else if (verbose) {
          cat("Missing param:", native_name, "\n")
        }
      }
    })

  if (verbose) {
    cat("Loaded", loaded, "/", length(names(ts_params)), "parameters\n")
  }

  invisible(native_decoder)
}

#' Native VAE Decoder
#'
#' Native R torch implementation of the SDXL VAE decoder.
#' Replaces TorchScript decoder for better GPU compatibility.
#'
#' @param latent_channels Number of latent channels (default 4)
#' @param out_channels Number of output channels (default 3 for RGB)
#'
#' @return An nn_module representing the VAE decoder
#' @export
#'
#' @examples
#' \dontrun{
#' decoder <- vae_decoder_native()
#' load_decoder_weights(decoder, "path/to/decoder.pt")
#' latents <- torch::torch_randn(c(1, 4, 64, 64))
#' image <- decoder(latents)
#' }
vae_decoder_native <- torch::nn_module(
  "VAEDecoderNative",

  initialize = function(
    latent_channels = 4,
    out_channels = 3
  ) {
    # SDXL VAE decoder architecture:
    # Block channels: 512, 512, 256, 128 (reversed from encoder)
    block_channels <- c(512, 512, 256, 128)

    # Input conv: latent_channels -> 512
    self$conv_in <- torch::nn_conv2d(latent_channels, 512, kernel_size = 3, padding = 1)

    # Mid block
    self$mid_block <- VAEMidBlock(512)

    # Up blocks (4 blocks, 3 with upsamplers)
    self$up_blocks <- torch::nn_module_list()

    # up_block 0: 512 -> 512, has upsampler
    self$up_blocks$append(VAEUpBlock(512, 512, num_resnets = 3, add_upsample = TRUE))

    # up_block 1: 512 -> 512, has upsampler
    self$up_blocks$append(VAEUpBlock(512, 512, num_resnets = 3, add_upsample = TRUE))

    # up_block 2: 512 -> 256, has upsampler
    self$up_blocks$append(VAEUpBlock(512, 256, num_resnets = 3, add_upsample = TRUE))

    # up_block 3: 256 -> 128, NO upsampler (final block)
    self$up_blocks$append(VAEUpBlock(256, 128, num_resnets = 3, add_upsample = FALSE))

    # Output layers
    self$conv_norm_out <- torch::nn_group_norm(32, 128, eps = 1e-6)
    self$conv_out <- torch::nn_conv2d(128, out_channels, kernel_size = 3, padding = 1)
  },

  forward = function(x) {
    # Input conv
    x <- self$conv_in(x)

    # Mid block
    x <- self$mid_block(x)

    # Up blocks
    for (i in seq_along(self$up_blocks)) {
      x <- self$up_blocks[[i]](x)
    }

    # Output
    x <- self$conv_norm_out(x)
    x <- torch::nnf_silu(x)
    x <- self$conv_out(x)

    x
  }
)

