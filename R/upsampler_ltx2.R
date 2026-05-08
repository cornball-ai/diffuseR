#' LTX-2 Latent Upsampler
#'
#' Spatial 2x upsampling of video latents using Conv3d ResBlocks and
#' PixelShuffle.  Used between Stage 1 (half-resolution) and Stage 2
#' (full-resolution) in the two-stage distilled pipeline.
#'
#' @name upsampler_ltx2
NULL

# -- Low-level modules --------------------------------------------------------

#' 2D PixelShuffle (channel -> spatial)
#'
#' Rearranges channels into spatial dimensions:
#' \code{[B, C*r*r, H, W] -> [B, C, H*r, W*r]}
#'
#' @param upscale_factor Integer. Upscale factor (default 2).
#' @return An \code{nn_module}.
#' @keywords internal
pixel_shuffle_2d <- torch::nn_module(
    "pixel_shuffle_2d",
    initialize = function(upscale_factor = 2L) {
        self$r <- as.integer(upscale_factor)
    },
    forward = function(x) {
        # x: [B, C*r*r, H, W]
        r <- self$r
        dims <- x$shape
        b <- dims[1]; c_in <- dims[2]; h <- dims[3]; w <- dims[4]
        c_out <- c_in %/% (r * r)

        # Reshape: [B, C, r, r, H, W] -> permute -> [B, C, H, r, W, r] -> reshape
        x <- x$view(c(b, c_out, r, r, h, w))
        x <- x$permute(c(1L, 2L, 5L, 3L, 6L, 4L))  # [B, C, H, r, W, r]
        x <- x$contiguous()$view(c(b, c_out, h * r, w * r))
        x
    }
)

#' Residual Block (Conv3d)
#'
#' Two Conv3d layers with GroupNorm and SiLU, plus skip connection.
#'
#' @param channels Integer. Input/output channels.
#' @param mid_channels Integer or NULL. Mid channels (default: same as channels).
#' @return An \code{nn_module}.
#' @keywords internal
upsampler_res_block <- torch::nn_module(
    "upsampler_res_block",
    initialize = function(channels, mid_channels = NULL) {
        if (is.null(mid_channels)) mid_channels <- channels
        self$conv1 <- torch::nn_conv3d(channels, mid_channels,
                                        kernel_size = 3L, padding = 1L)
        self$norm1 <- torch::nn_group_norm(32L, mid_channels)
        self$conv2 <- torch::nn_conv3d(mid_channels, channels,
                                        kernel_size = 3L, padding = 1L)
        self$norm2 <- torch::nn_group_norm(32L, channels)
        self$activation <- torch::nn_silu()
    },
    forward = function(x) {
        residual <- x
        x <- self$conv1(x)
        x <- self$norm1(x)
        x <- self$activation(x)
        x <- self$conv2(x)
        x <- self$norm2(x)
        x <- self$activation(x + residual)
        x
    }
)

#' Spatial Rational Resampler
#'
#' Per-frame spatial upsampling: Conv2d -> PixelShuffle -> optional BlurDownsample.
#' For scale=2.0: num=2, den=1 (no blur downsampling needed).
#'
#' @param mid_channels Integer. Number of intermediate channels.
#' @param scale Numeric. Spatial scale factor (default 2.0).
#' @return An \code{nn_module}.
#' @keywords internal
spatial_rational_resampler <- torch::nn_module(
    "spatial_rational_resampler",
    initialize = function(mid_channels, scale = 2.0) {
        self$scale <- scale
        # Rational decomposition: scale = num/den
        mapping <- list("2" = c(2L, 1L), "4" = c(4L, 1L),
                        "1.5" = c(3L, 2L), "0.75" = c(3L, 4L))
        key <- as.character(scale)
        if (is.null(mapping[[key]])) {
            stop("Unsupported scale: ", scale)
        }
        self$num <- mapping[[key]][1]
        self$den <- mapping[[key]][2]

        # Conv2d: mid_channels -> (num^2 * mid_channels)
        out_ch <- as.integer(self$num^2 * mid_channels)
        self$conv <- torch::nn_conv2d(mid_channels, out_ch,
                                       kernel_size = 3L, padding = 1L)
        self$pixel_shuffle <- pixel_shuffle_2d(self$num)

        # BlurDownsample (only active when den > 1)
        if (self$den > 1L) {
            self$blur_down <- blur_downsample_2d(stride = self$den)
        } else {
            self$blur_down <- NULL
        }
    },
    forward = function(x) {
        # x: [B, C, T, H, W] -> per-frame 2D
        dims <- x$shape
        b <- dims[1]; cc <- dims[2]; f <- dims[3]; h <- dims[4]; w <- dims[5]

        x <- x$permute(c(1L, 3L, 2L, 4L, 5L))       # [B, T, C, H, W]
        x <- x$reshape(c(b * f, cc, h, w))            # [B*T, C, H, W]
        x <- self$conv(x)
        x <- self$pixel_shuffle(x)
        if (!is.null(self$blur_down)) {
            x <- self$blur_down(x)
        }
        h2 <- x$shape[3]; w2 <- x$shape[4]
        x <- x$view(c(b, f, cc, h2, w2))
        x <- x$permute(c(1L, 3L, 2L, 4L, 5L))        # [B, C, T, H2, W2]
        x
    }
)

#' BlurDownsample (anti-aliased spatial downsampling)
#'
#' Fixed separable binomial kernel for anti-aliased downsampling.
#' With stride=1 this is the identity.
#'
#' @param stride Integer. Downsampling stride.
#' @param kernel_size Integer. Blur kernel size (default 5).
#' @return An \code{nn_module}.
#' @keywords internal
blur_downsample_2d <- torch::nn_module(
    "blur_downsample_2d",
    initialize = function(stride, kernel_size = 5L) {
        self$stride <- as.integer(stride)
        self$kernel_size <- as.integer(kernel_size)

        # Binomial kernel [1, 4, 6, 4, 1] for k=5
        k <- choose(kernel_size - 1L, seq(0L, kernel_size - 1L))
        k2d <- outer(k, k)
        k2d <- k2d / sum(k2d)
        kernel <- torch::torch_tensor(k2d, dtype = torch::torch_float32())
        self$kernel <- torch::nn_buffer(kernel$unsqueeze(1L)$unsqueeze(1L))
    },
    forward = function(x) {
        if (self$stride == 1L) return(x)

        cc <- x$shape[2]
        weight <- self$kernel$expand(c(cc, 1L, self$kernel_size,
                                        self$kernel_size))
        torch::nnf_conv2d(x, weight = weight, bias = NULL,
                           stride = self$stride,
                           padding = self$kernel_size %/% 2L,
                           groups = cc)
    }
)

# -- Main upsampler module ----------------------------------------------------

#' Latent Upsampler
#'
#' Full model: Conv3d initial -> GroupNorm -> SiLU -> 4x ResBlock -> SpatialRationalResampler -> 4x ResBlock -> Conv3d final.
#'
#' @param in_channels Integer. Input/output latent channels (default 128).
#' @param mid_channels Integer. Intermediate channels (default 1024).
#' @param num_blocks_per_stage Integer. ResBlocks per stage (default 4).
#' @param spatial_scale Numeric. Upscale factor (default 2.0).
#' @return An \code{nn_module}.
#' @keywords internal
latent_upsampler <- torch::nn_module(
    "latent_upsampler",
    initialize = function(in_channels = 128L,
                          mid_channels = 1024L,
                          num_blocks_per_stage = 4L,
                          spatial_scale = 2.0) {
        self$in_channels <- as.integer(in_channels)
        self$mid_channels <- as.integer(mid_channels)

        self$initial_conv <- torch::nn_conv3d(in_channels, mid_channels,
                                               kernel_size = 3L, padding = 1L)
        self$initial_norm <- torch::nn_group_norm(32L, mid_channels)
        self$initial_activation <- torch::nn_silu()

        self$res_blocks <- torch::nn_module_list(lapply(
            seq_len(num_blocks_per_stage),
            function(i) upsampler_res_block(mid_channels)
        ))

        self$upsampler <- spatial_rational_resampler(mid_channels,
                                                      scale = spatial_scale)

        self$post_upsample_res_blocks <- torch::nn_module_list(lapply(
            seq_len(num_blocks_per_stage),
            function(i) upsampler_res_block(mid_channels)
        ))

        self$final_conv <- torch::nn_conv3d(mid_channels, in_channels,
                                              kernel_size = 3L, padding = 1L)
    },
    forward = function(x) {
        # x: [B, C, T, H, W]
        x <- self$initial_conv(x)
        x <- self$initial_norm(x)
        x <- self$initial_activation(x)

        for (i in seq_along(self$res_blocks)) {
            x <- self$res_blocks[[i]](x)
        }

        x <- self$upsampler(x)

        for (i in seq_along(self$post_upsample_res_blocks)) {
            x <- self$post_upsample_res_blocks[[i]](x)
        }

        x <- self$final_conv(x)
        x
    }
)

# -- Weight loading -----------------------------------------------------------

#' Load LTX-2 Spatial Upsampler
#'
#' Loads the latent upsampler model from a safetensors file.
#'
#' @param weights_path Character. Path to safetensors weight file.
#' @param device Character. Target device ("cpu" or "cuda").
#' @param dtype Character. Target dtype ("float32", "float16", or "bfloat16").
#' @param verbose Logical. Print progress.
#' @return A \code{latent_upsampler} nn_module with loaded weights.
#'
#' @export
load_ltx2_upsampler <- function(weights_path,
                                 device = "cpu",
                                 dtype = "float32",
                                 verbose = TRUE) {
    if (!file.exists(weights_path)) {
        stop("Upsampler weights not found: ", weights_path)
    }

    if (verbose) message("Loading upsampler from: ", weights_path)

    # Create model
    model <- latent_upsampler(in_channels = 128L,
                               mid_channels = 1024L,
                               num_blocks_per_stage = 4L,
                               spatial_scale = 2.0)

    # Load weights
    weights <- safetensors::safe_load_file(weights_path, framework = "torch")

    # Map weight keys to model parameter names
    # safetensors keys use "." separators; R torch uses "$" but state_dict
    # uses "." too. The key names match 1:1 between Python and our R module.
    model_state <- model$state_dict()
    loaded <- 0L

    for (key in names(weights)) {
        # Map the Python key to R module key
        r_key <- .map_upsampler_key(key)
        if (r_key %in% names(model_state)) {
            model_state[[r_key]] <- weights[[key]]
            loaded <- loaded + 1L
        } else {
            if (verbose) message("  Skipping unmapped key: ", key,
                                 " -> ", r_key)
        }
    }

    model$load_state_dict(model_state)

    if (verbose) {
        message(sprintf("  Loaded %d/%d parameters", loaded,
                        length(names(weights))))
    }

    # Move to target device/dtype
    torch_dtype <- switch(dtype,
        "float16" = torch::torch_float16(),
        "bfloat16" = torch::torch_bfloat16(),
        torch::torch_float32()
    )
    model <- model$to(device = device, dtype = torch_dtype)

    model$eval()
    model
}

#' Map upsampler safetensors key to R module key
#' @keywords internal
.map_upsampler_key <- function(key) {
    # Direct mapping: Python and R module structures match exactly
    # Python: upsampler.conv.weight -> upsampler.conv.weight (in our module)
    # The blur_down.kernel is a buffer, maps to upsampler.blur_down.kernel
    key
}

# -- Upsample function --------------------------------------------------------

#' Upsample Video Latents
#'
#' Un-normalizes latents using VAE per-channel statistics, runs through
#' the upsampler, then re-normalizes.
#'
#' @param latents Tensor. Latent tensor \code{[B, C, T, H, W]}.
#' @param upsampler A \code{latent_upsampler} module.
#' @param latents_mean Tensor. Per-channel mean (from VAE).
#' @param latents_std Tensor. Per-channel std (from VAE).
#' @param device Character. Device for computation.
#' @param dtype Torch dtype for computation.
#' @return Upsampled latent tensor \code{[B, C, T, 2H, 2W]}.
#'
#' @keywords internal
upsample_video_latents <- function(latents, upsampler,
                                    latents_mean, latents_std,
                                    device = NULL, dtype = NULL) {
    # Un-normalize: x * std + mean
    lat_mean <- latents_mean$view(c(1L, -1L, 1L, 1L, 1L))$to(
        device = latents$device, dtype = latents$dtype)
    lat_std <- latents_std$view(c(1L, -1L, 1L, 1L, 1L))$to(
        device = latents$device, dtype = latents$dtype)
    latents <- latents * lat_std + lat_mean

    # Move to upsampler device if needed
    if (!is.null(device)) {
        latents <- latents$to(device = device)
    }
    if (!is.null(dtype)) {
        latents <- latents$to(dtype = dtype)
    }

    # Forward pass
    latents <- upsampler(latents)

    # Re-normalize: (x - mean) / std
    lat_mean <- lat_mean$to(device = latents$device, dtype = latents$dtype)
    lat_std <- lat_std$to(device = latents$device, dtype = latents$dtype)
    latents <- (latents - lat_mean) / lat_std

    latents
}
