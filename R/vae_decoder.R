#' VAE ResNet Block
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @keywords internal
VAEResnetBlock <- torch::nn_module(
                                   "VAEResnetBlock",

                                   initialize = function(
        in_channels,
        out_channels,
        norm_groups = 32
    ) {
    self$norm1 <- torch::nn_group_norm(norm_groups, in_channels, eps = 1e-6)
    self$conv1 <- torch::nn_conv2d(in_channels, out_channels,
                                   kernel_size = 3, padding = 1)
    self$norm2 <- torch::nn_group_norm(norm_groups, out_channels, eps = 1e-6)
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

                                      initialize = function(channels, norm_groups = 32) {
    self$group_norm <- torch::nn_group_norm(norm_groups, channels, eps = 1e-6)
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
    x <- x$permute(c(1, 3, 4, 2))$reshape(c(batch, height * width, channels))

    # QKV projections
    q <- self$to_q(x)
    k <- self$to_k(x)
    v <- self$to_v(x)

    # Scaled dot-product attention
    scale <- 1.0 / sqrt(channels)
    attn <- torch::torch_bmm(q, k$transpose(2, 3)) * scale
    attn <- torch::nnf_softmax(attn, dim = -1)
    out <- torch::torch_bmm(attn, v)

    # Project out
    out <- self$to_out[[1]](out)

    # Reshape back to (batch, channels, h, w)
    out <- out$reshape(c(batch, height, width, channels))$permute(c(1, 4, 2, 3))

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
        add_upsample = TRUE,
        norm_groups = 32
    ) {
    self$resnets <- torch::nn_module_list()

    for (i in seq_len(num_resnets)) {
        if (i == 1) {
            res_in <- in_channels
        } else {
            res_in <- out_channels
        }
        self$resnets$append(VAEResnetBlock(res_in, out_channels, norm_groups))
    }

    if (add_upsample) {
        self$upsamplers <- torch::nn_module_list(list(
                torch::nn_module(
                                 "Upsampler",
                                 initialize = function(channels) {
            self$conv <- torch::nn_conv2d(channels, channels,
                kernel_size = 3, padding = 1)
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

                                initialize = function(channels, norm_groups = 32) {
    self$resnets <- torch::nn_module_list(list(
            VAEResnetBlock(channels, channels, norm_groups),
            VAEResnetBlock(channels, channels, norm_groups)
        ))
    self$attentions <- torch::nn_module_list(list(VAEAttentionBlock(channels,
                norm_groups)))
},

                                forward = function(x) {
    x <- self$resnets[[1]](x)
    x <- self$attentions[[1]](x)
    x <- self$resnets[[2]](x)
    x
}
)

#' Load HF safetensors VAE weights into the native decoder
#'
#' Loads the decoder half of a diffusers AutoencoderKL safetensors file
#' (e.g. FLUX.1-schnell's \code{vae/diffusion_pytorch_model.safetensors}).
#' Keys under \code{decoder.} map to the native module 1:1; encoder and
#' quant-conv keys are skipped (the FLUX VAE has no quant convs, and
#' txt2img needs no encoder).
#'
#' @param native_decoder Native VAE decoder module
#' @param path Path to the VAE .safetensors file (or a directory
#'   containing diffusion_pytorch_model.safetensors)
#' @param verbose Print loading progress
#'
#' @return The native decoder with loaded weights (invisibly)
#' @export
load_decoder_safetensors <- function(native_decoder, path, verbose = TRUE) {
    path <- path.expand(path)
    if (dir.exists(path)) {
        path <- file.path(path, "diffusion_pytorch_model.safetensors")
    }
    handle <- safetensors::safetensors$new(path, framework = "torch")
    keys <- setdiff(handle$keys(), "__metadata__")
    dec_keys <- keys[startsWith(keys, "decoder.")]

    dests <- native_decoder$named_parameters()
    filled <- character(0)
    unmapped <- character(0)
    torch::with_no_grad({
        for (key in dec_keys) {
            native_name <- sub("^decoder\\.", "", key)
            dest <- dests[[native_name]]
            if (is.null(dest)) {
                unmapped <- c(unmapped, key)
                next
            }
            dest$copy_(handle$get_tensor(key))
            filled <- c(filled, native_name)
        }
    })

    unfilled <- setdiff(names(dests), filled)
    if (length(unmapped)) {
        stop("VAE decoder load: ", length(unmapped), " unmapped keys, e.g. ",
             paste(utils::head(unmapped, 3), collapse = ", "))
    }
    if (length(unfilled)) {
        stop("VAE decoder load: ", length(unfilled),
             " unfilled params, e.g. ",
             paste(utils::head(unfilled, 3), collapse = ", "))
    }
    if (verbose) {
        message("Loaded ", length(filled), " decoder parameters from ", path)
    }
    invisible(native_decoder)
}

#' Build a native VAE decoder from a diffusers safetensors directory
#'
#' The safetensors counterpart to the TorchScript decoder path:
#' constructs \code{\link{vae_decoder_native}} and loads the decoder half
#' of a diffusers AutoencoderKL checkpoint (no TorchScript, so it works
#' on Blackwell). \code{latent_channels} defaults to 4 (SD/SDXL); pass 16
#' for the FLUX/SD3 VAE. The SD/SDXL and FLUX VAEs share the decoder
#' shape and differ only in that channel count.
#'
#' @param path Path to the VAE directory (containing
#'   \code{diffusion_pytorch_model.safetensors}) or the file itself.
#' @param latent_channels Latent channel count (4 for SD/SDXL, 16 for FLUX).
#' @param verbose Print how many parameters were loaded.
#' @param ... Overrides for \code{\link{vae_decoder_native}} constructor args.
#'
#' @return The native VAE decoder in eval mode.
#' @export
vae_decoder_native_from_safetensors <- function(path, latent_channels = 4L,
                                                verbose = TRUE, ...) {
    model <- vae_decoder_native(latent_channels = latent_channels, ...)
    load_decoder_safetensors(model, path, verbose = verbose)
    model$eval()
    model
}

#' Load weights from TorchScript decoder into native decoder
#'
#' @param native_decoder Native VAE decoder module
#' @param torchscript_path Path to TorchScript decoder .pt file
#' @param verbose Print loading progress
#'
#' @return The native decoder with loaded weights (invisibly)
#' @export
load_decoder_weights <- function(native_decoder, torchscript_path,
                                 verbose = TRUE) {
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
#' @param latent_channels Number of latent channels (4 for SD/SDXL,
#'   16 for FLUX/SD3)
#' @param out_channels Number of output channels (default 3 for RGB)
#' @param block_channels Decoder block channels (reversed encoder
#'   block_out_channels; default matches SD/SDXL and FLUX)
#' @param norm_groups Group norm groups (default 32; must divide every
#'   entry of \code{block_channels})
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
        out_channels = 3,
        block_channels = c(512, 512, 256, 128),
        norm_groups = 32
    ) {
    # Diffusers AutoencoderKL decoder: block channels reversed from the
    # encoder's block_out_channels; upsamplers on all but the last block.
    # The SD/SDXL and FLUX/SD3 VAEs share this exact shape (FLUX differs
    # only in latent_channels = 16).
    n_blocks <- length(block_channels)

    self$conv_in <- torch::nn_conv2d(latent_channels, block_channels[1],
                                     kernel_size = 3, padding = 1)

    self$mid_block <- VAEMidBlock(block_channels[1], norm_groups)

    self$up_blocks <- torch::nn_module_list()
    for (i in seq_len(n_blocks)) {
        in_ch <- block_channels[max(i - 1, 1)]
        self$up_blocks$append(VAEUpBlock(in_ch, block_channels[i],
                num_resnets = 3,
                add_upsample = i < n_blocks,
                norm_groups = norm_groups))
    }

    last <- block_channels[n_blocks]
    self$conv_norm_out <- torch::nn_group_norm(norm_groups, last, eps = 1e-6)
    self$conv_out <- torch::nn_conv2d(last, out_channels, kernel_size = 3, padding = 1)
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
