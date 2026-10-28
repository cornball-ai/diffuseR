#' VAE Down Block
#'
#' diffusers DownEncoderBlock2D: resnets, then an optional stride-2
#' downsample conv. The downsample pads (0, 1, 0, 1) and convolves with
#' padding 0 (Downsample2D with \code{downsample_padding = 0}).
#'
#' @param in_channels Input channels
#' @param out_channels Output channels
#' @param num_resnets Number of resnet blocks
#' @param add_downsample Whether to add the downsampler
#' @param norm_groups Group norm groups
#' @keywords internal
VAEDownBlock <- torch::nn_module(
                                 "VAEDownBlock",

                                 initialize = function(in_channels, out_channels, num_resnets = 2,
        add_downsample = TRUE, norm_groups = 32) {
    self$resnets <- torch::nn_module_list()
    for (i in seq_len(num_resnets)) {
        res_in <- if (i == 1) in_channels else out_channels
        self$resnets$append(VAEResnetBlock(res_in, out_channels, norm_groups))
    }
    if (add_downsample) {
        self$downsamplers <- torch::nn_module_list(list(
                torch::nn_module(
                                 "Downsampler",
                                 initialize = function(channels) {
            self$conv <- torch::nn_conv2d(channels, channels,
                kernel_size = 3, stride = 2, padding = 0)
        },
                                 forward = function(x) {
            self$conv(torch::nnf_pad(x, c(0, 1, 0, 1)))
        }
                )(out_channels)
            ))
    }
},

                                 forward = function(x) {
    # Minor gc per resnet, as in VAEUpBlock
    block_gc <- isTRUE(getOption("diffuseR.block_gc"))
    for (i in seq_along(self$resnets)) {
        x <- self$resnets[[i]](x)
        if (block_gc) {
            gc(verbose = FALSE, full = FALSE)
        }
    }
    if (!is.null(self$downsamplers)) {
        x <- self$downsamplers[[1]](x)
    }
    x
}
)

#' FLUX.2 VAE encoder
#'
#' The AutoencoderKLFlux2 encode path: the standard diffusers AutoencoderKL
#' encoder body (conv_in, down blocks, attention mid block, GroupNorm +
#' SiLU, conv_out to 2 x latent channels) followed by quant_conv, plus the
#' BatchNorm running statistics used to normalize the patchified latent.
#' The forward returns the posterior mean (the reference pipeline encodes
#' with \code{sample_mode = "argmax"}). Reference:
#' src/diffusers/models/autoencoders/vae.py (Encoder) and
#' autoencoder_kl_flux2.py (Apache-2.0).
#'
#' @param latent_channels Integer (32 for FLUX.2).
#' @param block_channels Encoder block_out_channels.
#' @param layers_per_block Integer. Resnets per down block.
#' @param norm_groups Integer. Group norm groups.
#'
#' @return Module whose forward(x) encodes [B, 3, H, W] images in
#'   [-1, 1] to [B, 32, H / 8, W / 8] latent means.
#'
#' @export
flux2_vae_encoder <- torch::nn_module(
                                      "flux2_vae_encoder",
                                      initialize = function(latent_channels = 32L,
        block_channels = c(128L, 256L, 512L, 512L),
        layers_per_block = 2L, norm_groups = 32L) {
    body <- torch::nn_module(
                             "flux2_vae_encoder_body",
                             initialize = function() {
        n <- length(block_channels)
        self$conv_in <- torch::nn_conv2d(3L, block_channels[1],
            kernel_size = 3, padding = 1)
        self$down_blocks <- torch::nn_module_list()
        for (i in seq_len(n)) {
            in_ch <- block_channels[max(i - 1L, 1L)]
            self$down_blocks$append(VAEDownBlock(in_ch, block_channels[i],
                    num_resnets = layers_per_block,
                    add_downsample = i < n, norm_groups = norm_groups))
        }
        self$mid_block <- VAEMidBlock(block_channels[n], norm_groups)
        self$conv_norm_out <- torch::nn_group_norm(norm_groups,
            block_channels[n], eps = 1e-6)
        self$conv_out <- torch::nn_conv2d(block_channels[n],
            2L * latent_channels, kernel_size = 3, padding = 1)
    },
                             forward = function(x) {
        x <- self$conv_in(x)
        for (i in seq_along(self$down_blocks)) {
            x <- self$down_blocks[[i]](x)
        }
        x <- self$mid_block(x)
        x <- torch::nnf_silu(self$conv_norm_out(x))
        self$conv_out(x)
    }
    )
    self$encoder <- body()
    self$quant_conv <- torch::nn_conv2d(2L * latent_channels,
                                        2L * latent_channels, kernel_size = 1)
    bn_stats <- torch::nn_module(
                                 "flux2_bn_stats",
                                 initialize = function(n) {
        self$running_mean <- torch::nn_buffer(torch::torch_zeros(n))
        self$running_var <- torch::nn_buffer(torch::torch_ones(n))
    },
                                 forward = function(x) x
    )
    self$bn <- bn_stats(latent_channels * 4L)
    self$latent_channels <- latent_channels
},
                                      forward = function(x) {
    moments <- self$quant_conv(self$encoder(x))
    # DiagonalGaussianDistribution: mean is the first half of the channels
    moments$narrow(2L, 1L, self$latent_channels)
}
)

#' Load the FLUX.2 VAE encoder from safetensors
#'
#' Loads the encoder half plus quant_conv and the BatchNorm running
#' statistics; decoder and post_quant_conv keys are skipped.
#'
#' @param path Path to the VAE .safetensors file (or a directory
#'   containing diffusion_pytorch_model.safetensors).
#' @param latent_channels,block_channels,layers_per_block,norm_groups
#'   Constructor arguments for \code{\link{flux2_vae_encoder}}.
#' @param verbose Logical.
#'
#' @return The loaded \code{flux2_vae_encoder} in eval mode.
#'
#' @export
load_flux2_vae_encoder <- function(path, latent_channels = 32L,
                                   block_channels = c(128L, 256L, 512L, 512L),
                                   layers_per_block = 2L, norm_groups = 32L,
                                   verbose = TRUE) {
    path <- path.expand(path)
    if (dir.exists(path)) {
        path <- file.path(path, "diffusion_pytorch_model.safetensors")
    }
    enc <- flux2_vae_encoder(latent_channels = latent_channels,
                             block_channels = block_channels,
                             layers_per_block = layers_per_block,
                             norm_groups = norm_groups)

    handle <- safetensors::safetensors$new(path, framework = "torch")
    keys <- setdiff(handle$keys(), "__metadata__")
    keep <- keys[startsWith(keys, "encoder.") |
        startsWith(keys, "quant_conv.") |
        keys %in% c("bn.running_mean", "bn.running_var")]

    dests <- c(enc$named_parameters(), enc$named_buffers())
    filled <- character(0)
    unmapped <- character(0)
    torch::with_no_grad({
        for (key in keep) {
            dest <- dests[[key]]
            if (is.null(dest)) {
                unmapped <- c(unmapped, key)
                next
            }
            dest$copy_(handle$get_tensor(key))
            filled <- c(filled, key)
        }
    })
    unfilled <- setdiff(names(dests), filled)
    if (length(unmapped)) {
        stop("FLUX.2 VAE encoder load: ", length(unmapped),
             " unmapped keys, e.g. ",
             paste(utils::head(unmapped, 3), collapse = ", "))
    }
    if (length(unfilled)) {
        stop("FLUX.2 VAE encoder load: ", length(unfilled),
             " unfilled params, e.g. ",
             paste(utils::head(unfilled, 3), collapse = ", "))
    }
    if (verbose) {
        message("Loaded ", length(filled), " FLUX.2 VAE encoder tensors from ",
                path)
    }
    enc$eval()
    enc
}

#' Encode a reference image into FLUX.2 tokens
#'
#' VAE-encode (posterior mean), patchify 2x2, normalize with the VAE
#' BatchNorm statistics and pack to channels-last tokens. Reference:
#' Flux2KleinPipeline._encode_vae_image and prepare_image_latents.
#'
#' @param encoder A loaded \code{\link{flux2_vae_encoder}}.
#' @param image Tensor [1, 3, H, W] in [-1, 1]; H and W multiples of 16.
#' @param bn_eps Numeric. BatchNorm epsilon.
#'
#' @return \code{list(tokens, height, width)}: tokens [1, H/16 * W/16, 128]
#'   on the encoder's device and dtype, and the packed grid size.
#'
#' @export
flux2_encode_image <- function(encoder, image, bn_eps = 1e-4) {
    param <- encoder$encoder$conv_in$weight
    torch::with_no_grad({
        latents <- encoder(image$to(device = param$device, dtype = param$dtype))
        latents <- flux2_patchify_latents(latents)
        latents <- flux2_bn_normalize(latents, encoder$bn$running_mean,
                                      encoder$bn$running_var, eps = bn_eps)
        h <- latents$shape[3]
        w <- latents$shape[4]
        tokens <- flux2_pack_latents(latents)
    })
    list(tokens = tokens, height = h, width = w)
}
