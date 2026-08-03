#' LTX-2.3 Spatial Latent Upsampler
#'
#' Fresh R port of the LTX latent upsampler from the diffusers reference
#' (Apache-2.0, pipelines/ltx2/latent_upsampler.py and
#' pipeline_ltx2_latent_upsample.py), with the LTX 2.3 configuration:
#' Conv3d ResBlock stages around a per-frame 2x pixel-shuffle spatial
#' upsampler (no rational resampler). Operates on unnormalized latents.
#'
#' @name upsampler_ltx23
NULL

# Conv -> GroupNorm(32) twice with the activation applied to the residual sum
ltx23_upsampler_res_block <- torch::nn_module(
    "ltx23_upsampler_res_block",
    initialize = function(channels, mid_channels = NULL) {
    mid_channels <- mid_channels %||% channels
    self$conv1 <- torch::nn_conv3d(channels, mid_channels, kernel_size = 3L,
                                   padding = 1L)
    self$norm1 <- torch::nn_group_norm(32L, mid_channels)
    self$conv2 <- torch::nn_conv3d(mid_channels, channels, kernel_size = 3L, padding = 1L)
    self$norm2 <- torch::nn_group_norm(32L, channels)
},
    forward = function(x) {
    residual <- x
    x <- torch::nnf_silu(self$norm1(self$conv1(x)))
    x <- self$norm2(self$conv2(x))
    torch::nnf_silu(x + residual)
}
)

#' LTX-2.3 latent upsampler model
#'
#' Latents [B, 128, F, H, W] -> [B, 128, F, 2H, 2W].
#'
#' @param in_channels Integer. Latent channels.
#' @param mid_channels Integer.
#' @param num_blocks_per_stage Integer.
#'
#' @return Module whose forward(hidden_states) returns the 2x
#'   spatially upscaled latent, a tensor with the same batch, channel
#'   and frame counts and doubled height and width.
#'
#' @export
ltx23_latent_upsampler <- torch::nn_module(
    "ltx23_latent_upsampler",
    initialize = function(
                          in_channels = 128L,
                          mid_channels = 1024L,
                          num_blocks_per_stage = 4L
    ) {
    self$initial_conv <- torch::nn_conv3d(in_channels, mid_channels,
        kernel_size = 3L, padding = 1L)
    self$initial_norm <- torch::nn_group_norm(32L, mid_channels)

    self$res_blocks <- torch::nn_module_list(lapply(seq_len(num_blocks_per_stage),
            function(i) ltx23_upsampler_res_block(mid_channels)))

    # Per-frame 2D conv + pixel shuffle (upsampler.0 in the checkpoint)
    self$upsampler <- torch::nn_module_list(list(
            torch::nn_conv2d(mid_channels, 4L * mid_channels, kernel_size = 3L, padding = 1L)
        ))

    self$post_upsample_res_blocks <- torch::nn_module_list(
        lapply(seq_len(num_blocks_per_stage),
               function(i) ltx23_upsampler_res_block(mid_channels))
    )
    self$final_conv <- torch::nn_conv3d(mid_channels, in_channels,
                                        kernel_size = 3L, padding = 1L)
},
    forward = function(hidden_states) {
    batch_size <- hidden_states$shape[1]

    hidden_states <- torch::nnf_silu(self$initial_norm(self$initial_conv(hidden_states)))
    for (i in seq_along(self$res_blocks)) {
        hidden_states <- self$res_blocks[[i]](hidden_states)
    }

    # [B, C, F, H, W] -> per-frame [B*F, C, H, W], 2x pixel shuffle, back
    hidden_states <- hidden_states$permute(c(1L, 3L, 2L, 4L, 5L))$
    flatten(start_dim = 1L, end_dim = 2L)
    hidden_states <- self$upsampler[[1]](hidden_states)
    hidden_states <- torch::nnf_pixel_shuffle(hidden_states, 2L)
    hidden_states <- hidden_states$unflatten(1L, c(batch_size, -1L))$
    permute(c(1L, 3L, 2L, 4L, 5L))

    for (i in seq_along(self$post_upsample_res_blocks)) {
        hidden_states <- self$post_upsample_res_blocks[[i]](hidden_states)
    }
    self$final_conv(hidden_states)
}
)

#' Load the LTX-2.3 spatial upscaler weights
#'
#' The checkpoint keys match this module tree directly.
#'
#' @param path Path to e.g. \code{ltx-2.3-spatial-upscaler-x2-1.1.safetensors}.
#' @param device,dtype Placement for the loaded model.
#' @param verbose Logical.
#'
#' @return The loaded \code{ltx23_latent_upsampler}.
#'
#' @export
ltx23_load_upsampler <- function(path, device = "cuda", dtype = "bfloat16",
                                 verbose = TRUE) {
    torch_dtype <- switch(dtype, bfloat16 = torch::torch_bfloat16(),
                          float16 = torch::torch_float16(),
                          float32 = torch::torch_float32(),
                          stop("Unsupported dtype: ", dtype))
    handle <- safetensors::safetensors$new(path.expand(path), framework = "torch")
    keys <- setdiff(handle$keys(), "__metadata__")

    model <- ltx23_latent_upsampler()
    model$to(dtype = torch_dtype)
    dests <- c(model$named_parameters(), model$named_buffers())

    missing_dest <- setdiff(keys, names(dests))
    unfilled <- setdiff(names(dests), keys)
    if (length(missing_dest) || length(unfilled)) {
        stop("Upsampler load mismatch: ", length(missing_dest), " unmapped, ",
             length(unfilled), " unfilled")
    }
    torch::with_no_grad({
        for (key in keys) {
            dests[[key]]$copy_(handle$get_tensor(key))
        }
    })
    model$to(device = device)
    model$eval()
    if (verbose) {
        message("Loaded latent upsampler (", length(keys), " tensors)")
    }
    model
}

#' Adaptive instance normalization between latent tensors
#'
#' Matches each (batch, channel) slice's mean/std to the reference
#' latents, blended by \code{factor} (cf. diffusers
#' \code{LTX2LatentUpsamplePipeline.adain_filter_latent}).
#'
#' @param latents Tensor [B, C, F, H, W].
#' @param reference_latents Tensor with the target statistics.
#' @param factor Numeric blend in [-10, 10]; 0 is identity.
#'
#' @return Filtered latents.
#'
#' @export
ltx23_adain_filter_latent <- function(latents, reference_latents,
                                      factor = 1.0) {
    if (factor == 0) {
        return(latents)
    }
    dims <- c(3L, 4L, 5L)
    r_mean <- reference_latents$mean(dim = dims, keepdim = TRUE)
    r_sd <- reference_latents$std(dim = dims, keepdim = TRUE)
    i_mean <- latents$mean(dim = dims, keepdim = TRUE)
    i_sd <- latents$std(dim = dims, keepdim = TRUE)

    result <- (latents - i_mean)$div(i_sd)$mul(r_sd) + r_mean
    torch::torch_lerp(latents, result, factor)
}

#' Sigmoid tone mapping for latents
#'
#' Compresses the latent dynamic range (cf. diffusers
#' \code{tone_map_latents}). \code{compression} 0 is identity, 1 is the
#' full effect.
#'
#' @param latents Tensor.
#' @param compression Numeric in [0, 1].
#'
#' @return Tone-mapped latents.
#'
#' @export
ltx23_tone_map_latents <- function(latents, compression) {
    if (compression == 0) {
        return(latents)
    }
    stopifnot(compression >= 0, compression <= 1)
    scale_factor <- compression * 0.75
    abs_latents <- torch::torch_abs(latents)
    sigmoid_term <- torch::torch_sigmoid(abs_latents$add(-1)$mul(4 * scale_factor))
    scales <- sigmoid_term$mul(-0.8 * scale_factor)$add(1)
    latents * scales
}
