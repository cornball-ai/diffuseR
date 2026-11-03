#' LTX Latent Upsamplers
#'
#' Fresh R port of the LTX latent upsampler from the diffusers reference
#' (Apache-2.0, pipelines/ltx2/latent_upsampler.py and
#' pipeline_ltx2_latent_upsample.py), in the two layouts the LTX-2.3 and
#' LTX-2.5 releases ship: Conv3d ResBlock stages around either a
#' per-frame 2x pixel-shuffle spatial upsampler (no rational resampler)
#' or a Conv3d 2x temporal pixel shuffle. Operates on unnormalized
#' latents.
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

#' LTX latent upsampler model
#'
#' Spatial: latents [B, 128, F, H, W] -> [B, 128, F, 2H, 2W] (the LTX-2.3
#' and LTX-2.5 spatial x2 upscalers). Temporal: [B, 128, F, H, W] ->
#' [B, 128, 2F - 1, H, W] (the LTX-2.5 temporal x2 upscaler; the first
#' of the doubled frames is dropped, so 8k + 1 pixel frames become
#' 16k + 1).
#'
#' @param in_channels Integer. Latent channels.
#' @param mid_channels Integer (1024 for the spatial upscalers, 512 for
#'   the LTX-2.5 temporal one).
#' @param num_blocks_per_stage Integer.
#' @param temporal Logical. Build the temporal x2 layout.
#'
#' @return Module whose forward(hidden_states) returns the upscaled
#'   latent.
#'
#' @export
ltx23_latent_upsampler <- torch::nn_module(
    "ltx23_latent_upsampler",
    initialize = function(
                          in_channels = 128L,
                          mid_channels = 1024L,
                          num_blocks_per_stage = 4L,
                          temporal = FALSE
    ) {
    self$temporal <- temporal
    self$initial_conv <- torch::nn_conv3d(in_channels, mid_channels,
        kernel_size = 3L, padding = 1L)
    self$initial_norm <- torch::nn_group_norm(32L, mid_channels)

    self$res_blocks <- torch::nn_module_list(lapply(seq_len(num_blocks_per_stage),
            function(i) ltx23_upsampler_res_block(mid_channels)))

    # upsampler.0 in the checkpoint: a per-frame 2D conv before the
    # spatial pixel shuffle, or a 3D conv before the temporal one
    self$upsampler <- torch::nn_module_list(list(
            if (temporal) {
                torch::nn_conv3d(mid_channels, 2L * mid_channels,
                                 kernel_size = 3L, padding = 1L)
            } else {
                torch::nn_conv2d(mid_channels, 4L * mid_channels,
                                 kernel_size = 3L, padding = 1L)
            }
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

    if (self$temporal) {
        # b (c p) f h w -> b c (f p) h w, then drop the first frame
        hidden_states <- self$upsampler[[1]](hidden_states)
        hidden_states <- hidden_states$unflatten(2L, c(-1L, 2L))$
        permute(c(1L, 2L, 4L, 3L, 5L, 6L))$
        flatten(start_dim = 3L, end_dim = 4L)
        hidden_states <- hidden_states$narrow(3L, 2L,
                                              hidden_states$shape[3] - 1L)
    } else {
        # [B, C, F, H, W] -> per-frame [B*F, C, H, W], 2x pixel shuffle, back
        hidden_states <- hidden_states$permute(c(1L, 3L, 2L, 4L, 5L))$
        flatten(start_dim = 1L, end_dim = 2L)
        hidden_states <- self$upsampler[[1]](hidden_states)
        hidden_states <- torch::nnf_pixel_shuffle(hidden_states, 2L)
        hidden_states <- hidden_states$unflatten(1L, c(batch_size, -1L))$
        permute(c(1L, 3L, 2L, 4L, 5L))
    }

    for (i in seq_along(self$post_upsample_res_blocks)) {
        hidden_states <- self$post_upsample_res_blocks[[i]](hidden_states)
    }
    self$final_conv(hidden_states)
}
)

# An unloaded ltx23_latent_upsampler shaped like the checkpoint: mid
# channels from initial_conv, depth from the res_blocks keys, temporal
# when upsampler.0 is a 3D conv
.ltx_upsampler_for <- function(get_tensor, keys) {
    init <- get_tensor("initial_conv.weight")
    up <- get_tensor("upsampler.0.weight")
    blocks <- unique(sub("^res_blocks\\.(\\d+)\\..*", "\\1",
                         grep("^res_blocks\\.", keys, value = TRUE)))
    ltx23_latent_upsampler(in_channels = init$shape[2],
                           mid_channels = init$shape[1],
                           num_blocks_per_stage = length(blocks),
                           temporal = up$ndim == 5L)
}

# Normalized latents [B, 128, F, H, W] through an upsampler: the
# upsampler works on unnormalized latents, then optional AdaIN toward the
# input statistics and tone mapping (diffusers LTX2LatentUpsamplePipeline).
# Returns normalized float32 latents on the upsampler's device.
.ltx23_upsample_latents <- function(upsampler, latents, vae,
                                    adain_factor = 0,
                                    tone_map_compression = 0) {
    f32 <- torch::torch_float32()
    up_device <- upsampler$final_conv$weight$device
    up_dtype <- upsampler$final_conv$weight$dtype
    raw <- ltx23_denormalize_latents(latents$to(device = up_device),
                                     vae$latents_mean, vae$latents_std)
    up <- upsampler(raw$to(dtype = up_dtype))$to(dtype = f32)
    if (adain_factor != 0) {
        up <- ltx23_adain_filter_latent(up, raw$to(dtype = f32), adain_factor)
    }
    if (tone_map_compression > 0) {
        up <- ltx23_tone_map_latents(up, tone_map_compression)
    }
    ltx23_normalize_latents(up, vae$latents_mean, vae$latents_std)
}

#' Locate (or download) an LTX-2.5 latent upscaler file
#'
#' LTX-2.5 ships a spatial x2 upscaler (1.0 GB) and a temporal x2
#' upscaler (0.26 GB) under \code{latent_upscale_models/} in the gated
#' \code{Lightricks/LTX-2.5} repository: accept the LTX-2 Community
#' License on Hugging Face and set \code{HF_TOKEN}. The weights are read
#' from the Hugging Face cache and never redistributed.
#'
#' @param which "spatial" or "temporal".
#' @param download Logical. Download the file when it is not cached
#'   (asks first in interactive sessions; set
#'   \code{options(diffuseR.consent = TRUE)} otherwise).
#' @return The local file path, for \code{\link{ltx23_load_upsampler}}.
#' @export
ltx25_upscaler_path <- function(which = c("spatial", "temporal"),
                                download = FALSE) {
    which <- match.arg(which)
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to locate model weights.")
    }
    repo <- "Lightricks/LTX-2.5"
    file <- sprintf("latent_upscale_models/ltx-2.5-latent-%s-upscaler-x2-bf16-1.0.safetensors",
                    which)
    path <- tryCatch(hfhub::hub_download(repo, file, local_files_only = TRUE),
                     error = function(e) NULL)
    size <- if (which == "spatial") "1.0 GB" else "0.26 GB"
    if (is.null(path) && isTRUE(download) &&
        .ltx23_consent(sprintf("the LTX-2.5 %s upscaler (%s)", which, size))) {
        path <- hfhub::hub_download(repo, file)
    }
    if (is.null(path)) {
        stop("The LTX-2.5 ", which, " upscaler is not cached; call ",
             "ltx25_upscaler_path(\"", which, "\", download = TRUE) (gated: ",
             "accept the license for ", repo, " and set HF_TOKEN)",
             call. = FALSE)
    }
    path
}

#' Load LTX latent upscaler weights
#'
#' Loads the LTX-2.3 spatial x2 upscaler and the LTX-2.5 spatial and
#' temporal x2 upscalers. The layout (spatial or temporal, channel
#' width, depth) is read from the tensor shapes; the checkpoint keys
#' match this module tree directly.
#'
#' @param path Path to e.g. \code{ltx-2.3-spatial-upscaler-x2-1.1.safetensors},
#'   \code{ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors} or
#'   \code{ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors}.
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
    model <- .ltx_upsampler_for(handle$get_tensor, keys)
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
        message("Loaded ", if (model$temporal) "temporal" else "spatial",
                " latent upsampler (", length(keys), " tensors)")
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
