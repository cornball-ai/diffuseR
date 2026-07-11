#' Native SDXL pipeline from diffusers safetensors
#'
#' Assemble and run the native SDXL pipeline directly from a HuggingFace
#' diffusers directory (\code{unet/}, \code{vae/}, \code{text_encoder/},
#' \code{text_encoder_2/}), with no TorchScript \code{.pt} step - so it
#' works on Blackwell and loads the same weights everyone else uses. The
#' counterpart to \code{\link{sd_pipeline_from_safetensors}}, adding the
#' second text encoder (OpenCLIP ViT-bigG) and the added-conditioning
#' embeddings (pooled \code{text_embeds} + \code{time_ids}) SDXL needs.
#'
#' @name sdxl_pipeline_safetensors
NULL

.sdxl_repo <- "cornball-ai/sdxl-R"

# Read a diffusers VAE's scaling_factor from its config.json (SDXL is
# 0.13025, distinct from SD 2.1's 0.18215); fall back to `default` if the
# config is missing the field.
.vae_scaling_factor <- function(vae_dir, default = 0.13025) {
    cfg <- file.path(path.expand(vae_dir), "config.json")
    if (!file.exists(cfg)) {
        return(default)
    }
    j <- jsonlite::fromJSON(cfg, simplifyVector = TRUE)
    sf <- j$scaling_factor
    if (is.null(sf) || !is.numeric(sf)) {
        default
    } else {
        as.numeric(sf)
    }
}

#' Assemble a native SDXL pipeline from a diffusers safetensors directory
#'
#' Builds the two native CLIP text encoders, the native SDXL UNet, and the
#' VAE decode module from a diffusers directory using the
#' \code{*_from_safetensors} constructors, places each on its component
#' device, and returns the list the \code{\link{txt2img_sdxl}} denoise
#' loop expects.
#'
#' Both encoders return their penultimate hidden state (SDXL feeds the
#' UNet the concatenated \code{[text_encoder (768) | text_encoder_2
#' (1280)]} = 2048-dim penultimate embeds); \code{text_encoder} uses
#' \code{quick_gelu} (OpenAI CLIP ViT-L) and \code{text_encoder_2} uses
#' exact GELU (OpenCLIP bigG). The pooled \code{text_embeds} come from
#' \code{text_encoder_2}'s full stack. The VAE decodes in float32 (the
#' SDXL fp16 VAE overflows in fp16) and its \code{scaling_factor} is read
#' from \code{vae/config.json}.
#'
#' @param diffusers_dir Directory with \code{unet/}, \code{vae/},
#'   \code{text_encoder/}, \code{text_encoder_2/} subdirectories.
#' @param devices Named list of component devices (\code{unet},
#'   \code{decoder}, \code{text_encoder}, \code{text_encoder2});
#'   defaults to all-CPU. \code{text_encoder2} defaults to the
#'   \code{text_encoder} device when unset.
#' @param unet_dtype A torch dtype for the UNet (default float16 on CUDA,
#'   float32 on CPU).
#' @param verbose Logical.
#'
#' @return A list with \code{unet}, \code{decoder}, \code{text_encoder},
#'   \code{text_encoder2}, \code{vae_scaling}, and \code{native_decode}
#'   (\code{TRUE}; the decoder already applies \code{post_quant_conv}).
#'
#' @export
sdxl_pipeline_from_safetensors <- function(diffusers_dir, devices = NULL,
    unet_dtype = NULL, verbose = TRUE) {
    diffusers_dir <- path.expand(diffusers_dir)
    if (is.null(devices)) {
        devices <- list(unet = "cpu", decoder = "cpu", text_encoder = "cpu",
                        text_encoder2 = "cpu")
    }
    if (is.null(devices$text_encoder2)) {
        devices$text_encoder2 <- devices$text_encoder
    }
    if (is.null(unet_dtype)) {
        unet_dtype <- if (identical(devices$unet, "cuda")) {
            torch::torch_float16()
        } else {
            torch::torch_float32()
        }
    }

    if (verbose) {
        message("Building text encoder 1 (CLIP ViT-L)...")
    }
    # CLIP ViT-L: quick_gelu, penultimate hidden state, no final LN.
    text_encoder <- text_encoder_native_from_safetensors(
        file.path(diffusers_dir, "text_encoder"),
        apply_final_ln = FALSE, return_penultimate = TRUE,
        gelu_type = "quick", verbose = FALSE)
    text_encoder$to(device = torch::torch_device(devices$text_encoder),
                    dtype = torch::torch_float32())

    if (verbose) {
        message("Building text encoder 2 (OpenCLIP bigG)...")
    }
    # OpenCLIP bigG: penultimate hidden state + pooled text_embeds.
    text_encoder2 <- text_encoder2_native_from_safetensors(
        file.path(diffusers_dir, "text_encoder_2"),
        return_penultimate = TRUE, verbose = FALSE)
    text_encoder2$to(device = torch::torch_device(devices$text_encoder2),
                     dtype = torch::torch_float32())

    if (verbose) {
        message("Building UNet...")
    }
    unet <- unet_sdxl_native_from_safetensors(file.path(diffusers_dir, "unet"),
        verbose = FALSE)
    unet$to(device = torch::torch_device(devices$unet), dtype = unet_dtype)

    if (verbose) {
        message("Building VAE decoder...")
    }
    # decode = post_quant_conv + decoder, in float32 (the SDXL fp16 VAE
    # overflows in fp16; up-casting the stored fp16 weights is lossless).
    decoder <- .sd_vae_decode_from_safetensors(file.path(diffusers_dir, "vae"),
        latent_channels = 4L)
    decoder$to(device = torch::torch_device(devices$decoder),
               dtype = torch::torch_float32())

    vae_scaling <- .vae_scaling_factor(file.path(diffusers_dir, "vae"),
                                       default = 0.13025)

    list(unet = unet, decoder = decoder, text_encoder = text_encoder,
         text_encoder2 = text_encoder2, vae_scaling = vae_scaling,
         native_decode = TRUE)
}

# Hosted on the cornball-ai/sdxl-R dataset under diffusers/. fp16, sub-2 GB
# per file (the 5 GB UNet is re-sharded via reshard_safetensors so it is
# CRAN-safetensors readable). The native UNet constructor uses the SDXL
# defaults; only the two CLIP encoders and the VAE need a config.json.
.sdxl_files <- c("diffusers/vae/config.json",
                 "diffusers/vae/diffusion_pytorch_model.safetensors",
                 "diffusers/text_encoder/config.json",
                 "diffusers/text_encoder/model.safetensors",
                 "diffusers/text_encoder_2/config.json",
                 "diffusers/text_encoder_2/model.safetensors",
                 "diffusers/unet/config.json",
                 "diffusers/unet/diffusion_pytorch_model.safetensors.index.json")

#' Download the Stable Diffusion XL diffusers weights
#'
#' Fetches the UNet (re-sharded to sub-2 GB shards), VAE, and both CLIP
#' text encoders from the \code{cornball-ai/sdxl-R} HuggingFace dataset
#' (fp16 diffusers safetensors, converted from the original
#' \code{stabilityai/stable-diffusion-xl-base-1.0} OpenRAIL++ weights).
#' About 7 GB, one-time. The native tokenizer and DDIM scheduler need no
#' downloads.
#'
#' @param verbose Logical.
#'
#' @return Invisibly, the diffusers directory (the parent of
#'   \code{unet/}, \code{vae/}, \code{text_encoder/},
#'   \code{text_encoder_2/}).
#'
#' @export
download_sdxl <- function(verbose = TRUE) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to download model weights.")
    }
    have <- !is.null(tryCatch(
                              hfhub::hub_download(.sdxl_repo, .sdxl_files[[1]],
                repo_type = "dataset", local_files_only = TRUE),
                              error = function(e) NULL))
    if (!have) {
        ok <- .ltx23_consent(
                             "Stable Diffusion XL UNet + VAE + 2 CLIP text encoders (~7 GB)")
        if (!ok) {
            stop("Download cancelled.", call. = FALSE)
        }
        if (verbose) {
            message("Downloading Stable Diffusion XL (diffusers safetensors)...")
        }
    }
    paths <- vapply(.sdxl_files, function(f) {
        hfhub::hub_download(.sdxl_repo, f, repo_type = "dataset")
    }, character(1))
    # The UNet ships sharded: fetch every shard named in its index.
    index_path <- paths[[length(.sdxl_files)]]
    idx <- jsonlite::fromJSON(index_path, simplifyVector = TRUE)
    shards <- unique(unlist(idx$weight_map))
    for (s in shards) {
        hfhub::hub_download(.sdxl_repo, paste0("diffusers/unet/", s),
                            repo_type = "dataset")
    }
    # diffusers root = parent of vae/ (paths[[1]] is diffusers/vae/config.json)
    diffusers_dir <- dirname(dirname(paths[[1]]))
    if (verbose) {
        message("SDXL ready: ", diffusers_dir)
    }
    invisible(diffusers_dir)
}

