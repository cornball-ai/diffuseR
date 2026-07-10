#' Native Stable Diffusion pipelines from diffusers safetensors
#'
#' Assemble and run the native SD pipeline directly from a HuggingFace
#' diffusers directory (\code{unet/}, \code{vae/}, \code{text_encoder/}),
#' with no TorchScript \code{.pt} step - so it works on Blackwell and
#' loads the same weights everyone else uses. SD21 is wired end to end
#' here; SDXL still needs its second text encoder and added-conditioning
#' embeddings (tracked in tasks/todo.md).
#'
#' @name sd_pipeline_safetensors
NULL

.sd21_repo <- "stabilityai/stable-diffusion-2-1"

# The SD/SDXL AutoencoderKL applies a 1x1 post_quant_conv to the latent
# before the decoder (decode(z) = decoder(post_quant_conv(z))). The
# native vae_decoder_native is the decoder submodule only (the FLUX VAE
# has no post_quant_conv), so the SD decode path must apply it - without
# it the decode is badly wrong (cos ~0.5 vs the reference). This wrapper
# restores it; its weights come from the artifact's own
# post_quant_conv.{weight,bias}.
sd_vae_decode <- torch::nn_module(
                                  "SDVAEDecode",
                                  initialize = function(decoder, post_quant_conv) {
    self$post_quant_conv <- post_quant_conv
    self$decoder <- decoder
},
                                  forward = function(z) {
    self$decoder(self$post_quant_conv(z))
}
)

# Build the SD decode module (post_quant_conv + native decoder) from a
# diffusers VAE directory.
.sd_vae_decode_from_safetensors <- function(vae_dir, latent_channels = 4L) {
    decoder <- vae_decoder_native_from_safetensors(vae_dir,
        latent_channels = latent_channels, verbose = FALSE)
    path <- file.path(path.expand(vae_dir),
                      "diffusion_pytorch_model.safetensors")
    h <- safetensors::safetensors$new(path, framework = "torch")
    pqc <- torch::nn_conv2d(latent_channels, latent_channels, kernel_size = 1L)
    torch::with_no_grad({
        pqc$weight$copy_(h$get_tensor("post_quant_conv.weight"))
        pqc$bias$copy_(h$get_tensor("post_quant_conv.bias"))
    })
    m <- sd_vae_decode(decoder, pqc)
    m$eval()
    m
}

.sd21_files <- c("unet/config.json",
                 "unet/diffusion_pytorch_model.safetensors",
                 "vae/config.json",
                 "vae/diffusion_pytorch_model.safetensors",
                 "text_encoder/config.json",
                 "text_encoder/model.safetensors")

#' Download the Stable Diffusion 2.1 diffusers weights
#'
#' Fetches the UNet, VAE, and CLIP text encoder (config + safetensors)
#' from \code{stabilityai/stable-diffusion-2-1} into the hfhub cache. The
#' native tokenizer and DDIM scheduler need no downloads. About 5 GB
#' (fp32 UNet); one-time.
#'
#' @param verbose Logical.
#'
#' @return Invisibly, the diffusers directory (the parent of
#'   \code{unet/}, \code{vae/}, \code{text_encoder/}).
#'
#' @export
download_sd21 <- function(verbose = TRUE) {
    if (!requireNamespace("hfhub", quietly = TRUE)) {
        stop("The hfhub package is required to download model weights.")
    }
    have <- !is.null(tryCatch(
                              hfhub::hub_download(.sd21_repo, .sd21_files[[2]],
            local_files_only = TRUE),
                              error = function(e) NULL
        ))
    if (!have) {
        ok <- .ltx23_consent(
                             "Stable Diffusion 2.1 UNet + VAE + CLIP text encoder (~5 GB)"
        )
        if (!ok) {
            stop("Download cancelled.", call. = FALSE)
        }
        if (verbose) {
            message("Downloading Stable Diffusion 2.1 (diffusers safetensors)...")
        }
    }
    dl <- function(f) {
        tryCatch(hfhub::hub_download(.sd21_repo, f), error = function(e) {
            msg <- conditionMessage(e)
            if (grepl("404|not found|[Uu]nauthorized|[Ff]orbidden|401|403",
                      msg)) {
                stop(
                     "Could not fetch ", .sd21_repo, " (", sub("\n.*", "", msg),
                     "). This repo is gated on HuggingFace; a 404 with a valid ",
                     "token means the license has not been accepted yet. To fix:\n",
                     "  1. Visit https://huggingface.co/", .sd21_repo,
                     " and accept the license.\n",
                     "  2. Ensure HF_TOKEN is set (Sys.getenv(\"HF_TOKEN\")).\n",
                     "Or pass an accessible diffusers directory straight to ",
                     "sd_pipeline_from_safetensors() / txt2img_sd21(diffusers_dir=).",
                     call. = FALSE)
            }
            stop(e)
        })
    }
    paths <- vapply(.sd21_files, dl, character(1))
    # the diffusers root is the parent of unet/ (paths[[1]] is unet/config.json)
    diffusers_dir <- dirname(dirname(paths[[1]]))
    if (verbose) {
        message("SD 2.1 ready: ", diffusers_dir)
    }
    invisible(diffusers_dir)
}

#' Assemble a native SD pipeline from a diffusers safetensors directory
#'
#' Builds the native UNet, VAE decoder, and CLIP text encoder from a
#' diffusers directory using the \code{*_from_safetensors} constructors,
#' places each on its component device, and returns the \code{$unet /
#' $decoder / $text_encoder} list the \code{txt2img_*} denoise loop
#' expects.
#'
#' @param diffusers_dir Directory with \code{unet/}, \code{vae/},
#'   \code{text_encoder/} subdirectories.
#' @param model_name Currently "sd21" (SDXL pending its second encoder).
#' @param devices Named list of component devices (\code{unet},
#'   \code{decoder}, \code{text_encoder}); defaults to all-CPU.
#' @param unet_dtype A torch dtype for the UNet (default float16 on CUDA,
#'   float32 on CPU).
#' @param verbose Logical.
#'
#' @return A list with \code{unet}, \code{decoder}, \code{text_encoder}.
#'
#' @export
sd_pipeline_from_safetensors <- function(diffusers_dir, model_name = "sd21",
                                         devices = NULL, unet_dtype = NULL,
                                         verbose = TRUE) {
    if (!identical(model_name, "sd21")) {
        stop("sd_pipeline_from_safetensors currently supports \"sd21\"; ",
             "SDXL needs its second text encoder and added-conditioning ",
             "embeddings (see tasks/todo.md).")
    }
    diffusers_dir <- path.expand(diffusers_dir)
    if (is.null(devices)) {
        devices <- list(unet = "cpu", decoder = "cpu", text_encoder = "cpu")
    }
    if (is.null(unet_dtype)) {
        unet_dtype <- if (identical(devices$unet, "cuda")) {
            torch::torch_float16()
        } else {
            torch::torch_float32()
        }
    }

    if (verbose) {
        message("Building text encoder...")
    }
    # SD 2.1 uses the final-LN last hidden state (v-prediction path). The
    # text encoder and VAE decoder compute in float32 (the denoise loop
    # casts prompt embeds to unet_dtype and the latent to float32 before
    # decode); only the UNet runs at unet_dtype. Up-casting a float16
    # artifact to float32 for these is lossless.
    text_encoder <- text_encoder_native_from_safetensors(
        file.path(diffusers_dir, "text_encoder"), apply_final_ln = TRUE,
        verbose = FALSE)
    text_encoder$to(device = torch::torch_device(devices$text_encoder),
                    dtype = torch::torch_float32())

    if (verbose) {
        message("Building UNet...")
    }
    unet <- unet_native_from_safetensors(file.path(diffusers_dir, "unet"),
                                         verbose = FALSE)
    unet$to(device = torch::torch_device(devices$unet), dtype = unet_dtype)

    if (verbose) {
        message("Building VAE decoder...")
    }
    # decode = post_quant_conv + decoder (the SD VAE needs the 1x1
    # post_quant_conv the FLUX-derived native decoder omits)
    decoder <- .sd_vae_decode_from_safetensors(file.path(diffusers_dir, "vae"),
                                               latent_channels = 4L)
    decoder$to(device = torch::torch_device(devices$decoder),
               dtype = torch::torch_float32())

    list(unet = unet, decoder = decoder, text_encoder = text_encoder)
}
