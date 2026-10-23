#' Open the LTX-2.5 component files as one checkpoint
#'
#' LTX-2.3 shipped every component in one file; LTX-2.5 splits them: the
#' transformer (with the text connectors), the Gemma 4 text encoder file
#' (which carries the connectors' \code{text_embedding_projection}), the
#' video VAE, and the audio VAE plus vocoder. This presents them as a
#' single \code{ltx23_checkpoint} under the 2.3 key names (the video VAE
#' keys gain their \code{vae.} prefix), so \code{\link{ltx23_load_pipeline}}
#' and its component loaders work unchanged.
#'
#' @param transformer The 2.5 transformer file
#'   (\code{ltx-2.5-22b-distilled-transformer-bf16.safetensors}) or an
#'   fp8/NF4 artifact directory quantized from it.
#' @param text_encoder The 2.5 Gemma 4 text encoder file, or its NF4
#'   artifact directory (\code{\link{gemma4_quantize_nf4}} keeps the
#'   projection); only \code{text_embedding_projection.*} is used here.
#' @param video_vae The 2.5 video VAE file. Use the convolutional decoder
#'   (\code{ltx-2.5-video-vae-conv-bf16.safetensors}); the default
#'   diffusion decoder is not supported.
#' @param audio_vae Optional 2.5 audio VAE + vocoder file
#'   (\code{ltx-2.5-audio-vae-bf16.safetensors}).
#'
#' @return An \code{ltx23_checkpoint} whose \code{handle$get_tensor()}
#'   reads each key from its source file.
#'
#' @export
ltx25_open_checkpoint <- function(transformer, text_encoder, video_vae,
                                  audio_vae = NULL) {
    open_any <- function(path, require_version) {
        if (file.exists(file.path(path, "manifest.json"))) {
            ltx23_open_fp8_checkpoint(path)
        } else {
            ltx23_open_checkpoint(path, require_version = require_version)
        }
    }
    dit <- open_any(transformer, "2.5")
    if (!startsWith(dit$version %||% "", "2.5")) {
        stop("Transformer is model_version '", dit$version %||% "none",
             "', expected 2.5*: ", transformer)
    }
    te <- open_any(text_encoder, NULL)
    vae <- ltx23_open_checkpoint(video_vae, require_version = "2.5")
    if (any(grepl("(^|\\.)transformer_blocks\\.", vae$keys)) ||
        !any(grepl("^(vae\\.)?decoder\\.up_blocks\\.", vae$keys))) {
        stop("Not a convolutional video VAE file (the diffusion decoder is ",
             "not supported): ", video_vae)
    }
    sources <- list(dit = dit, te = te, vae = vae)
    if (!is.null(audio_vae)) {
        sources$audio <- ltx23_open_checkpoint(audio_vae, require_version = "2.5")
    }

    # canonical key -> (source, original key)
    proj <- grep("^text_embedding_projection\\.", te$keys, value = TRUE)
    if (!length(proj)) {
        stop("No text_embedding_projection.* in ", text_encoder)
    }
    vae_keys <- vae$keys
    routes <- list(
        dit = list(orig = dit$keys, canon = dit$keys),
        te = list(orig = proj, canon = proj),
        vae = list(orig = vae_keys,
                   canon = ifelse(startsWith(vae_keys, "vae."), vae_keys,
                                  paste0("vae.", vae_keys)))
    )
    if (!is.null(sources$audio)) {
        ak <- grep("^(audio_vae|vocoder)\\.", sources$audio$keys, value = TRUE)
        routes$audio <- list(orig = ak, canon = ak)
    }
    canon <- unlist(lapply(routes, `[[`, "canon"), use.names = FALSE)
    orig <- unlist(lapply(routes, `[[`, "orig"), use.names = FALSE)
    src <- rep(names(routes), vapply(routes, function(r) length(r$canon), 1L))
    dup <- canon[duplicated(canon)]
    if (length(dup)) {
        stop("Key present in more than one LTX-2.5 file: ", dup[1])
    }

    handle <- list(get_tensor = function(key) {
        i <- match(key, canon)
        if (is.na(i)) {
            stop("Key not found in the LTX-2.5 files: ", key)
        }
        sources[[src[i]]]$handle$get_tensor(orig[i])
    })

    vcfg <- vae$config
    acfg <- sources$audio$config
    config <- list(transformer = dit$config$transformer,
                   scheduler = dit$config$scheduler,
                   vae = vcfg$vae %||% vcfg,
                   audio_vae = acfg$audio_vae, vocoder = acfg$vocoder)
    structure(
              list(handle = handle, keys = canon, version = dit$version,
                   config = config, format = dit$format, path = transformer,
                   sources = vapply(list(transformer = transformer,
                                         text_encoder = text_encoder,
                                         video_vae = video_vae,
                                         audio_vae = audio_vae %||% NA_character_),
                                    path.expand, "")),
              class = "ltx23_checkpoint"
    )
}
