#' Convert cornball SD 2.1 TorchScript weights to a diffusers artifact
#'
#' Rebuilds a diffusers-layout directory (\code{unet/}, \code{vae/},
#' \code{text_encoder/}) from the cornball-ai/sd21-R TorchScript
#' component \code{.pt} files, so the native safetensors pipeline
#' (\code{\link{download_sd21}} / \code{\link{sd_pipeline_from_safetensors}})
#' can load SD 2.1 with no TorchScript.
#'
#' A TorchScript trace preserves the exact parameter tensors, so the
#' result is bit-identical to the source at the chosen dtype. This is the
#' provenance-clean way to build the hosted artifact: the upstream
#' \code{stabilityai/stable-diffusion-2-1} repo was deprecated, SD 2.1 is
#' CreativeML OpenRAIL++-M (redistributable), and cornball already hosts
#' these weights as \code{.pt}. At \code{float16} the components are all
#' sub-2 GB single files (unet ~1.7 GB, text_encoder ~0.65 GB, vae
#' ~0.16 GB), so they load on stock CRAN safetensors.
#'
#' @param pt_dir Directory holding \code{unet-cpu.pt},
#'   \code{decoder-cpu.pt}, \code{text_encoder-cpu.pt} (default: the
#'   diffuseR \code{sd21} data location).
#' @param output_dir Output diffusers directory (default: the
#'   \code{sd_pipeline_from_safetensors} / \code{download_sd21} location).
#' @param dtype \code{"float16"} (default, the hosted tier) or
#'   \code{"float32"}.
#' @param verbose Logical.
#'
#' @return Invisibly, \code{output_dir}.
#'
#' @export
convert_sd21_pt_to_diffusers <- function(pt_dir = NULL, output_dir = NULL,
    dtype = c("float16", "float32"),
    verbose = TRUE) {
    dtype <- match.arg(dtype)
    if (!requireNamespace("safetensors", quietly = TRUE)) {
        stop("The safetensors package is required to write the artifact.")
    }
    if (is.null(pt_dir)) {
        pt_dir <- file.path(tools::R_user_dir("diffuseR", "data"), "sd21")
    }
    if (is.null(output_dir)) {
        output_dir <- file.path(tools::R_user_dir("diffuseR", "data"),
                                "sd21-diffusers")
    }
    pt_dir <- path.expand(pt_dir)
    td <- switch(dtype, float16 = torch::torch_float16(),
                 float32 = torch::torch_float32())

    comps <- list(
                  list(pt = "unet-cpu.pt", strip = "^unet\\.", subdir = "unet",
                       file = "diffusion_pytorch_model.safetensors"),
                  list(pt = "decoder-cpu.pt", strip = "^vae\\.", subdir = "vae",
                       file = "diffusion_pytorch_model.safetensors"),
                  list(pt = "text_encoder-cpu.pt", strip = "^text_encoder\\.",
                       subdir = "text_encoder", file = "model.safetensors")
    )
    absent <- Filter(function(c) !file.exists(file.path(pt_dir, c$pt)), comps)
    if (length(absent)) {
        stop("Missing TorchScript component(s) in ", pt_dir, ": ",
             paste(vapply(absent, `[[`, character(1), "pt"), collapse = ", "),
             ". Fetch them first (e.g. run a native sd21 pipeline once, or ",
             "download_model(\"sd21\")).")
    }
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    for (comp in comps) {
        m <- torch::jit_load(file.path(pt_dir, comp$pt))
        params <- m$parameters
        keys <- sub(comp$strip, "", names(params))
        sd <- stats::setNames(
                              lapply(params, function(p) p$detach()$to(dtype = td)$contiguous()),
                              keys)
        d <- file.path(output_dir, comp$subdir)
        dir.create(d, showWarnings = FALSE)
        safetensors::safe_save_file(sd, file.path(d, comp$file))
        if (identical(comp$subdir, "text_encoder")) {
            .sd21_write_clip_config(params, file.path(d, "config.json"))
        }
        if (verbose) {
            message(sprintf("  %-13s %d tensors -> %s/%s", comp$subdir,
                            length(sd), comp$subdir, comp$file))
        }
        rm(m, params, sd)
        gc(verbose = FALSE)
    }

    if (verbose) {
        message("SD 2.1 diffusers artifact (", dtype, "): ", output_dir)
    }
    invisible(output_dir)
}

# Derive a minimal CLIPTextConfig (what text_encoder_native_from_safetensors
# reads) from the text_encoder .pt parameters, keyed with the export prefix.
.sd21_write_clip_config <- function(params, path) {
    g <- function(k) params[[paste0("text_encoder.", k)]]
    tok <- g("text_model.embeddings.token_embedding.weight")
    pos <- g("text_model.embeddings.position_embedding.weight")
    fc1 <- g("text_model.encoder.layers.0.mlp.fc1.weight")
    layers <- grep("encoder\\.layers\\.", names(params), value = TRUE)
    n_layers <- length(unique(sub(".*layers\\.([0-9]+)\\..*", "\\1", layers)))
    hidden <- as.integer(tok$shape[2])
    cfg <- list(vocab_size = as.integer(tok$shape[1]), hidden_size = hidden,
                num_hidden_layers = as.integer(n_layers),
                num_attention_heads = as.integer(hidden %/% 64L),
                intermediate_size = as.integer(fc1$shape[1]),
                max_position_embeddings = as.integer(pos$shape[1]))
    jsonlite::write_json(cfg, path, auto_unbox = TRUE, pretty = TRUE)
    invisible(cfg)
}
