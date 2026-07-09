#' Load HF safetensors weights into the native SD/SDXL UNet
#'
#' The native UNet modules mirror the diffusers
#' \code{UNet2DConditionModel} state-dict keys 1:1, with the sole
#' exception that the time- (and, for SDXL, add-) embedding MLPs are
#' flattened from dotted to underscored names
#' (\code{time_embedding.linear_1} -> \code{time_embedding_linear_1}).
#' These loaders read \code{unet/diffusion_pytorch_model.safetensors}
#' (single file or sharded via its \code{.index.json}) and copy each
#' weight into the matching native parameter, verifying that every native
#' parameter is filled and no key or shape is left unmatched.
#'
#' Reads route through the shared sharded opener, so an oversize (>2 GB)
#' single-file checkpoint on stock CRAN safetensors surfaces the
#' actionable "rebuild with smaller shards or install the fork" message
#' rather than a raw 32-bit overflow.
#'
#' @name unet_safetensors
NULL

# time_embedding.linear_{1,2} -> time_embedding_linear_{1,2}
.unet_remap_sd21 <- function(key) {
    key <- sub("^time_embedding\\.linear_1", "time_embedding_linear_1", key)
    sub("^time_embedding\\.linear_2", "time_embedding_linear_2", key)
}

# SD21 rules plus add_embedding.linear_{1,2} -> add_embedding_linear_{1,2}
.unet_remap_sdxl <- function(key) {
    key <- .unet_remap_sd21(key)
    key <- sub("^add_embedding\\.linear_1", "add_embedding_linear_1", key)
    sub("^add_embedding\\.linear_2", "add_embedding_linear_2", key)
}

# Shared loader: open the (single or sharded) diffusers UNet directory,
# map each HF key to a native parameter through `remap`, copy with a
# shape check, and fail loudly on any unmapped key, shape mismatch, or
# unfilled native parameter.
.load_unet_safetensors <- function(native_unet, path, remap, label,
                                   verbose = TRUE) {
    if (!requireNamespace("safetensors", quietly = TRUE)) {
        stop("The safetensors package is required to read UNet weights.")
    }
    path <- path.expand(path)
    dir <- if (dir.exists(path)) path else dirname(path)
    opened <- .flux_open_sharded_dir(dir, "diffusion_pytorch_model")
    keys <- opened$keys

    dests <- native_unet$named_parameters()
    filled <- character(0)
    unmapped <- character(0)
    mismatch <- character(0)

    torch::with_no_grad({
        for (key in keys) {
            native_name <- remap(key)
            dest <- dests[[native_name]]
            if (is.null(dest)) {
                unmapped <- c(unmapped, key)
                next
            }
            src <- opened$handle$get_tensor(key)
            if (!all(dest$shape == src$shape)) {
                mismatch <- c(mismatch, sprintf("%s (%s vs %s)", native_name,
                                                paste(as.integer(src$shape),
                                                      collapse = "x"),
                                                paste(as.integer(dest$shape),
                                                      collapse = "x")))
                next
            }
            dest$copy_(src)
            filled <- c(filled, native_name)
        }
    })

    if (length(unmapped)) {
        stop(label, " load: ", length(unmapped), " unmapped keys, e.g. ",
             paste(utils::head(unmapped, 3), collapse = ", "))
    }
    if (length(mismatch)) {
        stop(label, " load: ", length(mismatch), " shape mismatches, e.g. ",
             paste(utils::head(mismatch, 3), collapse = ", "))
    }
    unfilled <- setdiff(names(dests), filled)
    if (length(unfilled)) {
        stop(label, " load: ", length(unfilled), " unfilled params, e.g. ",
             paste(utils::head(unfilled, 3), collapse = ", "))
    }
    if (verbose) {
        message("Loaded ", length(filled), " ", label, " parameters")
    }
    invisible(native_unet)
}

#' Load HF safetensors weights into the native SD21 UNet
#'
#' @param native_unet A \code{\link{unet_native}} module.
#' @param path Path to the UNet directory (containing
#'   \code{diffusion_pytorch_model.safetensors} or its shard index) or
#'   directly to the single-file checkpoint.
#' @param verbose Print how many parameters were loaded.
#'
#' @return The native UNet with weights loaded (invisibly).
#' @export
load_unet_safetensors <- function(native_unet, path, verbose = TRUE) {
    .load_unet_safetensors(native_unet, path, .unet_remap_sd21, "SD21 UNet",
                           verbose = verbose)
}

#' Load HF safetensors weights into the native SDXL UNet
#'
#' @param native_unet A \code{\link{unet_sdxl_native}} module.
#' @param path Path to the UNet directory (containing
#'   \code{diffusion_pytorch_model.safetensors} or its shard index) or
#'   directly to the single-file checkpoint.
#' @param verbose Print how many parameters were loaded.
#'
#' @return The native UNet with weights loaded (invisibly).
#' @export
load_unet_sdxl_safetensors <- function(native_unet, path, verbose = TRUE) {
    .load_unet_safetensors(native_unet, path, .unet_remap_sdxl, "SDXL UNet",
                           verbose = verbose)
}
