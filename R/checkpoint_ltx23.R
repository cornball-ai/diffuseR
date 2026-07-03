#' LTX-2.3 Single-File Checkpoint Reader
#'
#' LTX 2.3 checkpoints ship as one safetensors file containing every
#' component (transformer, connectors, video VAE, audio VAE, vocoder),
#' with the model version and full component configuration embedded in
#' the safetensors metadata. These helpers open the file, validate the
#' version, split the key space by component, and stream tensors into
#' R torch modules one at a time so the 46 GB file is never fully
#' materialized in memory.
#'
#' @name checkpoint_ltx23
NULL

#' Open an LTX-2.3 checkpoint
#'
#' Opens a single-file LTX checkpoint lazily (header only), validates the
#' \code{model_version} metadata, and parses the embedded component
#' configuration.
#'
#' @param path Path to the checkpoint .safetensors file.
#' @param require_version Character. Required \code{model_version} prefix
#'   (default "2.3"). Set to NULL to skip the check.
#'
#' @return An object of class \code{ltx23_checkpoint}: a list with
#'   \code{handle} (safetensors reader), \code{keys}, \code{version},
#'   \code{config} (parsed component configs, or NULL), and \code{path}.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' ckpt <- ltx23_open_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
#' str(ltx23_split_keys(ckpt$keys), max.level = 1)
#' }
ltx23_open_checkpoint <- function(
  path,
  require_version = "2.3"
) {
  if (!requireNamespace("safetensors", quietly = TRUE)) {
    stop("The safetensors package is required to read LTX checkpoints.")
  }
  path <- path.expand(path)
  if (!file.exists(path)) {
    stop("Checkpoint not found: ", path)
  }

  handle <- safetensors::safetensors$new(path, framework = "torch")

  meta <- handle$metadata[["__metadata__"]]
  version <- if (is.list(meta) || is.character(meta)) meta[["model_version"]] else NULL

  if (!is.null(require_version)) {
    if (is.null(version)) {
      stop(
        "Checkpoint has no model_version metadata; expected an LTX ",
        require_version, " single-file checkpoint: ", path
      )
    }
    if (!startsWith(version, require_version)) {
      stop(
        "Checkpoint model_version is '", version, "' but '",
        require_version, "*' is required: ", path
      )
    }
  }

  config <- NULL
  if (!is.null(meta[["config"]])) {
    config <- tryCatch(
      jsonlite::fromJSON(meta[["config"]], simplifyVector = TRUE),
      error = function(e) NULL
    )
  }

  keys <- setdiff(handle$keys(), "__metadata__")

  structure(
    list(
      handle = handle,
      keys = keys,
      version = version,
      config = config,
      path = path
    ),
    class = "ltx23_checkpoint"
  )
}

#' @export
print.ltx23_checkpoint <- function(x, ...) {
  cat("<ltx23_checkpoint>\n")
  cat("  path:    ", x$path, "\n")
  cat("  version: ", x$version %||% "(none)", "\n")
  cat("  tensors: ", length(x$keys), "\n")
  groups <- ltx23_split_keys(x$keys)
  for (g in names(groups)) {
    cat(sprintf("  %-10s %d keys\n", g, length(groups[[g]])))
  }
  invisible(x)
}

#' Split checkpoint keys by component
#'
#' Splits the flat key space of an LTX single-file checkpoint into its
#' component groups. Connector tensors live under the
#' \code{model.diffusion_model.} prefix alongside the transformer, plus a
#' top-level \code{text_embedding_projection.} group; both are routed to
#' the \code{connectors} component.
#'
#' @param keys Character vector of checkpoint tensor names.
#'
#' @return Named list of character vectors: \code{dit},
#'   \code{connectors}, \code{vae}, \code{audio_vae}, \code{vocoder},
#'   and \code{other} (anything unrecognized; should be empty).
#'
#' @export
ltx23_split_keys <- function(keys) {
  dm_prefix <- "model.diffusion_model."
  connector_res <- c(
    "^model\\.diffusion_model\\.video_embeddings_connector\\.",
    "^model\\.diffusion_model\\.audio_embeddings_connector\\.",
    "^text_embedding_projection\\."
  )

  is_connector <- Reduce(`|`, lapply(connector_res, grepl, x = keys))
  is_dit <- startsWith(keys, dm_prefix) & !is_connector
  is_vae <- startsWith(keys, "vae.")
  is_audio_vae <- startsWith(keys, "audio_vae.")
  is_vocoder <- startsWith(keys, "vocoder.")

  claimed <- is_connector | is_dit | is_vae | is_audio_vae | is_vocoder

  list(
    dit = keys[is_dit],
    connectors = keys[is_connector],
    vae = keys[is_vae],
    audio_vae = keys[is_audio_vae],
    vocoder = keys[is_vocoder],
    other = keys[!claimed]
  )
}

#' Stream a checkpoint key group into a module
#'
#' Reads tensors one at a time from an open checkpoint and copies them
#' into the matching parameters/buffers of \code{module}. Destination
#' names are derived by \code{map_key}; \code{$copy_()} handles any
#' dtype/device conversion, so the module may already live on its target
#' device in its target dtype.
#'
#' @param ckpt An \code{ltx23_checkpoint}.
#' @param keys Character vector of checkpoint keys to load (one group
#'   from \code{\link{ltx23_split_keys}}).
#' @param module A torch nn_module to populate.
#' @param map_key Function mapping a checkpoint key to the module's
#'   parameter/buffer name, or NA to skip the key deliberately.
#' @param verbose Logical. Report progress and coverage.
#' @param gc_every Integer. Run \code{gc()} after this many tensors.
#'
#' @return Invisibly, a list with \code{unmapped} (checkpoint keys that
#'   found no destination), \code{skipped} (keys the mapper declined),
#'   and \code{unfilled} (module parameters/buffers never written).
#'   A perfectly loaded group has zero \code{unmapped} and zero
#'   \code{unfilled}.
#'
#' @export
ltx23_load_group <- function(
  ckpt,
  keys,
  module,
  map_key = identity,
  verbose = TRUE,
  gc_every = 50L
) {
  stopifnot(inherits(ckpt, "ltx23_checkpoint"))

  params <- module$named_parameters()
  buffers <- module$named_buffers()
  dests <- c(params, buffers)

  unmapped <- character(0)
  skipped <- character(0)
  filled <- character(0)

  n <- length(keys)
  torch::with_no_grad({
    for (i in seq_along(keys)) {
      key <- keys[[i]]
      dest_name <- map_key(key)

      if (length(dest_name) != 1L || is.na(dest_name)) {
        skipped <- c(skipped, key)
        next
      }
      dest <- dests[[dest_name]]
      if (is.null(dest)) {
        unmapped <- c(unmapped, key)
        next
      }

      value <- ckpt$handle$get_tensor(key)
      if (!identical(as.integer(dest$shape), as.integer(value$shape))) {
        stop(sprintf(
          "Shape mismatch for '%s' -> '%s': checkpoint [%s], module [%s]",
          key, dest_name,
          paste(value$shape, collapse = ","),
          paste(dest$shape, collapse = ",")
        ))
      }
      dest$copy_(value)
      filled <- c(filled, dest_name)
      rm(value)

      if (i %% gc_every == 0L) {
        gc(verbose = FALSE)
        if (verbose && n > 500L) {
          message(sprintf("  loaded %d/%d tensors", i, n))
        }
      }
    }
  })
  gc(verbose = FALSE)

  unfilled <- setdiff(names(dests), filled)

  if (verbose) {
    message(sprintf(
      "Loaded %d/%d tensors (%d skipped); unmapped: %d, unfilled: %d",
      length(filled), n, length(skipped),
      length(unmapped), length(unfilled)
    ))
    if (length(unmapped)) {
      message("  unmapped e.g.: ", paste(utils::head(unmapped, 3), collapse = ", "))
    }
    if (length(unfilled)) {
      message("  unfilled e.g.: ", paste(utils::head(unfilled, 3), collapse = ", "))
    }
  }

  invisible(list(unmapped = unmapped, skipped = skipped, unfilled = unfilled))
}

#' Summarize checkpoint key coverage
#'
#' @param ckpt An \code{ltx23_checkpoint}.
#'
#' @return A data.frame with one row per component group and its key count.
#'
#' @export
ltx23_census <- function(ckpt) {
  stopifnot(inherits(ckpt, "ltx23_checkpoint"))
  groups <- ltx23_split_keys(ckpt$keys)
  data.frame(
    group = names(groups),
    keys = vapply(groups, length, integer(1)),
    row.names = NULL
  )
}
