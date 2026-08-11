#' Hosted prebuilt NF4 artifacts
#'
#' diffuseR hosts prebuilt NF4 artifacts for the two models whose
#' licenses permit redistribution: FLUX.2-klein-4B and Z-Image-Turbo
#' (both Apache-2.0, ungated). The artifacts are the exact output of
#' \code{\link{flux_quantize}} (sub-2 GB shards, bf16 residents), so
#' stock CRAN safetensors reads them. Nothing else in the catalog is
#' hosted: FLUX.1-schnell sits behind a HuggingFace license gate and
#' LTX-2.3's license does not permit redistribution, so both still
#' download their sources and quantize locally.
#'
#' @name download_prebuilt
#' @keywords internal
NULL

# model_name -> hosted dataset repo + artifact directory name. The repo
# follows the cornball-ai/<model>-R dataset convention (see
# hf_download_pt); the base mirrors the local artifact directory name so
# a fetched artifact is indistinguishable from a locally built one.
.prebuilt_nf4_spec <- list(
                           flux2 = list(repo = "cornball-ai/flux2-R", base = "flux2-klein-4b-nf4"),
                           zimage = list(repo = "cornball-ai/zimage-R", base = "zimage-turbo-nf4")
)

# Fetch a hosted NF4 artifact into output_dir. Returns TRUE when the
# artifact is complete there, FALSE when the model has no hosted
# artifact or the fetch failed (callers fall back to source +
# quantize). Files are hard-linked out of the hfhub cache when the
# filesystem allows it, copied otherwise.
.flux_fetch_prebuilt <- function(model, output_dir, verbose = TRUE) {
    spec <- .prebuilt_nf4_spec[[model]]
    if (is.null(spec)) {
        return(FALSE)
    }
    manifest_cache <- tryCatch(
                               hfhub::hub_download(spec$repo, paste0(spec$base, "/manifest.json"),
            repo_type = "dataset"),
                               error = function(e) NULL)
    if (is.null(manifest_cache)) {
        if (verbose) {
            message("No hosted NF4 artifact reachable for '", model,
                    "'; building locally instead.")
        }
        return(FALSE)
    }
    manifest <- jsonlite::fromJSON(manifest_cache)
    if (verbose) {
        message("Downloading the prebuilt NF4 artifact from ", spec$repo,
                " (", length(manifest$shards), " shards)...")
    }
    shard_cache <- vapply(manifest$shards, function(s) {
        tryCatch(hfhub::hub_download(spec$repo, paste0(spec$base, "/", s),
                                     repo_type = "dataset"),
                 error = function(e) NA_character_)
    }, character(1))
    if (anyNA(shard_cache)) {
        if (verbose) {
            message("Prebuilt artifact fetch incomplete; ",
                    "building locally instead.")
        }
        return(FALSE)
    }
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    src <- c(manifest_cache, shard_cache)
    dst <- file.path(output_dir, c("manifest.json", manifest$shards))
    for (i in seq_along(src)) {
        .link_or_copy(normalizePath(src[[i]]), dst[[i]])
    }
    ok <- file.exists(file.path(output_dir, "manifest.json"))
    if (ok && verbose) {
        message("NF4 artifact ready: ", output_dir)
    }
    ok
}

# Hard link (free on one filesystem), falling back to a copy. An
# existing destination is replaced.
.link_or_copy <- function(from, to) {
    if (file.exists(to)) {
        unlink(to)
    }
    ok <- suppressWarnings(file.link(from, to))
    if (!ok) {
        ok <- file.copy(from, to, overwrite = TRUE)
    }
    if (!ok) {
        stop("Could not place ", basename(to), " in ", dirname(to),
             call. = FALSE)
    }
    invisible(ok)
}
