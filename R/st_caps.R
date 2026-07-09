#' safetensors read-capability probes and fork messaging
#'
#' CRAN safetensors (<= 0.2.1) reads bfloat16 but cannot write it, and
#' has no float8 support at all; the fixes are upstream
#' (mlverse/safetensors#11 for bfloat16 write, #13 for float8) and in the
#' cornball-ai/safetensors fork. Two capabilities matter and they differ:
#'
#' \itemize{
#'   \item \emph{write} (\code{\link{flux_quantize}}'s internal
#'     \code{.st_can_write}, in quantize_flux.R): needed to BUILD a
#'     quantized artifact in that dtype.
#'   \item \emph{read} (\code{.st_can_read}, here): needed to LOAD a
#'     hosted artifact in that dtype. This is the capability that gates
#'     user-facing recommendations. It is strictly weaker than write:
#'     CRAN safetensors reads bfloat16 it cannot write, so the write
#'     probe is the wrong signal for whether a hosted bf16 artifact will
#'     load.
#' }
#'
#' Both are capability-probed, never version-pinned, so the fork
#' requirement self-heals the day the fixes reach CRAN.
#'
#' @name st_caps
NULL

# Read-probe cache, keyed by dtype. Separate from quantize_flux.R's
# `.st_caps` write cache: the same dtype can be readable but not
# writable (bfloat16 on CRAN), so the two must not share entries.
.st_read_caps <- new.env(parent = emptyenv())

# Write a minimal 2-element safetensors file by hand: a u64
# little-endian header length, the JSON header, then the raw tensor
# bytes. Deliberately does NOT go through safetensors::safe_save_file -
# that is the whole point, since a CRAN safetensors cannot WRITE
# bfloat16 yet can READ it. Lets `.st_can_read` test read capability in
# isolation from write capability.
.st_write_min <- function(path, dtype_name, payload) {
    hdr <- sprintf('{"w":{"dtype":"%s","shape":[2],"data_offsets":[0,%d]}}',
                   dtype_name, length(payload))
    hb <- charToRaw(hdr)
    n <- length(hb)
    # u64 header length, little-endian (headers here are < 256 bytes)
    len_bytes <- as.raw(c(n %% 256L, (n %/% 256L) %% 256L, 0L, 0L, 0L, 0L,
                          0L, 0L))
    con <- file(path, "wb")
    on.exit(close(con), add = TRUE)
    writeBin(len_bytes, con)
    writeBin(hb, con)
    writeBin(payload, con)
    invisible(path)
}

# The [0, 1] byte patterns for the two dtypes the recommender gates on.
# bfloat16 is the top 16 bits of float32: 1.0f = 0x3F800000 -> 0x3F80
# (little-endian 0x80 0x3F); 0.0 -> 0x0000. float8_e4m3fn: 1.0 = exp
# bias 7, mantissa 0 = 0x38; 0.0 = 0x00.
.st_read_probe_spec <- list(
                            bfloat16 = list(name = "BF16",
                                            bytes = as.raw(c(0x00, 0x00, 0x80, 0x3f))),
                            float8_e4m3fn = list(name = "F8_E4M3",
                                                 bytes = as.raw(c(0x00, 0x38)))
)

#' Probe whether the installed safetensors can READ a dtype
#'
#' Hand-builds a tiny safetensors file of the dtype (via
#' \code{.st_write_min}, no safetensors writer involved) and tries to
#' load it back. Cached per session;
#' \code{options(diffuseR.st_read_caps = list(bfloat16 = TRUE, ...))}
#' overrides the probe for tests and for forcing a tier.
#'
#' @param dtype "bfloat16" or "float8_e4m3fn".
#' @return Logical.
#' @keywords internal
.st_can_read <- function(dtype = c("bfloat16", "float8_e4m3fn")) {
    dtype <- match.arg(dtype)
    override <- getOption("diffuseR.st_read_caps")
    if (!is.null(override) && !is.null(override[[dtype]])) {
        return(isTRUE(override[[dtype]]))
    }
    cached <- .st_read_caps[[dtype]]
    if (!is.null(cached)) {
        return(cached)
    }
    ok <- requireNamespace("safetensors", quietly = TRUE) && tryCatch({
        spec <- .st_read_probe_spec[[dtype]]
        tmp <- tempfile(fileext = ".safetensors")
        on.exit(unlink(tmp), add = TRUE)
        .st_write_min(tmp, spec$name, spec$bytes)
        y <- safetensors::safe_load_file(tmp, framework = "torch")
        !is.null(y$w) && identical(as.integer(y$w$shape), 2L)
    }, error = function(e) FALSE)
    .st_read_caps[[dtype]] <- ok
    ok
}

# The standard "install the fork, or press on with nf4" message. Shared
# by the recommender (read side, fit = TRUE: "best fit for your card")
# and the download graceful-fallback path (write side, fit = FALSE, since
# the user asked for it outright) so the wording stays identical
# everywhere. No em dashes (house style).
.st_fork_note <- function(precision, fit = TRUE) {
    precision <- as.character(precision)
    detail <- switch(precision,
                     fp8 = "float8 support (mlverse/safetensors#13)",
                     float8_e4m3fn = "float8 support (mlverse/safetensors#13)",
                     bf16 = "bfloat16 write support (mlverse/safetensors#11)",
                     bfloat16 = "bfloat16 write support (mlverse/safetensors#11)",
                     paste0(precision, " support (mlverse/safetensors)"))
    lead <- if (fit) {
        sprintf("%s is the best fit for your card but needs", precision)
    } else {
        sprintf("%s needs", precision)
    }
    sprintf(paste0(
                   "%s cornball-ai/safetensors until CRAN safetensors ships ",
                   "%s. Install ",
                   "remotes::install_github(\"cornball-ai/safetensors\"), or ",
                   "press on with nf4: same weights, slightly lower precision, ",
                   "and it just works."),
            lead, detail)
}

# When a user explicitly asks for fp8/bf16 but the needed safetensors
# capability is missing, print the fork suggestion and fall back to nf4
# instead of letting a downstream builder or loader fail. nf4, fp16,
# fp32 and anything unrecognized pass through untouched. `mode` selects
# the capability that matters: "write" when about to BUILD an artifact,
# "read" when about to LOAD one.
.st_graceful_precision <- function(precision, mode = c("write", "read"),
                                   verbose = TRUE) {
    mode <- match.arg(mode)
    cap <- switch(precision, fp8 = "float8_e4m3fn", bf16 = "bfloat16",
                  bfloat16 = "bfloat16", NULL)
    if (is.null(cap)) {
        return(precision)
    }
    ok <- if (mode == "write") .st_can_write(cap) else .st_can_read(cap)
    if (ok) {
        return(precision)
    }
    if (verbose) {
        message(.st_fork_note(precision, fit = FALSE),
                "\nFalling back to nf4 for now.")
    }
    "nf4"
}

# Actionable message for the >2 GB shard-read overflow. Split out from
# .st_read_or_breadcrumb so it can be unit-tested without a real 2 GB
# file.
.st_overflow_message <- function(file_path, size_bytes, underlying) {
    sprintf(paste0(
                   "Could not read %s (%.1f GB). Stock CRAN safetensors ",
                   "overflows a 32-bit offset on files at or above 2^31 ",
                   "bytes (~2.15 GB). Rebuild the artifact with smaller ",
                   "shards (the quantizers now default to ",
                   "shard_bytes = 1.9e9), or install ",
                   "remotes::install_github(\"cornball-ai/safetensors\"). ",
                   "Underlying error: %s"),
            basename(file_path), size_bytes / 1e9, underlying)
}

# Run a safetensors read; if it fails AND the backing shard is at/above
# the 2^31-byte ceiling, translate the cryptic overflow into the
# fork-or-smaller-shards breadcrumb. A read that succeeds (fork, or a
# sub-2 GB shard) is untouched; a failure on a small shard rethrows
# verbatim. Reactive by design, so it never false-alarms on a machine
# that can read large files.
.st_read_or_breadcrumb <- function(read_fn, file_path = NULL) {
    tryCatch(read_fn(), error = function(e) {
        sz <- if (!is.null(file_path)) {
            tryCatch(file.size(file_path), error = function(...) NA_real_)
        } else {
            NA_real_
        }
        if (!is.na(sz) && sz >= 2^31) {
            stop(.st_overflow_message(file_path, sz, conditionMessage(e)),
                 call. = FALSE)
        }
        stop(e)
    })
}
