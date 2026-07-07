#' JIT-Traced Decode for the LTX-2.3 VAEs and Vocoder
#'
#' The video/audio decoders and the vocoder are static feed-forward
#' graphs, so \code{torch::jit_trace} converts them wholesale: one
#' R-to-libtorch crossing per forward, intermediates freed eagerly by
#' libtorch instead of accumulating as R handles until gc. Traces are
#' shape-specialized (runtime sizes bake into the graph as constants;
#' a mismatched input errors), so they are cached per instance, input
#' shape, dtype, device, and call tag, and re-traced on a miss.
#'
#' A trace captures the module's weight tensors, which would pin them
#' on the GPU across phase offloads; the pipeline releases all traces
#' whenever a component offloads (\code{.ltx23_release_vae_traces}).
#'
#' Controlled by \code{options(diffuseR.jit_vae)} (default TRUE);
#' anything falsy runs the eager modules.
#'
#' @name jit_vae_ltx23
NULL

.ltx23_vae_traces <- new.env(parent = emptyenv())

#' Run a module forward through a shape-specialized trace
#'
#' @param module The nn_module (identity for the cache key; also the
#'   default callable).
#' @param x Input tensor.
#' @param forward Optional closure wrapping the call (for extra fixed
#'   arguments like \code{causal}); must be pure in \code{x}.
#' @param tag Character. Distinguishes call variants of one module.
#'
#' @return The forward result.
#'
#' @keywords internal
.ltx23_traced_call <- function(module, x, forward = NULL, tag = "") {
    run <- forward %||% module
    if (!isTRUE(getOption("diffuseR.jit_vae", TRUE))) {
        return(run(x))
    }
    key <- paste(
                 format(environment(module)), tag,
                 paste(as.integer(x$shape), collapse = "x"),
                 x$dtype$.type(),
                 x$device$type, x$device$index %||% 0L,
                 sep = "|"
    )
    tr <- .ltx23_vae_traces[[key]]
    if (is.null(tr)) {
        # Tracing captures parameters as constants, which requires
        # grad-free tensors; these modules are inference-only. Always
        # trace through a plain closure: jit_trace treats a bare
        # nn_module via a separate (and here broken) code path.
        for (p in module$parameters) p$requires_grad_(FALSE)
        tr <- torch::jit_trace(function(z) run(z), x)
        .ltx23_vae_traces[[key]] <- tr
    }
    tr(x)
}

#' Release all cached decode traces
#'
#' Traces hold references to the weight tensors they captured; drop
#' them when a component leaves the GPU so its memory actually frees.
#'
#' @return Invisibly, NULL.
#'
#' @keywords internal
.ltx23_release_vae_traces <- function() {
    rm(list = ls(.ltx23_vae_traces), envir = .ltx23_vae_traces)
    invisible(NULL)
}

# One choke point for the three video-decoder call sites (direct,
# spatially tiled, temporally tiled)
.ltx23_decode_tile <- function(vae, x, causal) {
    dec <- vae$decoder
    .ltx23_traced_call(
                       dec, x,
                       forward = function(z) dec(z, causal = causal),
                       tag = paste0("causal:", format(causal))
    )
}
