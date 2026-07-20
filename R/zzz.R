#' @importFrom utils head
NULL

# torch indexing uses `..` as a Python-style ellipsis
utils::globalVariables("..")

.onLoad <- function(libname, pkgname) {
    # start_torch() reads torch.threshold_call_gc exactly once at torch
    # init and there is no live setter (unlike the CUDA gates, which
    # ltx23_tune_gc pushes via cpp), so this is the only place the
    # option can land in time. The 4000 MB default fires an R gc for
    # every few GB of host allocation; raising it measurably cuts gc
    # counts, though an LTX load-time A/B showed no wall-clock change
    # (those gcs were cheap). A user-set option wins.
    if (is.null(getOption("torch.threshold_call_gc"))) {
        options(torch.threshold_call_gc = 16000)
    }
}
