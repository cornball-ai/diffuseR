#' @importFrom utils head
NULL

# torch indexing uses `..` as a Python-style ellipsis
utils::globalVariables("..")

.onLoad <- function(libname, pkgname) {
    # start_torch() reads torch.threshold_call_gc exactly once at torch
    # init and there is no live setter (unlike the CUDA gates, which
    # ltx23_tune_gc pushes via cpp). The 4000 MB default fires an R gc
    # for every few GB of host allocation - measured ~550 gcs / ~12 s
    # across an LTX pipeline load. Defaulting it here lands before
    # torch starts in any session that loads diffuseR first; a
    # user-set option wins.
    if (is.null(getOption("torch.threshold_call_gc"))) {
        options(torch.threshold_call_gc = 16000)
    }
}
