#' VRAM Detection and Management Utilities
#'
#' Device detection, VRAM reporting, and module offloading helpers shared
#' by the image and video pipelines.
#'
#' @name vram
NULL

#' Check if GPU is Blackwell Architecture
#'
#' Blackwell GPUs (RTX 50xx) may need special handling.
#'
#' @return Logical. TRUE if Blackwell GPU detected.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' if (is_blackwell_gpu()) {
#'   message("Using Blackwell-compatible settings")
#' }
#' }
is_blackwell_gpu <- function() {
    # Check compute capability via torch. cuda_is_available() ERRORS
    # (not FALSE) when the torch package is installed without its
    # lantern binaries - fresh installs, win-builder, CRAN checks - so
    # probe soft: no lantern means no CUDA means not Blackwell.
    avail <- tryCatch(torch::cuda_is_available(), error = function(e) FALSE)
    if (avail) {
        cap <- tryCatch(torch::cuda_get_device_capability(0L),
                        error = function(e) NULL)
        if (!is.null(cap)) {
            # Blackwell is compute 12.x
            return(as.integer(cap[[1]]) >= 12L)
        }
    }

    FALSE
}

#' Detect Available VRAM
#'
#' Asks nvidia-smi.
#'
#' @param use_free Logical. If TRUE, return free VRAM. If FALSE, return total.
#'
#' @return Numeric. VRAM in GB, or 0 if no GPU detected.
#' @keywords internal
.detect_vram <- function(use_free = FALSE) {
    smi <- suppressWarnings(tryCatch(
                                     system2("nvidia-smi",
                c(paste0("--query-gpu=memory.",
                        if (use_free) "free" else "total"),
                    "--format=csv,noheader,nounits"),
                stdout = TRUE, stderr = FALSE),
                                     error = function(e) character(0)
        ))
    mb <- suppressWarnings(as.numeric(smi[1]))
    if (isTRUE(mb > 0)) {
        return(mb / 1024)
    }

    # Fallback: check if CUDA available but can't determine VRAM
    if (torch::cuda_is_available()) {
        # Conservative estimate - assume 8GB if we can't detect
        message("Could not detect VRAM via nvidia-smi; assuming 8 GB.")
        return(8)
    }

    # No GPU detected
    0
}

#' Offload Module to CPU
#'
#' Moves a torch module and all its parameters to CPU.
#'
#' @param module A torch nn_module.
#' @param gc Logical. Run garbage collection after offload.
#'
#' @return The module (modified in place).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' model$to(device = "cuda")
#' output <- model(x)
#' offload_to_cpu(model)
#' }
offload_to_cpu <- function(module, gc = TRUE) {
    module$to(device = "cpu")
    if (gc && torch::cuda_is_available()) {
        gc()
        torch::cuda_empty_cache()
    }
    invisible(module)
}

#' Load Module to GPU
#'
#' Moves a torch module and all its parameters to CUDA.
#'
#' @param module A torch nn_module.
#' @param device Character. Target device (default "cuda").
#'
#' @return The module (modified in place).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' load_to_gpu(model)
#' output <- model(x)
#' offload_to_cpu(model)
#' }
load_to_gpu <- function(module, device = "cuda") {
    module$to(device = device)
    invisible(module)
}

#' Report VRAM Usage
#'
#' Prints current VRAM usage from nvidia-smi.
#'
#' @param label Character. Label for the report.
#'
#' @return Invisibly returns a list with used and free VRAM in GB.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' vram_report("After model load")
#' }
vram_report <- function(label = "") {
    if (!torch::cuda_is_available()) {
        message("[", label, "] No CUDA available")
        return(invisible(list(used = 0, free = 0)))
    }

    smi <- suppressWarnings(tryCatch(
                                     system2("nvidia-smi",
                c("--query-gpu=memory.used,memory.free",
                    "--format=csv,noheader,nounits"),
                stdout = TRUE, stderr = FALSE),
                                     error = function(e) character(0)
        ))
    vals <- suppressWarnings(as.numeric(strsplit(smi[1], ",")[[1]]))
    if (length(vals) == 2L && all(is.finite(vals))) {
        used <- vals[1] / 1024
        free <- vals[2] / 1024
        message(sprintf("[%s] VRAM: %.2f GB used, %.2f GB free", label,
                        used, free))
        return(invisible(list(used = used, free = free)))
    }

    message("[", label, "] VRAM: (nvidia-smi unavailable)")
    invisible(list(used = NA, free = NA))
}

#' Clear VRAM Cache
#'
#' Forces garbage collection and clears CUDA memory cache.
#'
#' @param verbose Logical. Print memory status before/after.
#'
#' @return Invisibly returns NULL.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' clear_vram()
#' }
clear_vram <- function(verbose = FALSE) {
    if (!torch::cuda_is_available()) {
        return(invisible(NULL))
    }

    if (verbose) {
        vram_report("Before clear")
    }

    gc()
    tryCatch(torch::cuda_empty_cache(), error = function(e) NULL)

    if (verbose) {
        vram_report("After clear")
    }

    invisible(NULL)
}
