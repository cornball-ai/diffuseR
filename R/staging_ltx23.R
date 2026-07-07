#' Pinned Staging for Phase-Sequential Components
#'
#' Phase offloading moves each large component (transformer,
#' connectors, VAEs, vocoder) between CPU and GPU every render. From
#' pageable memory those copies run through the driver's bounce
#' buffer at a fraction of PCIe speed; page-locked (pinned) host
#' memory transfers by DMA at full rate. Each component's parameters
#' and buffers are pinned once at load; onload swaps every tensor to
#' a non-blocking GPU copy of its pinned source, and offload simply
#' re-points the tensors at the still-valid pinned copies — weights
#' are immutable during inference, so offload moves no bytes at all.
#'
#' Costs: the pinned copies are non-swappable host RAM for the life
#' of the pipeline (about the model's CPU footprint), and page-locking
#' ~20GB adds ~30s to pipeline load. Measured per render the win is
#' small (~2.7s: warm pinned onload 0.54s vs pageable 0.99s, offload
#' 0.06s vs 2.36s) because the real phase-transition cost was
#' allocator pool regrowth, fixed separately by the loader's pool
#' pre-warm and by not emptying the CUDA cache between phases. Off by
#' default; enable for long multi-render sessions with
#' \code{options(diffuseR.pin_staging = TRUE)} before
#' \code{ltx23_load_pipeline} (breaks even after ~11 renders).
#'
#' @name staging_ltx23
NULL

#' Pin a component's tensors for fast phase transfer
#'
#' @param module An nn_module on the CPU.
#'
#' @return A list of \code{list(live, pinned)} tensor pairs, or NULL
#'   if pinning is unavailable (no CUDA, or page-locking failed).
#'
#' @keywords internal
.ltx23_pin_component <- function(module) {
    if (!torch::cuda_is_available()) {
        return(NULL)
    }
    cuda_dev <- torch::torch_device("cuda")
    tryCatch({
        tensors <- c(module$parameters, module$buffers)
        suppressWarnings(lapply(tensors, function(p) {
            # The device argument is deprecated upstream but this
            # torch build requires it
            pinned <- p$pin_memory(device = cuda_dev)
            p$set_data(pinned)
            list(live = p, pinned = pinned)
        }))
    }, error = function(e) NULL)
}

#' Move a pinned component onto the compute device
#'
#' Non-blocking copies from pinned memory share the default stream,
#' so later kernels are ordered after them; no explicit sync needed.
#'
#' @keywords internal
.ltx23_staged_onload <- function(staging, device) {
    for (pair in staging) {
        pair$live$set_data(pair$pinned$to(device = device, non_blocking = TRUE))
    }
    invisible(NULL)
}

#' Return a pinned component to the CPU
#'
#' Weights are immutable during inference, so the pinned host copies
#' are still current: offload is a pointer swap, no transfer.
#'
#' @keywords internal
.ltx23_staged_offload <- function(staging) {
    for (pair in staging) {
        pair$live$set_data(pair$pinned)
    }
    invisible(NULL)
}
