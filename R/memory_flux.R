#' FLUX Memory Profiles
#'
#' VRAM-based execution profiles for the FLUX.1-schnell pipeline. The
#' 12B transformer runs NF4 (~7 GB) or fp8 (~12 GB), phase-onloaded to
#' the GPU for denoise; the T5-XXL text encoder phase-onloads to the GPU
#' (bfloat16, pinned) on 14 GB+ cards and computes on the CPU (float32)
#' below that, where its ~9.8 GB encode phase does not fit.
#'
#' @name memory_flux
NULL

# Allocator gc gates for the resident-weight image pipelines
# (backend:native). The allocated/reserved ratio gate is chronically
# over its 0.95 threshold under a lazily-reserving caching allocator
# (measured 0.957 at steady state), so the R gc callback fired on
# nearly every allocation - ~2,200 gcs x 62 ms = 89% of wall time on a
# Z-Image generation. Disabling the ratio gate alone OOMs (dead tensor
# handles accumulate until finalizers run), so the absolute
# allocated_rate gate takes over garbage-accumulation duty at 0.65 of
# total VRAM, above the ~8.3 GB live phase peak with headroom under the
# card limit. Measured (RTX 5060 Ti 16 GB): Z-Image 512^2 142 -> 12 s,
# 1024^2 143 -> 25 s; klein 1024^2 48 -> 13 s, no OOM, reserved peak
# 12.9 GB. ltx23_tune_gc's reserved_rate stays as the fragmentation
# net; the LTX video pipelines keep their own separately measured
# gates.
.flux_gc_gates <- function(footprint_gb = 12) {
    if (is.null(getOption("torch.cuda_allocator_allocated_reserved_rate"))) {
        options(torch.cuda_allocator_allocated_reserved_rate = 1.0)
    }
    if (is.null(getOption("torch.cuda_allocator_allocated_rate"))) {
        options(torch.cuda_allocator_allocated_rate = 0.65)
    }
    ltx23_tune_gc(footprint_gb = footprint_gb)
    # start_torch() reads the three gate options ONCE at torch init and
    # pushes them into the C++ allocator; option changes after that are
    # inert. Torch is long started by the time a loader runs, so push
    # the current values into the live allocator directly.
    push <- get0("cpp_set_cuda_allocator_allocator_thresholds",
                 envir = asNamespace("torch"))
    if (is.function(push)) {
        try(push(
                 getOption("torch.cuda_allocator_reserved_rate", 0.2),
                 getOption("torch.cuda_allocator_allocated_rate", 0.8),
                 getOption("torch.cuda_allocator_allocated_reserved_rate", 0.8)
            ), silent = TRUE)
    }
    invisible(NULL)
}

#' Resolve a FLUX memory profile
#'
#' A thin adapter over \code{\link{recommend}} for the FLUX.1 pipeline,
#' kept for back-compatibility. \code{recommend("flux1")} is the policy;
#' this reshapes it into the legacy profile fields the loader consumes.
#' Precision now rises with VRAM (nf4 default, fp8 GPU-resident on 14 GB+
#' cards when safetensors can read float8, bf16 on 24 GB+); the old
#' bands, which put fp8 in a narrow low-VRAM slot it can no longer fit,
#' were backwards.
#'
#' @param vram_gb Numeric or NULL. Available VRAM; auto-detected when
#'   NULL (via nvidia-smi).
#'
#' @return List with \code{name}, \code{precision} ("nf4"/"fp8"/"bf16"),
#'   \code{attn_chunk}, \code{text_device}, \code{phase_offload},
#'   \code{max_pixels}, and (advisory) \code{fork_suggested} and
#'   \code{note}.
#'
#' @export
flux_memory_profile <- function(vram_gb = NULL) {
    r <- recommend("flux1", vram_gb = vram_gb)
    name <- if (identical(r$devices$transformer, "cpu")) {
        "cpu_only"
    } else if (r$precision %in% c("bf16", "fp8")) {
        "high"
    } else if (r$max_pixels >= 1024L * 1024L) {
        "medium"
    } else {
        "low"
    }
    list(name = name, precision = r$precision, attn_chunk = r$attn_chunk,
         text_device = r$text_device, phase_offload = r$offload,
         max_pixels = r$max_pixels, fork_suggested = r$fork_suggested,
         note = r$note)
}
