#' FLUX Memory Profiles
#'
#' VRAM-based execution profiles for the FLUX.1-schnell pipeline,
#' following the LTX-2.3 profile pattern. The 12B transformer runs NF4
#' (~7 GB, GPU-resident) or fp8 (~12 GB, CPU-resident and streamed);
#' the T5-XXL text encoder runs float32 on the CPU by default.
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
#' @param vram_gb Numeric or NULL. Available VRAM; auto-detected when
#'   NULL (via gpu.ctl or nvidia-smi).
#'
#' @return List with \code{name}, \code{precision} ("nf4"/"fp8"),
#'   \code{attn_chunk}, \code{text_device}, \code{phase_offload}, and
#'   \code{max_pixels} (largest validated image area).
#'
#' @export
flux_memory_profile <- function(vram_gb = NULL) {
    if (is.null(vram_gb)) {
        vram_gb <- .detect_vram(use_free = TRUE)
        if (is.null(vram_gb) || is.na(vram_gb) || vram_gb <= 0) {
            vram_gb <- 0
        }
    }

    if (vram_gb >= 12) {
        list(name = "high", precision = "nf4", attn_chunk = NULL,
             text_device = "cpu", phase_offload = TRUE,
             max_pixels = 1536L * 1536L)
    } else if (vram_gb >= 9) {
        list(name = "medium", precision = "nf4", attn_chunk = 2048L,
             text_device = "cpu", phase_offload = TRUE,
             max_pixels = 1024L * 1024L)
    } else if (vram_gb >= 7) {
        list(name = "low", precision = "fp8", attn_chunk = 1024L,
             text_device = "cpu", phase_offload = TRUE,
             max_pixels = 768L * 768L)
    } else {
        list(name = "cpu_only", precision = "nf4", attn_chunk = NULL,
             text_device = "cpu", phase_offload = FALSE,
             max_pixels = 512L * 512L)
    }
}
