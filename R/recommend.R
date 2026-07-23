#' Recommend a precision and device configuration for a model
#'
#' One VRAM-and-capability-aware recommendation for every diffuseR
#' model. The policy:
#'
#' \itemize{
#'   \item nf4 is the default tier for the quantized families (flux1,
#'     flux2, zimage, ltx). Its weights are packed uint8 plus float32
#'     blocks in sub-2 GB shards, which every safetensors reads, so it
#'     always loads. The SD models ship no quantized weights; their
#'     floor is fp16 with placement varying by VRAM.
#'   \item When the card has room for a higher-quality tier (fp8 or
#'     bf16) AND the installed safetensors can \emph{read} that dtype
#'     (\code{\link{.st_can_read}}), that tier is recommended instead.
#'   \item When the card has room but safetensors cannot read the tier,
#'     nf4 is recommended and the fork suggestion is surfaced in
#'     \code{note} (never an error).
#' }
#'
#' This is the policy engine; it does no disk I/O and does not know which
#' artifacts are built. Loaders reconcile the recommendation with what is
#' on disk (see \code{\link{flux_load_pipeline}}). Thresholds are
#' validated on an RTX 5060 Ti (16 GB) and are deliberately conservative
#' elsewhere. Video sizing for \code{"ltx"} is coarse here; the LTX
#' pipeline uses \code{\link{ltx23_memory_profile}} for frame-aware
#' placement.
#'
#' The pinning decision: phase-swapped weights are page-locked host
#' copies (see \code{\link{staging}}) that transfer at DMA rate -
#' but pinned pages are unswappable, so on small-RAM machines they turn
#' memory pressure into OOM kills. \code{pin} is TRUE when available
#' host RAM covers the model's pinned set twice over, FALSE below that,
#' FALSE on the cpu tier (nothing stages), and TRUE when RAM cannot be
#' detected (page-locking already fails soft per component). The LTX
#' pipeline, the Gemma3 encoder, and the FLUX-family image loaders
#' (flux1, flux2, zimage) take \code{pin} arguments and stage pinned
#' weights (see \code{\link{staging}}); the SD-family loaders place
#' components statically and do not phase-swap, so \code{pin} is inert
#' for them. \code{options(diffuseR.pin_staging)} is the global switch.
#'
#' @param model "sd21", "sdxl", "flux1", "flux2", "zimage", or "ltx".
#' @param vram_gb Numeric or NULL. Free VRAM in GB; auto-detected via
#'   nvidia-smi when NULL.
#' @param st_caps NULL or a named logical list with \code{bfloat16}
#'   and/or \code{float8_e4m3fn} - the safetensors READ capabilities.
#'   NULL probes the installed safetensors.
#' @param host_ram_gb Numeric or NULL. Available host RAM in GB;
#'   auto-detected (Linux \code{MemAvailable}) when NULL, NA where
#'   undetectable.
#'
#' @return A list with \code{model}, \code{precision}, \code{devices}
#'   (named component -> device map), \code{offload} (phase-offloading
#'   logical), \code{max_pixels}, \code{text_device}, \code{attn_chunk},
#'   \code{vram_gb}, \code{pin} (page-lock the phase-swapped host
#'   copies), \code{pinned_set_gb} (estimated pinned bytes),
#'   \code{host_ram_gb}, \code{fork_suggested} (logical), and
#'   \code{note} (the fork suggestion string, or NULL).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Auto-detect VRAM and probe the installed safetensors
#' recommend("flux2")
#'
#' # A 16 GB card without float8 support: fp8 wanted, nf4 recommended
#' r <- recommend("flux1", vram_gb = 16,
#'                st_caps = list(bfloat16 = TRUE, float8_e4m3fn = FALSE))
#' r$precision       # "nf4"
#' r$fork_suggested  # TRUE
#' cat(r$note)       # the fork-or-nf4 message
#' }
recommend <- function(model = c("sd21", "sdxl", "flux1", "flux2", "zimage",
                                "ltx"),
                      vram_gb = NULL, st_caps = NULL, host_ram_gb = NULL) {
    model <- match.arg(model)
    if (is.null(vram_gb)) {
        vram_gb <- .detect_vram(use_free = TRUE)
    }
    if (is.null(vram_gb) || is.na(vram_gb) || vram_gb < 0) {
        vram_gb <- 0
    }
    if (is.null(st_caps)) {
        st_caps <- list(bfloat16 = .st_can_read("bfloat16"),
                        float8_e4m3fn = .st_can_read("float8_e4m3fn"))
    }

    tiers <- .recommend_specs()[[model]]

    chosen <- NULL
    want <- NULL # first VRAM-eligible tier blocked by a missing read cap
    for (tier in tiers) {
        if (vram_gb < tier$min_vram) {
            next
        }
        need <- tier$needs
        if (is.null(need) || isTRUE(st_caps[[need]])) {
            chosen <- tier
            break
        }
        if (is.null(want)) {
            want <- tier
        }
    }
    if (is.null(chosen)) {
        chosen <- tiers[[length(tiers)]] # terminal cpu tier
    }

    fork <- !is.null(want) && !identical(want$precision, chosen$precision)

    if (is.null(host_ram_gb)) {
        host_ram_gb <- .detect_host_ram()
    }
    pinned_set <- .pinned_set_gb(model, chosen$precision)
    pin <- .pin_decision(model, chosen$precision, host_ram_gb,
                         cpu = isTRUE(chosen$cpu))

    list(
         model = model,
         precision = chosen$precision,
         devices = chosen$devices,
         offload = isTRUE(chosen$offload),
         max_pixels = chosen$max_pixels,
         text_device = chosen$text_device %||% "cpu",
         attn_chunk = chosen$attn_chunk,
         vram_gb = vram_gb,
         pin = pin,
         pinned_set_gb = pinned_set,
         host_ram_gb = host_ram_gb,
         fork_suggested = fork,
         note = if (fork) .st_fork_note(want$precision) else NULL
    )
}

# Available host RAM in GB (Linux MemAvailable); NA where undetectable
# (macOS, Windows). NA feeds a keep-pinning decision: page-locking
# already falls back silently per component, and platforms without
# /proc rarely pair with CUDA.
.detect_host_ram <- function() {
    if (!file.exists("/proc/meminfo")) {
        return(NA_real_)
    }
    tryCatch({
        line <- grep("^MemAvailable:", readLines("/proc/meminfo", n = 20L),
                     value = TRUE)
        if (!length(line)) {
            return(NA_real_)
        }
        kb <- as.numeric(strsplit(trimws(sub("^MemAvailable:", "", line)),
                                  "[[:space:]]+")[[1]][1])
        kb / 1024 ^ 2
    }, error = function(e) NA_real_)
}

# Host GB that pinned staging would page-lock for a model at a given
# precision: the quantized checkpoint plus the text encoder's host
# copy. Coarse artifact-scale estimates - the pin decision needs the
# order of magnitude, not the byte.
.pinned_set_gb <- function(model, precision) {
    sets <- list(
                 sd21 = c(fp16 = 3),
                 sdxl = c(fp16 = 8),
                 flux1 = c(bf16 = 34, fp8 = 22, nf4 = 17), # DiT + T5 bf16 host copy
                 flux2 = c(bf16 = 17, fp8 = 12, nf4 = 10), # DiT + Qwen3 bf16
                 zimage = c(bf16 = 21, fp8 = 14, nf4 = 12), # DiT + Qwen3 bf16
                 ltx = c(fp8 = 34, nf4 = 28) # checkpoint + Gemma3 NF4
    )
    v <- unname(sets[[model]][precision])
    if (length(v) != 1L || is.na(v)) {
        0
    } else {
        v
    }
}

# Host-RAM-aware pinning decision for a model at a resolved precision.
# TRUE when available host RAM covers the pinned set twice over; FALSE
# on the cpu tier (nothing stages, so nothing to page-lock); TRUE when
# RAM is undetectable (page-locking already fails soft per component).
# Shared by recommend() and the phase-offloading loaders.
.pin_decision <- function(model, precision, host_ram_gb = NULL,
                          cpu = FALSE) {
    if (isTRUE(cpu)) {
        return(FALSE)
    }
    if (is.null(host_ram_gb)) {
        host_ram_gb <- .detect_host_ram()
    }
    if (is.na(host_ram_gb)) {
        return(TRUE)
    }
    host_ram_gb >= 2 * .pinned_set_gb(model, precision)
}

# Resolve a loader's pin argument. Precedence: an explicit pin= wins,
# then options(diffuseR.pin_staging) (read with no default so an unset
# option falls through), then the RAM-aware .pin_decision. A "full"
# checkpoint is raw bf16, so it maps to the "bf16" pinned-set row.
.resolve_pin <- function(pin, model, format) {
    if (!is.null(pin)) {
        return(isTRUE(pin))
    }
    opt <- getOption("diffuseR.pin_staging")
    if (!is.null(opt)) {
        return(isTRUE(opt))
    }
    precision <- if (identical(format, "full")) "bf16" else format
    .pin_decision(model, precision)
}

# flux-family component placement: the big DiT and the VAE compute on
# the GPU (or all on CPU for the cpu tier); the text encoder is resident
# on the CPU and phase-onloaded during its own phase.
.dev_flux <- function(gpu = TRUE) {
    if (gpu) {
        list(transformer = "cuda", text = "cpu", vae = "cuda")
    } else {
        list(transformer = "cpu", text = "cpu", vae = "cpu")
    }
}

# One bf16/fp8/nf4/nf4-tight/cpu ladder for the flux-family image models.
# Precision rises with VRAM (nf4 default, fp8/bf16 as upgrades) - the
# inverse of the old flux_memory_profile, which had fp8 in a narrow
# low-VRAM band it can no longer fit now that fp8 is GPU-resident.
#
# text_device is the phase where the encoder computes: "cuda" onloads it
# to the GPU for the encode (fast, pinned), "cpu" keeps it resident on
# the host. flux1's T5-XXL needs ~9.8 GB for its encode phase, so it
# only GPU-encodes above gpu_encode_vram; the small flux2/zimage Qwen3
# encodes on the GPU on every GPU tier (gpu_encode_vram = 0).
.flux_family_tiers <- function(bf16_vram, fp8_vram, nf4_vram, nf4_tight_vram,
                               max_hi, max_mid, max_lo, max_cpu,
                               attn_tight = NULL, gpu_encode_vram = 0) {
    gpu_text <- function(min_vram) {
        if (min_vram >= gpu_encode_vram) "cuda" else "cpu"
    }
    list(
         list(precision = "bf16", min_vram = bf16_vram, needs = "bfloat16",
              devices = .dev_flux(TRUE), offload = TRUE, max_pixels = max_hi,
              text_device = gpu_text(bf16_vram), attn_chunk = NULL),
         list(precision = "fp8", min_vram = fp8_vram, needs = "float8_e4m3fn",
              devices = .dev_flux(TRUE), offload = TRUE, max_pixels = max_mid,
              text_device = gpu_text(fp8_vram), attn_chunk = NULL),
         list(precision = "nf4", min_vram = nf4_vram, needs = NULL,
              devices = .dev_flux(TRUE), offload = TRUE, max_pixels = max_mid,
              text_device = gpu_text(nf4_vram), attn_chunk = NULL),
         list(precision = "nf4", min_vram = nf4_tight_vram, needs = NULL,
              devices = .dev_flux(TRUE), offload = TRUE, max_pixels = max_lo,
              text_device = gpu_text(nf4_tight_vram), attn_chunk = attn_tight),
         list(precision = "nf4", min_vram = 0, needs = NULL, cpu = TRUE,
              devices = .dev_flux(FALSE), offload = FALSE, max_pixels = max_cpu,
              text_device = "cpu", attn_chunk = NULL)
    )
}

# SD-family ladder: the SD models ship no quantized weights, so every
# tier is fp16 - what varies is placement (all-GPU, UNet-only-GPU, or
# CPU) and the area budget. Device maps reuse the auto_devices
# strategy builder.
.sd_tiers <- function(model, fp16_vram, nf4_vram, max_fp16, max_nf4, max_cpu) {
    list(
         list(precision = "fp16", min_vram = fp16_vram, needs = NULL,
              devices = .build_fallback_devices(model, "full_gpu"),
              offload = FALSE, max_pixels = max_fp16),
         list(precision = "fp16", min_vram = nf4_vram, needs = NULL,
              devices = .build_fallback_devices(model, "unet_gpu"),
              offload = TRUE, max_pixels = max_nf4),
         list(precision = "fp16", min_vram = 0, needs = NULL, cpu = TRUE,
              devices = .build_fallback_devices(model, "cpu_only"),
              offload = FALSE, max_pixels = max_cpu)
    )
}

# Per-model tier ladders. Built lazily (device maps call helpers) so the
# thresholds live in one auditable place.
.recommend_specs <- function() {
    px <- function(n) as.integer(n) * as.integer(n)
    list(
         sd21 = .sd_tiers("sd21", fp16_vram = 6, nf4_vram = 3,
                          max_fp16 = px(1024), max_nf4 = px(768),
                          max_cpu = px(512)),
         sdxl = .sd_tiers("sdxl", fp16_vram = 12, nf4_vram = 6,
                          max_fp16 = px(1024), max_nf4 = px(1024),
                          max_cpu = px(768)),
         # 12B: nf4 peaks ~9.6 GB at 1024^2, fp8 ~12 GB resident,
         # bf16 needs a 24 GB card. attn-chunk the tight nf4 tier.
         flux1 = .flux_family_tiers(bf16_vram = 24, fp8_vram = 14,
                                    nf4_vram = 10, nf4_tight_vram = 8,
                                    max_hi = px(1536), max_mid = px(1024),
                                    max_lo = px(768), max_cpu = px(512),
                                    attn_tight = 2048L, gpu_encode_vram = 14),
         # 4B but activation-heavy: 1024^2 peaks ~12.5 GB regardless of
         # weight precision, so the 1024^2 tiers want ~13 GB free.
         flux2 = .flux_family_tiers(bf16_vram = 16, fp8_vram = 14,
                                    nf4_vram = 13, nf4_tight_vram = 8,
                                    max_hi = px(1024), max_mid = px(1024),
                                    max_lo = px(768), max_cpu = px(512)),
         # 6B: 1024^2 peaks ~13.1 GB, 512^2 ~half that.
         zimage = .flux_family_tiers(bf16_vram = 18, fp8_vram = 14,
                                     nf4_vram = 13, nf4_tight_vram = 8,
                                     max_hi = px(1024), max_mid = px(1024),
                                     max_lo = px(512), max_cpu = px(512)),
         # 22B video. fp8 is CPU-resident and streamed here (unlike the
         # image models); nf4 keeps the transformer resident. Coarse -
         # ltx23_memory_profile does the frame-aware sizing.
         ltx = list(
                    list(precision = "nf4", min_vram = 14, needs = NULL,
                         devices = .dev_flux(TRUE), offload = TRUE,
                         max_pixels = px(1280), text_device = "cpu",
                         attn_chunk = NULL),
                    list(precision = "fp8", min_vram = 10,
                         needs = "float8_e4m3fn", devices = .dev_flux(TRUE),
                         offload = TRUE, max_pixels = px(1024),
                         text_device = "cpu", attn_chunk = 4096L),
                    list(precision = "nf4", min_vram = 0, needs = NULL,
                         cpu = TRUE, devices = .dev_flux(FALSE),
                         offload = FALSE, max_pixels = px(512),
                         text_device = "cpu", attn_chunk = NULL)
        )
    )
}
