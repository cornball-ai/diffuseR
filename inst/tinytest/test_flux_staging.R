# Flux-family pinned staging surface + gate logic. The CPU/no-torch
# parts (formals, .resolve_pin precedence, .flux_build_staging opt-out
# paths) run everywhere; the pin round trip is CUDA-gated below.

library(diffuseR)

# --- loader surface: pin + text_device knobs --------------------------------------

for (fn in list(flux_load_pipeline, flux2_load_pipeline,
                zimage_load_pipeline)) {
    expect_true("pin" %in% names(formals(fn)))
    expect_true("text_device" %in% names(formals(fn)))
}
# flux1's text_device default moved from "cpu" to NULL (profile-resolved)
expect_null(eval(formals(flux_load_pipeline)$text_device))

# --- .resolve_pin precedence: explicit > option > decision ------------------------

# Explicit argument wins over everything.
old <- getOption("diffuseR.pin_staging")
on.exit(options(diffuseR.pin_staging = old), add = TRUE)
options(diffuseR.pin_staging = FALSE)
expect_true(diffuseR:::.resolve_pin(TRUE, "flux1", "nf4"))
expect_false(diffuseR:::.resolve_pin(FALSE, "flux1", "nf4"))
# NULL falls through to the option.
expect_false(diffuseR:::.resolve_pin(NULL, "flux1", "nf4"))
options(diffuseR.pin_staging = TRUE)
expect_true(diffuseR:::.resolve_pin(NULL, "flux1", "nf4"))
# Unset option falls through to the RAM-aware decision (logical result).
options(diffuseR.pin_staging = NULL)
expect_true(is.logical(diffuseR:::.resolve_pin(NULL, "flux2", "fp8")))

# --- .pin_decision: cpu -> FALSE, NA RAM -> TRUE, else 2x rule ---------------------

expect_false(diffuseR:::.pin_decision("flux1", "nf4", host_ram_gb = 64,
                                      cpu = TRUE))
expect_true(diffuseR:::.pin_decision("flux1", "nf4", host_ram_gb = NA))
expect_true(diffuseR:::.pin_decision("flux1", "nf4", host_ram_gb = 128))
expect_false(diffuseR:::.pin_decision("flux1", "nf4", host_ram_gb = 8))

# --- .pinned_set_gb flux1 row: T5 host copy now bf16 (was fp32) --------------------

expect_equal(diffuseR:::.pinned_set_gb("flux1", "nf4"), 17)
expect_equal(diffuseR:::.pinned_set_gb("flux1", "fp8"), 22)
expect_equal(diffuseR:::.pinned_set_gb("flux1", "bf16"), 34)

# --- flux tier text_device: flux1 GPU-encodes only where T5 fits ------------------

# fp8/bf16 tiers (>=14 GB) onload T5; nf4 tiers keep CPU-fp32 encode.
#
# st_caps is pinned explicitly here. Left NULL, recommend() probes the
# *installed* safetensors, so the fp8 tier only exists on a machine whose
# build can read float8 -- which made this block pass on a dev box with
# the fork and fail on Windows R-devel, and would have failed on CRAN,
# where flux1 at 16 GB resolves to nf4 and its CPU encode instead.
st_fork <- list(bfloat16 = TRUE, float8_e4m3fn = TRUE)
st_cran <- list(bfloat16 = TRUE, float8_e4m3fn = FALSE)
expect_equal(recommend("flux1", vram_gb = 24, st_caps = st_fork)$text_device,
             "cuda")
expect_equal(recommend("flux1", vram_gb = 16, st_caps = st_fork)$text_device,
             "cuda") # fp8 tier
expect_equal(recommend("flux1", vram_gb = 16, st_caps = st_cran)$text_device,
             "cpu") # same card, stock safetensors: nf4 tier
expect_equal(recommend("flux1", vram_gb = 10, st_caps = st_fork)$text_device,
             "cpu") # nf4 tier
expect_equal(recommend("flux1", vram_gb = 0, st_caps = st_fork)$text_device,
             "cpu") # cpu tier
# The small flux2/zimage encoder rides the GPU on every GPU tier,
# whichever safetensors is installed.
expect_equal(recommend("flux2", vram_gb = 16, st_caps = st_fork)$text_device,
             "cuda")
expect_equal(recommend("flux2", vram_gb = 16, st_caps = st_cran)$text_device,
             "cuda")

# --- .flux_build_staging opt-out paths return NULL (no torch needed) --------------

dummy <- list(format = "nf4")
expect_null(diffuseR:::.flux_build_staging(dummy, pin = FALSE,
    phase_offload = TRUE, device = "cuda", components = "transformer"))
expect_null(diffuseR:::.flux_build_staging(dummy, pin = TRUE,
    phase_offload = FALSE, device = "cuda", components = "transformer"))
expect_null(diffuseR:::.flux_build_staging(dummy, pin = TRUE,
    phase_offload = TRUE, device = "cpu", components = "transformer"))

# --- fp8 field collector + pin round trip (torch / CUDA gated) --------------------

if (!requireNamespace("torch", quietly = TRUE) ||
    !torch::torch_is_installed()) {
    exit_file("torch not fully installed")
}

# .flux_fp8_collect walks children and returns weight_fp8 + weight_scale
# pairs. Build two fp8 linears under a parent and check the count.
parent <- torch::nn_module(
    "fp8_parent",
    initialize = function() {
        self$a <- ltx23_fp8_linear(4L, 4L, bias = FALSE)
        self$b <- ltx23_fp8_linear(4L, 4L, bias = FALSE)
    },
    forward = function(x) x
)()
w <- torch::torch_randn(4L, 4L)$to(dtype = torch::torch_float8_e4m3fn())
s <- torch::torch_tensor(0.5)
parent$a$set_fp8_weight(w, s)
parent$b$set_fp8_weight(w, s)
collected <- diffuseR:::.flux_fp8_collect(parent)
expect_equal(length(collected), 4L)   # 2 linears x (weight_fp8, weight_scale)

if (!torch::cuda_is_available()) {
    exit_file("no CUDA")
}

# .pin_component with extra fields pins params/buffers AND the fp8 fields;
# staged onload/offload moves them all and preserves the forward output.
m <- ltx23_feed_forward(16L)
m$eval()
extra_t <- torch::torch_randn(8L, 8L)$to(dtype = torch::torch_float8_e4m3fn())
scale0 <- torch::torch_tensor(2.0)   # 0-dim scale must pin too
st <- diffuseR:::.pin_component(m, extra = list(extra_t, scale0))
expect_false(is.null(st))
# params + buffers + 2 extras all became pinned live/pinned pairs
expect_true(length(st) >= length(c(m$parameters, m$buffers)) + 2L)
expect_true(suppressWarnings(
    st[[length(st)]]$live$is_pinned(device = torch::torch_device("cuda"))
))

x <- torch::torch_randn(2L, 5L, 16L)
torch::with_no_grad(ref <- m(x))
diffuseR:::.staged_onload(st, "cuda")
expect_equal(st[[1]]$live$device$type, "cuda")
torch::with_no_grad(out_gpu <- m(x$to(device = "cuda"))$cpu())
expect_true(as.numeric((out_gpu - ref)$abs()$max()) < 1e-5)
diffuseR:::.staged_offload(st)
expect_equal(st[[1]]$live$device$type, "cpu")
torch::with_no_grad(out_back <- m(x))
expect_true(as.numeric((out_back - ref)$abs()$max()) == 0)
