# Residency contract: component discovery, byte accounting, the state
# machine, and the pinned<->GPU round trip.
#
# The state-machine tests build a handle by hand around synthetic
# nn_modules, so they exercise the transitions without loading a real
# multi-GB pipeline. The round-trip test needs CUDA and is skipped
# without it.

library(tinytest)
library(diffuseR)

fake_pipeline <- function() {
    structure(list(transformer = torch::nn_linear(8, 8),
                   decoder = torch::nn_linear(8, 4),
                   text_encoder = torch::nn_linear(4, 8),
                   config = list(family = "test"),
                   scheduler = "flowmatch",
                   phase_offload = TRUE),
              class = "test_pipeline")
}

# Everything from here to the "state machine" section builds real
# nn_modules or torch dtypes, so it needs a working lantern. torch
# installed WITHOUT its lantern binaries is the normal state on
# win-builder and CRAN, and there `torch::nn_linear()` errors rather
# than returning. The state-machine sections below are pure R and run
# everywhere.
have_torch <- requireNamespace("torch", quietly = TRUE) &&
    torch::torch_is_installed()

# --- component discovery ----------------------------------------------------------

if (have_torch) {

pipe <- fake_pipeline()
comps <- diffuseR:::.resident_components(pipe)
expect_equal(sort(names(comps)), c("decoder", "text_encoder", "transformer"))
# Non-module fields are not components.
expect_false("config" %in% names(comps))
expect_false("scheduler" %in% names(comps))
expect_false("phase_offload" %in% names(comps))

# A pipeline with no modules yields an empty set, not an error.
expect_equal(length(diffuseR:::.resident_components(list(a = 1, b = "x"))), 0L)

# --- dtype byte table -------------------------------------------------------------

expect_equal(diffuseR:::.dtype_bytes(torch::torch_float32()), 4)
expect_equal(diffuseR:::.dtype_bytes(torch::torch_float16()), 2)
expect_equal(diffuseR:::.dtype_bytes(torch::torch_bfloat16()), 2)
expect_equal(diffuseR:::.dtype_bytes(torch::torch_uint8()), 1)
expect_equal(diffuseR:::.dtype_bytes(torch::torch_int64()), 8)

} # end have_torch

# --- byte formatting --------------------------------------------------------------

expect_equal(diffuseR:::.fmt_gb(0), "0 GB")
expect_equal(diffuseR:::.fmt_gb(NA), "0 GB")
expect_equal(diffuseR:::.fmt_gb(NULL), "0 GB")
expect_equal(diffuseR:::.fmt_gb(1024^3), "1.00 GB")

# --- state guard ------------------------------------------------------------------

mk <- function(state, staging = list(), phase_offload = FALSE,
               pinned_bytes = 0, store_field = TRUE) {
    e <- new.env(parent = emptyenv())
    e$model <- "flux2"
    e$device <- "cuda:0"
    e$state <- state
    e$staging <- staging
    e$components <- character(0)
    e$pinned_bytes <- pinned_bytes
    e$last_error <- NULL
    e$loaded_at <- Sys.time()
    # store_field = FALSE mimics LTX, which never records phase_offload
    # on the pipeline object.
    e$pipeline <- if (store_field) list(phase_offload = phase_offload) else list()
    e$phase_offload <- phase_offload
    structure(e, class = "diffuseR_resident")
}

# A pipeline that does not store the field must still be read as
# phase-offloading (this is what sent LTX down the bulk path and made it
# demand 18.41 GB on a 15 GB card).
expect_true(isTRUE(mk("inactive", phase_offload = TRUE,
                      store_field = FALSE)$phase_offload))

expect_error(diffuseR:::.resident_guard(mk("unloaded"), "activate"),
             pattern = "unloaded")
expect_error(diffuseR:::.resident_guard(mk("broken"), "activate"),
             pattern = "broken")
expect_true(diffuseR:::.resident_guard(mk("inactive"), "activate"))
expect_true(diffuseR:::.resident_guard(mk("active"), "generate"))

# --- illegal transitions ----------------------------------------------------------

# Deactivating something that was never activated is an error, not a
# silent no-op, unless it is already inactive.
expect_silent(resident_deactivate(mk("inactive")))
expect_error(resident_deactivate(mk("activating")), pattern = "cannot deactivate")
expect_error(resident_activate(mk("deactivating")), pattern = "cannot activate")

# Activate on an already-active handle is idempotent.
expect_silent(resident_activate(mk("active")))

# Generation is refused unless active.
expect_error(resident_generate(mk("inactive"), "a cat"),
             pattern = "resident_activate")
expect_error(resident_generate(mk("broken"), "a cat"), pattern = "broken")

# --- a set that cannot fit is refused before any transfer -------------------------

# Named figures beat a libtorch allocator error: FLUX.1's 15.73 GB pinned
# set does not fit a 15.47 GiB card, and the message should say so.
#
# free_gb is stated rather than measured. Measured, it is 0 on any
# machine without a GPU -- which means "cannot tell", so the check
# declines to refuse and these assertions saw no error. That is correct
# behaviour and a broken test: it passed on the GPU box and failed CI on
# both ubuntu-latest and macos-latest.
huge <- mk("inactive", pinned_bytes = 16 * 1024^3)
expect_error(diffuseR:::.resident_check_fits(huge, free_gb = 15.47),
             pattern = "needs")
expect_error(diffuseR:::.resident_check_fits(huge, free_gb = 15.47),
             pattern = "phase_offload")
# A set that fits is not refused.
expect_true(diffuseR:::.resident_check_fits(huge, free_gb = 24))
# A trivial set is never refused.
expect_true(diffuseR:::.resident_check_fits(mk("inactive", pinned_bytes = 0),
                                            free_gb = 15.47))
# Undetectable VRAM (0, as on a GPU-less machine) must not block: the
# check refuses only when it can prove the set does not fit.
expect_true(diffuseR:::.resident_check_fits(huge, free_gb = 0))
expect_true(diffuseR:::.resident_check_fits(huge, free_gb = NA_real_))

# --- status and print -------------------------------------------------------------

h <- mk("inactive")
s <- resident_status(h)
expect_true(all(c("model", "state", "device", "components",
                  "components_on_gpu", "pinned_bytes", "gpu_allocated",
                  "gpu_reserved", "loaded_at", "last_error") %in% names(s)))
# An empty staging set has nothing resident.
expect_equal(s$components_on_gpu, 0L)
# The allocator numbers must come from a torch function that actually
# exists: cuda_memory_allocated() does not (R CMD check catches it, but
# only as a WARNING buried in the dependencies step).
expect_true("cuda_memory_stats" %in% getNamespaceExports("torch"))
mem <- diffuseR:::.cuda_bytes()
expect_true(all(c("allocated", "reserved") %in% names(mem)))
expect_equal(s$state, "inactive")
expect_null(s$last_error)

out <- capture.output(print(h))
expect_true(any(grepl("diffuseR resident", out)))
expect_true(any(grepl("flux2", out)))

# --- unload is terminal and idempotent --------------------------------------------

u <- mk("inactive")
resident_unload(u)
expect_equal(u$state, "unloaded")
expect_silent(resident_unload(u))
expect_error(resident_activate(u), pattern = "unloaded")
# Status still works on an unloaded handle.
expect_equal(resident_status(u)$state, "unloaded")

# --- CUDA round trip --------------------------------------------------------------

if (have_torch && at_home() && torch::cuda_is_available()) {
    pipe <- fake_pipeline()
    staging <- diffuseR:::.resident_pin(pipe, verbose = FALSE)
    expect_equal(sort(names(staging)),
                 c("decoder", "text_encoder", "transformer"))

    # Pinning leaves everything on the host.
    expect_true(diffuseR:::.resident_all_on(staging, "cpu"))
    expect_true(diffuseR:::.resident_pinned_bytes(staging) > 0)

    res <- mk("inactive", staging)
    res$components <- names(staging)

    resident_activate(res)
    expect_equal(res$state, "active")
    expect_true(diffuseR:::.resident_all_on(staging, "cuda"))
    expect_equal(resident_status(res)$components_on_gpu, 3L)

    resident_deactivate(res)
    expect_equal(res$state, "inactive")
    expect_true(diffuseR:::.resident_all_on(staging, "cpu"))
    expect_equal(resident_status(res)$components_on_gpu, 0L)

    # components_on_gpu is ground truth, not the declared state: evict
    # behind the handle's back and status must notice while state still
    # says "active". This is what a phase-offloading render does.
    resident_activate(res)
    for (nm in names(staging)) diffuseR:::.staged_offload(staging[[nm]])
    expect_equal(res$state, "active")
    expect_equal(resident_status(res)$components_on_gpu, 0L)
    resident_deactivate(res)

    # A second round trip reuses the same pinned buffers.
    resident_activate(res)
    expect_true(diffuseR:::.resident_all_on(staging, "cuda"))
    resident_deactivate(res)
    expect_true(diffuseR:::.resident_all_on(staging, "cpu"))

    # The module still computes correctly after the round trip.
    y <- torch::with_no_grad(pipe$transformer(torch::torch_randn(c(2, 8))))
    expect_equal(as.integer(y$shape), c(2L, 8L))

    resident_unload(res)
    expect_equal(res$state, "unloaded")
}
