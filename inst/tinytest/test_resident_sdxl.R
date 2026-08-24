# SDXL residency, and the two fixes it needed on the way in.
#
# None of this requires a GPU or the 7 GB of weights: the staging layer is
# already covered by test_resident.R and test_staging.R, and what is new
# here is dispatch, the device/dtype resolution that a supplied pipeline
# takes, and the NF4 scratch release on deactivate.

library(tinytest)
library(diffuseR)

# --- dispatch ---------------------------------------------------------------------

expect_true("sdxl" %in% diffuseR:::.resident_families)

# The match.arg default has to list it too, or resident_load("sdxl")
# is refused before any of the above matters. Read off the formals rather
# than restated, so the two cannot drift apart.
expect_true("sdxl" %in% eval(formals(resident_load)$model))
expect_equal(sort(eval(formals(resident_load)$model)),
             sort(diffuseR:::.resident_families))

expect_true(is.function(sdxl_load_pipeline))

# --- the loader shim ---------------------------------------------------------------

fm <- formals(sdxl_load_pipeline)
expect_true(all(c("model_dir", "device", "unet_dtype", "phase_offload",
                  "verbose") %in% names(fm)))

# model_dir defaults to NULL so the download_sdxl() cache is resolved.
expect_null(fm$model_dir)

# SDXL has no phased path, so a resident handle must transfer on activate
# rather than treat activation as a claim. This default is what
# resident_activate() reads to decide.
expect_false(fm$phase_offload)

# --- the fit check charges only what is actually going to the card ------------------

# SDXL pins 8.0 GB and sends 5.1 GB of it. Charging the full pinned figure
# would refuse activations that fit.
h <- new.env(parent = emptyenv())
h$model <- "sdxl"
h$pinned_bytes <- 8 * 1024^3
expect_error(diffuseR:::.resident_check_fits(structure(h, class = "diffuseR_resident"),
                                             free_gb = 6),
             pattern = "needs 8")
expect_true(diffuseR:::.resident_check_fits(structure(h, class = "diffuseR_resident"),
                                            free_gb = 6,
                                            need_bytes = 5.1 * 1024^3))

# --- dtype resolution ---------------------------------------------------------------

# resident_load() binds an explicit "cuda:N" so later transitions cannot
# drift, and passes it through as the component device. Before the fix that
# matched neither branch of setup_dtype() and hit "Invalid device".
expect_equal(diffuseR:::setup_dtype(list(unet = "cuda:0"), NULL),
             torch::torch_float16())
expect_equal(diffuseR:::setup_dtype(list(unet = "cuda:3"), NULL),
             torch::torch_float16())
# Unqualified devices keep their existing meaning.
expect_equal(diffuseR:::setup_dtype(list(unet = "cuda"), NULL),
             torch::torch_float16())
expect_equal(diffuseR:::setup_dtype(list(unet = "cpu"), NULL),
             torch::torch_float32())
# An explicit dtype still wins over the device's default.
expect_equal(diffuseR:::setup_dtype(list(unet = "cuda:0"), "float32"),
             torch::torch_float32())
# A genuinely unknown device is still refused.
expect_error(diffuseR:::setup_dtype(list(unet = "tpu"), NULL),
             pattern = "Invalid device")

# --- device config without the file check --------------------------------------------

# models2devices() ends by verifying TorchScript .pt files. A native
# safetensors pipeline never reads one, so a caller holding a pipeline takes
# the path that skips it. Same four fields, no disk.
d <- diffuseR:::.devices_for_pipeline("sdxl", "cuda:0", NULL)
expect_equal(sort(names(d)),
             c("device_cpu", "device_cuda", "devices", "unet_dtype"))
expect_equal(d$unet_dtype, torch::torch_float16())
expect_true(all(c("unet", "decoder", "text_encoder", "text_encoder2") %in%
                names(d$devices)))
expect_equal(d$devices$unet, "cuda:0")

# A named list is carried through, not flattened to one device.
d2 <- diffuseR:::.devices_for_pipeline(
    "sdxl", list(unet = "cuda", decoder = "cpu",
                 text_encoder = "cpu", text_encoder2 = "cpu"), NULL)
expect_equal(d2$devices$unet, "cuda")
expect_equal(d2$devices$decoder, "cpu")
# dtype follows the UNet, not the other components.
expect_equal(d2$unet_dtype, torch::torch_float16())

# --- generate-time device injection ---------------------------------------------------

sdxl_names <- c("unet", "decoder", "text_encoder", "text_encoder2")

mk_h <- function(model, device = "cuda:0", gpu = "unet",
                 comps = sdxl_names) {
    e <- new.env(parent = emptyenv())
    e$model <- model
    e$device <- device
    e$gpu_components <- gpu
    # .resident_gpu_set() intersects against the staging set, so the fake
    # handle needs one; the values are never read.
    e$staging <- stats::setNames(vector("list", length(comps)), comps)
    structure(e, class = "diffuseR_resident")
}

# --- which components go to the card ---------------------------------------------

# NULL means everything, which is what the other families want.
expect_equal(sort(diffuseR:::.resident_gpu_set(mk_h("flux2", gpu = NULL))),
             sort(sdxl_names))
expect_equal(diffuseR:::.resident_gpu_set(mk_h("sdxl")), "unet")
# A named component that was never pinned is dropped rather than onloaded.
expect_equal(diffuseR:::.resident_gpu_set(mk_h("sdxl", gpu = c("unet", "nope"))),
             "unet")

inj <- diffuseR:::.resident_gen_args(mk_h("sdxl"), list())
expect_true(!is.null(inj$devices))
# The list is keyed by what standardize_devices() requires, not by what the
# pipeline builds: SDXL's required set also names an `encoder` (the VAE
# encoder img2img uses) that the text-to-image pipeline never constructs.
expect_equal(sort(names(inj$devices)),
             sort(diffuseR:::get_required_components("sdxl")))
expect_true(all(c("unet", "decoder", "text_encoder", "text_encoder2") %in%
                names(inj$devices)))

# Only the UNet is on the card: bulk-onloading all four fits the weights and
# then OOMs in the fp32 VAE decode. The encoders and decoder compute on the
# host from their pinned copies.
expect_equal(inj$devices$unet, "cuda:0")
expect_equal(inj$devices$decoder, "cpu")
expect_equal(inj$devices$text_encoder, "cpu")
expect_equal(inj$devices$text_encoder2, "cpu")

# The bound ordinal is carried, so a handle on the second card does not
# quietly render on the first.
expect_equal(diffuseR:::.resident_gen_args(mk_h("sdxl", "cuda:1"),
                                           list())$devices$unet, "cuda:1")

# The placement follows gpu_components rather than being hard-coded, so a
# roomier card can be given the decoder too without touching this logic.
wide <- diffuseR:::.resident_gen_args(
    mk_h("sdxl", gpu = c("unet", "decoder")), list())
expect_equal(wide$devices$decoder, "cuda:0")
expect_equal(wide$devices$text_encoder, "cpu")

# An explicit devices= from the caller is a decision, not a gap to fill.
keep <- list(devices = list(unet = "cpu"))
expect_equal(diffuseR:::.resident_gen_args(mk_h("sdxl"), keep)$devices,
             list(unet = "cpu"))

# Other arguments ride through untouched.
o <- diffuseR:::.resident_gen_args(mk_h("sdxl"), list(seed = 7L))
expect_equal(o$seed, 7L)

# The phase-offloading families place components themselves; injecting a
# device list for them would fight their own per-phase movement.
for (m in c("flux1", "flux2", "zimage", "ltx")) {
    expect_null(diffuseR:::.resident_gen_args(mk_h(m), list())$devices)
}

# --- NF4 dequant scratch is released on deactivate -------------------------------------

# NF4 linears dequantize into a package-level environment rather than into
# the module, so offloading the weights does not free it, and
# txt2vid_ltx2() deliberately skips its own release while the transformer is
# resident. Nothing else reclaims it: the environment still holds a
# reference, so gc() and cuda_empty_cache() cannot.
mk_d <- function(model) {
    e <- new.env(parent = emptyenv())
    e$model <- model
    e$device <- "cuda:0"
    e$state <- "active"
    e$staging <- list()
    e$pipeline <- list()
    e$last_error <- NULL
    structure(e, class = "diffuseR_resident")
}

buf <- diffuseR:::.ltx23_dequant_buffers
assign("probe", 1L, envir = buf)
expect_true("probe" %in% ls(buf))
resident_deactivate(mk_d("ltx"), release = FALSE)
expect_equal(length(ls(buf)), 0L)

# Unconditional on `release`: a broker that passes release = FALSE to keep
# the pool warm for the next tenant is precisely the caller that must not
# be handed a budget short by this scratch.
assign("probe2", 1L, envir = buf)
resident_deactivate(mk_d("ltx"), release = TRUE)
expect_equal(length(ls(buf)), 0L)

# Only LTX skips the release inside its own generate, so only LTX needs it
# here. The image families already clear it at the end of every render, and
# clearing it for them would be reaching into another family's business.
assign("probe3", 1L, envir = buf)
resident_deactivate(mk_d("flux2"), release = FALSE)
expect_true("probe3" %in% ls(buf))
rm("probe3", envir = buf)

# --- allocator pre-warm ----------------------------------------------------------------

# Best-effort by contract: it is an optimisation, and a card that cannot
# seat the block in one piece must fall back to the per-tensor path rather
# than fail the activation.
expect_silent(diffuseR:::.resident_prewarm(0, "cuda"))
expect_silent(diffuseR:::.resident_prewarm(-1, "cuda"))
expect_silent(diffuseR:::.resident_prewarm(NA_real_, "cuda"))
expect_silent(diffuseR:::.resident_prewarm(NULL, "cuda"))
# An impossible size on a real card must be swallowed, not raised.
expect_silent(diffuseR:::.resident_prewarm(1e18, "cuda"))
expect_equal(diffuseR:::.resident_prewarm(0, "cuda"), 0)

# It must GROW the pool, not re-request it. Asking for the full figure on
# every activation doubled the cache once a render had fragmented it: the
# single large request could not be served from cache and took a fresh
# cudaMalloc beside the old block. Under release = FALSE -- which a
# residency broker passes deliberately -- nothing empties the cache, so
# SDXL went 5.299 -> 10.322 GiB and the third activation was refused.
gb <- 1024^3

# Pool already covers the transfer: ask for nothing at all.
expect_equal(diffuseR:::.resident_prewarm(4 * gb, "cuda", held = 5 * gb), 0)
# Exactly equal still counts as covered.
expect_equal(diffuseR:::.resident_prewarm(4 * gb, "cuda", held = 4 * gb), 0)

# Pool short: ask for the SHORTFALL, not the whole need. Requesting the
# whole need here is precisely the bug.
expect_equal(diffuseR:::.resident_prewarm(4 * gb, "cuda", held = 3 * gb),
             1 * gb * 1.05)

# Cold pool: byte-identical to the original behaviour, so the 74x
# cold-start win is untouched.
expect_equal(diffuseR:::.resident_prewarm(4 * gb, "cuda", held = 0),
             4 * gb * 1.05)

# A nonsense reading must not be trusted into a negative request.
expect_equal(diffuseR:::.resident_prewarm(4 * gb, "cuda", held = NA_real_),
             4 * gb * 1.05)
expect_equal(diffuseR:::.resident_prewarm(4 * gb, "cuda", held = -1),
             4 * gb * 1.05)
