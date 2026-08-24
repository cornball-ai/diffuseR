# SD 2.1 residency. The sibling of test_resident_sdxl.R, and mostly about
# the two places SD 2.1 disagrees with SDXL: the UNet dtype, and a
# component set that names an `encoder` the pipeline never builds.

library(tinytest)
library(diffuseR)

# --- dispatch ---------------------------------------------------------------------

expect_true("sd21" %in% diffuseR:::.resident_families)
expect_true("sd21" %in% eval(formals(resident_load)$model))
expect_equal(sort(eval(formals(resident_load)$model)),
             sort(diffuseR:::.resident_families))
expect_true(is.function(sd21_load_pipeline))

fm <- formals(sd21_load_pipeline)
expect_true(all(c("model_dir", "device", "unet_dtype", "phase_offload",
                  "verbose") %in% names(fm)))
expect_null(fm$model_dir)
expect_false(fm$phase_offload)
# NULL means the loader decides, and it must decide float32 -- see below.
expect_null(fm$unet_dtype)

# --- generate-time device injection -------------------------------------------------

mk_h <- function(model, device = "cuda:0", gpu = "unet") {
    e <- new.env(parent = emptyenv())
    e$model <- model
    e$device <- device
    e$gpu_components <- gpu
    comps <- if (identical(model, "sd21")) {
        c("unet", "decoder", "text_encoder")
    } else {
        c("unet", "decoder", "text_encoder", "text_encoder2")
    }
    e$staging <- stats::setNames(vector("list", length(comps)), comps)
    structure(e, class = "diffuseR_resident")
}

inj <- diffuseR:::.resident_gen_args(mk_h("sd21"), list())

# The device list has to satisfy standardize_devices(), which requires
# every component get_required_components() names -- including `encoder`,
# the VAE encoder img2img needs and the text-to-image pipeline never
# builds. Omitting it is "Missing required component: encoder".
expect_equal(sort(names(inj$devices)),
             sort(diffuseR:::get_required_components("sd21")))
expect_true("encoder" %in% names(inj$devices))

# Only the UNet is on the card: SD 2.1's float32 denoise wants ~11.4 GB at
# 768 on its own, so the weights cannot all sit beside it.
expect_equal(inj$devices$unet, "cuda:0")
expect_equal(inj$devices$decoder, "cpu")
expect_equal(inj$devices$text_encoder, "cpu")
# The phantom encoder follows the decoder, which is where it would live.
expect_equal(inj$devices$encoder, "cpu")

# ... and it keeps following the decoder when the decoder is on the card,
# rather than being pinned to one answer.
wide <- diffuseR:::.resident_gen_args(
    mk_h("sd21", gpu = c("unet", "decoder")), list())
expect_equal(wide$devices$decoder, "cuda:0")
expect_equal(wide$devices$encoder, "cuda:0")
expect_equal(wide$devices$text_encoder, "cpu")

# The injected list must actually survive the function it is built for.
expect_silent(diffuseR:::standardize_devices(
    inj$devices, diffuseR:::get_required_components("sd21")))

# An explicit devices= from the caller still wins.
expect_equal(diffuseR:::.resident_gen_args(mk_h("sd21"),
                                           list(devices = list(unet = "cpu")))$devices,
             list(unet = "cpu"))

# SDXL keeps its own four-component shape, no encoder.
sx <- diffuseR:::.resident_gen_args(mk_h("sdxl"), list())
expect_true("text_encoder2" %in% names(sx$devices))
expect_equal(sort(names(sx$devices)),
             sort(diffuseR:::get_required_components("sdxl")))

# --- the float16 trap ----------------------------------------------------------------

# SD 2.1's attention overflows in float16 and the pipeline returns all-NaN
# rather than raising, so a float16 resident would load, activate, generate
# and hand back a blank image with nothing to catch. The resident loader
# fixes the dtype before the weights are pinned, so this is the only place
# the decision can be made.
#
# setup_dtype() would answer float16 for a CUDA device, which is why
# sd21_load_pipeline does NOT delegate to it.
expect_equal(diffuseR:::setup_dtype(list(unet = "cuda:0"), NULL),
             torch::torch_float16())

# Assert the loader's own choice against real weights when they are here.
# Skipped in R CMD check and on a machine without the model.
if (at_home()) {
    dir <- file.path(tools::R_user_dir("diffuseR", "data"), "sd21-diffusers")
    if (dir.exists(file.path(dir, "unet"))) {
        p <- sd21_load_pipeline(model_dir = dir, device = "cuda",
                                verbose = FALSE)
        expect_equal(as.character(p$unet$parameters[[1]]$dtype), "Float")
        expect_equal(p$gpu_components, "unet")
        expect_false(p$phase_offload)
    }
}
