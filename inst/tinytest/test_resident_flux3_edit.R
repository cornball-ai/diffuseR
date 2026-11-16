# resident_generate() on a flux3 handle routes an `image` to the
# single-reference edit; without one, and for every other family, the
# text generator stands. No model, no GPU.

library(tinytest)
library(diffuseR)

pick <- diffuseR:::.resident_generator
mk <- function(model, encoder = TRUE) {
    h <- new.env(parent = emptyenv())
    h$model <- model
    h$state <- "active"
    h$pipeline <- if (encoder) list(encoder = "vae-encoder") else list()
    class(h) <- "diffuseR_resident"
    h
}

expect_identical(pick(mk("flux3"), list()), txt2img_flux3)
expect_identical(pick(mk("flux3"), list(seed = 7L)), txt2img_flux3)
expect_identical(pick(mk("flux3"), list(image = "ref.png", seed = 7L)),
                 img2img_flux3)
expect_error(pick(mk("flux3", encoder = FALSE), list(image = "ref.png")),
             "VAE encoder")
# an image on the video family is its start frame, untouched
expect_identical(pick(mk("ltx"), list(image = "frame.png")), txt2vid_ltx2)
expect_identical(pick(mk("flux2"), list()), txt2img_flux2)
expect_identical(pick(mk("zimage"), list()), txt2img_zimage)

# the state check comes before the routing
h <- mk("flux3")
h$state <- "inactive"
expect_error(resident_generate(h, "p", image = "ref.png"),
             pattern = "resident_activate")
