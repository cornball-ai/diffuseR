# resident_load(stream_text = TRUE) streams a Gemma 4 encoder, so it is
# refused before any loading when there is no Gemma 4 artifact to stream.
# These checks run ahead of the CUDA check and need no GPU.

library(tinytest)
library(diffuseR)

expect_true("stream_text" %in% names(formals(resident_load)))
expect_false(formals(resident_load)$stream_text)

expect_error(resident_load("ltx", stream_text = TRUE,
                           checkpoint_path = tempdir()),
             "Gemma 4")
not_gemma4 <- file.path(tempdir(), "not-gemma4")
dir.create(not_gemma4, showWarnings = FALSE)
expect_error(resident_load("ltx", stream_text = TRUE,
                           checkpoint_path = tempdir(),
                           text_encoder = not_gemma4, tokenizer = not_gemma4),
             "Gemma 4")
expect_error(resident_load("flux2", stream_text = TRUE), "Gemma 4")
