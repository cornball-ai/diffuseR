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
# stage_from = "disk" covers the encoder too, which only Gemma 4 can do
expect_error(resident_load("ltx", stage_from = "disk",
                           checkpoint_path = tempdir(),
                           text_encoder = not_gemma4, tokenizer = not_gemma4),
             "Gemma 4")

# pinned_bytes counts the streamed sets that live on the modules
if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
    exit_file("torch not fully installed")
}
pair <- function(n) {
    list(live = NULL, pinned = torch::torch_zeros(n, dtype = torch::torch_bfloat16()))
}
tr <- torch::nn_linear(2, 2)
tr$block_stream <- list(n_resident = 1L, staging = list(list(pair(10)), list(pair(6))))
con <- torch::nn_module(initialize = function() {
    self$video_connector <- torch::nn_linear(2, 2)
    self$audio_connector <- torch::nn_linear(2, 2)
})()
con$unit_stream <- list(video_text_proj_in = list(pair(4)))
con$video_connector$layer_stream <- list(list(pair(3)))
con$audio_connector$layer_stream <- list(list(pair(2), list(live = NULL, pinned = NULL)))
bytes <- diffuseR:::.resident_stream_bytes(list(transformer = tr, connectors = con))
expect_equal(bytes, 2 * (10 + 6 + 4 + 3 + 2))
expect_equal(diffuseR:::.resident_stream_bytes(list(transformer = torch::nn_linear(2, 2))), 0)
