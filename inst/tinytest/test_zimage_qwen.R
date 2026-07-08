# Z-Image Qwen3 delta: the enable_thinking=TRUE chat template against
# shipped-tokenizer renders (tools/gen_zimage_qwen_template_cases.py) and
# the hidden_states[-2] + mask-slice convention against a tiny reference
# model (tools/gen_fixtures_zimage_qwen.py).

if (!requireNamespace("torch", quietly = TRUE) || !torch::torch_is_installed()) {
  exit_file("torch not fully installed")
}
if (!requireNamespace("safetensors", quietly = TRUE)) {
  exit_file("safetensors not installed")
}

library(diffuseR)

max_abs_diff <- function(a, b) {
  as.numeric(torch::torch_max(torch::torch_abs(
    a$to(dtype = torch::torch_float32()) - b$to(dtype = torch::torch_float32())
  )))
}

# --- hidden_states[-2] + mask slicing (tiny reference model) ----------------------

fixture_path <- system.file("tinytest", "fixtures", "zimage_qwen.safetensors",
  package = "diffuseR")
if (fixture_path == "") fixture_path <- "fixtures/zimage_qwen.safetensors"
if (!file.exists(fixture_path)) exit_file("zimage qwen fixtures missing")

fx <- safetensors::safe_load_file(fixture_path, framework = "torch")

sd <- fx[grep("^sd\\.", names(fx))]
names(sd) <- paste0("model.", sub("^sd\\.", "", names(sd)))

enc <- qwen3_encoder(
  vocab_size = 128L, hidden_size = 32L, intermediate_size = 64L,
  num_hidden_layers = 4L, num_attention_heads = 4L,
  num_key_value_heads = 2L, head_dim = 8L, rope_theta = 1e6,
  rms_norm_eps = 1e-6
)
dests <- c(enc$named_parameters(), enc$named_buffers())
# The final norm never runs for intermediate hidden states; every
# checkpoint key must still land
expect_true(all(names(sd) %in% names(dests)))
torch::with_no_grad({
  for (name in names(sd)) dests[[name]]$copy_(sd[[name]])
})
enc$eval()

ids <- torch::torch_tensor(
  matrix(as.integer(as.array(fx$input_ids)) + 1L, nrow = 1),
  dtype = torch::torch_long()
)
mask <- fx$attention_mask$to(dtype = torch::torch_long())

# hidden_states[-2] of a 4-layer model = state after layer 3
states <- torch::with_no_grad(enc(ids, attention_mask = mask, out_layers = 3L))
expect_true(max_abs_diff(states[[1]], fx$penult) < 1e-5)

# Mask slice to the variable-length caption (right padding -> first n)
n_real <- as.integer(sum(as.array(fx$attention_mask)))
expect_equal(n_real, 9L)
sliced <- states[[1]][1, 1:n_real, ]
expect_true(max_abs_diff(sliced, fx$penult_sliced) < 1e-5)

# --- chat template with enable_thinking = TRUE -------------------------------------

find_qwen_tokenizer <- function() {
  p <- Sys.getenv("DIFFUSER_QWEN_TOKENIZER", "")
  if (nzchar(p) && file.exists(p)) {
    return(p)
  }
  if (requireNamespace("hfhub", quietly = TRUE)) {
    for (repo in c(
      "Tongyi-MAI/Z-Image-Turbo",
      "black-forest-labs/FLUX.2-klein-4B"
    )) {
      p <- tryCatch(
        suppressMessages(hfhub::hub_download(
          repo, "tokenizer/tokenizer.json", local_files_only = TRUE
        )),
        error = function(e) ""
      )
      if (nzchar(p) && file.exists(p)) {
        return(p)
      }
    }
  }
  p <- "../../tools/cache/tokenizer_qwen.json"
  if (file.exists(p)) {
    return(p)
  }
  ""
}

tok_path <- find_qwen_tokenizer()
if (!nzchar(tok_path)) exit_file("no Qwen tokenizer.json available")

cases_path <- system.file("tinytest", "fixtures", "zimage_template_cases.json",
  package = "diffuseR")
if (cases_path == "") cases_path <- "fixtures/zimage_template_cases.json"
if (!file.exists(cases_path)) exit_file("zimage template cases missing")

cases <- jsonlite::fromJSON(cases_path, simplifyVector = FALSE)

expect_equal(cases$meta$padding_side, "right")
expect_equal(cases$meta$pad_token_id, 151643L)

tok <- qwen_bpe_tokenizer(tok_path)

for (case in cases$templated) {
  got <- encode_qwen(tok, case$text, max_length = case$max_length,
    chat_template = TRUE, enable_thinking = TRUE)
  expect_equal(got$input_ids[1, ], as.integer(unlist(case$ids)),
    info = sprintf("ids for: %s", substr(case$text, 1, 40)))
  expect_equal(got$attention_mask[1, ], as.integer(unlist(case$mask)),
    info = sprintf("mask for: %s", substr(case$text, 1, 40)))
}
