# Qwen3-4B text-encoder parity fixture (torch reference).
# Instantiates a 27-layer diffuseR::qwen3_encoder, loads the matching
# bf16->f32 weights from the sharded FLUX.2 text_encoder checkpoint,
# runs on fixed random token ids with a padding mask, and saves ids +
# mask + concatenated mid-stack output. Weights are reloaded by the test
# via yunque::yq_qwen3_load_weights.
#
# Usage: r tools/gen_fixture_qwen3.R

suppressMessages(library(torch))

te <- file.path(
  Sys.getenv("HOME"),
  ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B",
  "snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294", "text_encoder"
)
stopifnot(dir.exists(te))
fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/yunque/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "qwen3_encoder.safetensors")

N_LAYERS <- 27L
OUT_LAYERS <- c(9L, 18L, 27L)

# ---- sharded reader (bf16 -> f32) ----
idx <- jsonlite::fromJSON(file.path(te, "model.safetensors.index.json"))
wm <- idx$weight_map
shards <- unique(unlist(wm))
cons <- lapply(shards, function(s) {
  con <- file(file.path(te, s), "rb")
  hl <- readBin(con, "integer", n = 1, size = 8, endian = "little")
  h <- jsonlite::fromJSON(readChar(con, hl, useBytes = TRUE))
  list(con = con, header = h, start = 8 + hl)
})
names(cons) <- shards
read_one <- function(key) {
  sc <- cons[[wm[[key]]]]
  meta <- sc$header[[key]]
  n <- prod(meta$shape)
  seek(sc$con, sc$start + meta$data_offsets[1])
  if (meta$dtype == "F32") {
    vals <- readBin(sc$con, "numeric", n = n, size = 4L, endian = "little")
  } else {
    b <- readBin(sc$con, "raw", n = n * 2L)
    o <- raw(n * 4L); ii <- seq_len(n)
    o[4L * ii - 1L] <- b[2L * ii - 1L]; o[4L * ii] <- b[2L * ii]
    vals <- readBin(o, "numeric", n = n, size = 4L, endian = "little")
  }
  torch_tensor(vals, dtype = torch_float32())$reshape(meta$shape)
}

m <- diffuseR::qwen3_encoder(num_hidden_layers = N_LAYERS)
need <- names(m$state_dict())
cat("loading", length(need), "tensors into", N_LAYERS, "-layer ref...\n")
sd <- setNames(lapply(need, read_one), need)
m$load_state_dict(sd)
m$eval()
for (sc in cons) close(sc$con)
rm(sd); gc()

# ---- inputs: S=32, some padding ----
set.seed(7)
S <- 32L; vocab <- 151936L
ids0 <- sample.int(vocab, S, replace = TRUE) - 1L   # 0-based
attn <- rep(1L, S); attn[27:S] <- 0L                # last 6 padded
ids_t <- torch_tensor(matrix(ids0 + 1L, 1L), dtype = torch_long())  # 1-based
mask_t <- torch_tensor(matrix(attn, 1L), dtype = torch_long())

states <- with_no_grad(m(ids_t, attention_mask = mask_t,
                         out_layers = OUT_LAYERS))
out <- torch_cat(states, dim = -1L)                 # [1, S, 3*hidden]

safetensors::safe_save_file(list(
  ids0 = torch_tensor(matrix(ids0, 1L), dtype = torch_float32()),
  attn = torch_tensor(matrix(attn, 1L), dtype = torch_float32()),
  out = out
), fixture)
cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("out shape %s  sd %.4f\n",
            paste(dim(out), collapse = "x"), out$std()$item()))
