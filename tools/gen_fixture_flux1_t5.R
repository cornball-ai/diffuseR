# FLUX.1 T5-XXL encoder parity fixture (torch reference), RANDOM-INIT at
# a SMALL config for architecture parity without the 4.7B checkpoint.
# Instantiates diffuseR::t5_encoder with random weights, runs it under
# with_no_grad on a fixed token-id sequence (no padding - FLUX passes no
# mask), and saves the full state_dict + ids + output to one f32
# safetensors file. The anvl test reloads the weights via
# yq_t5_load_weights and feeds the same ids.
#
# Usage: /home/troy/diffuseR-f1-lib/ranvl tools/gen_fixture_flux1_t5.R

suppressMessages(library(torch))
suppressMessages(library(diffuseR))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "flux1_t5_small.safetensors")

# Small T5-v1.1 config: architecture parity, a few hundred KB fixture.
VOCAB <- 48L; D_MODEL <- 64L; D_KV <- 16L; NUM_HEADS <- 4L; D_FF <- 128L
NUM_LAYERS <- 3L; NUM_BUCKETS <- 32L; MAX_DIST <- 128L; EPS <- 1e-6; S <- 16L

set.seed(1)
torch_manual_seed(1)

m <- t5_encoder(vocab_size = VOCAB, d_model = D_MODEL, d_kv = D_KV,
                num_heads = NUM_HEADS, d_ff = D_FF, num_layers = NUM_LAYERS,
                relative_attention_num_buckets = NUM_BUCKETS,
                relative_attention_max_distance = MAX_DIST,
                layer_norm_epsilon = EPS)
m$eval()

# Fixed inputs: S valid tokens, no padding (FLUX uses no attention mask).
ids0 <- sample.int(VOCAB, S, replace = TRUE) - 1L         # 0-based
ids_t <- torch_tensor(matrix(ids0 + 1L, 1L), dtype = torch_long())  # 1-based
out <- with_no_grad(m(ids_t))                             # [1, S, D_MODEL]

# Save the full state_dict (contiguous f32) plus ids + output.
sd <- m$state_dict()
save_list <- lapply(sd, function(t) t$to(dtype = torch_float32())$contiguous())
save_list$input_ids <- torch_tensor(matrix(ids0, 1L), dtype = torch_float32())
save_list$output <- out$contiguous()

safetensors::safe_save_file(save_list, fixture)

cat(sprintf("fixture: %s (%.2f KB)\n", fixture, file.size(fixture) / 1e3))
cat(sprintf("out shape %s  sd %.4f  range [%.3f, %.3f]\n",
            paste(dim(out), collapse = "x"), out$std()$item(),
            out$min()$item(), out$max()$item()))
