# LTX-2.3 Gemma3 text-encoder parity fixture (torch reference).
#
# Instantiates diffuseR::gemma3_text_model at a SMALL random-init config
# (architecture parity, memory-light: ~1M params / ~4 MB f32 vs the real
# 12B), randomizes the RMSNorm weights (default init is zeros, which would
# make the Gemma (1+weight) trick indistinguishable from plain weight),
# runs it under with_no_grad on a fixed short token-id sequence with a
# padding mask, and saves state_dict + ids + attn + stacked hidden states
# (the [B, S, hidden, num_layers+1] tensor the LTX connectors consume).
# The anvl port reloads the weights from this same file.
#
# Usage: /home/troy/diffuseR-ltx-lib/ranvl tools/gen_fixture_ltx_gemma3.R

suppressMessages(library(torch))

fixture_dir <- file.path(Sys.getenv("HOME"), ".local/share/R/diffuseR/fixtures")
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
fixture <- file.path(fixture_dir, "ltx_gemma3.safetensors")

# ---- SMALL config (mirrors the real Gemma3-LTX2 shape choices) ----
# head_dim != num_heads*head_dim/hidden (real: 16*256=4096 != 3840);
# query_pre_attn_scalar == head_dim (real: 256 == 256);
# head_dim (48) != seq_len (24) so the sdpa scratch coincidence can't bite.
config <- list(
  vocab_size = 320L,
  hidden_size = 128L,
  intermediate_size = 256L,
  num_hidden_layers = 6L,          # layer_idx 5 (1-based 6) is global, rest sliding
  num_attention_heads = 4L,
  num_key_value_heads = 2L,        # GQA groups = 2
  head_dim = 48L,
  max_position_embeddings = 2048L,
  rms_norm_eps = 1e-6,
  rope_theta = 1000000.0,          # global theta
  rope_scaling = list(factor = 8.0, type = "linear"),  # global inv_freq /= 8
  sliding_window = 1024L,
  sliding_window_pattern = 6L,     # every 6th layer is global attention
  attn_logit_softcapping = NULL,   # not used by this Gemma3 variant
  query_pre_attn_scalar = 48L      # == head_dim -> scale 1/sqrt(head_dim)
)

set.seed(7)
torch::torch_manual_seed(7L)

model <- diffuseR::gemma3_text_model(config)

# Randomize the RMSNorm weights (default init = zeros -> (1+w)=1, which
# would not distinguish the Gemma (1+weight) form from plain weight).
torch::with_no_grad({
  sd0 <- model$state_dict()
  for (nm in names(sd0)) {
    if (grepl("norm", nm)) {
      p <- model$parameters[[nm]]
      p$copy_(torch::torch_randn(p$shape) * 0.2)
    }
  }
})
model$eval()

# ---- inputs: S = 24, last 5 tokens padded ----
S <- 24L
ids0 <- sample.int(config$vocab_size, S, replace = TRUE) - 1L   # 0-based
attn <- rep(1L, S); attn[20:S] <- 0L
ids_t <- torch_tensor(matrix(ids0, 1L), dtype = torch_long())   # 0-based; model adds 1
mask_t <- torch_tensor(matrix(attn, 1L), dtype = torch_long())

out <- torch::with_no_grad(
  model(ids_t, attention_mask = mask_t, output_hidden_states = TRUE))
# Stack ALL hidden states (embedding + every layer, num_layers+1 entries):
# [B, S, hidden, num_layers+1] -- exactly what encode_with_gemma3 returns.
stacked <- torch::torch_stack(out$hidden_states, dim = -1L)$contiguous()

# ---- save state_dict (contiguous) + inputs + output ----
sd <- model$state_dict()
save_list <- lapply(sd, function(t) t$to(dtype = torch_float32())$contiguous())
save_list$ids0 <- torch_tensor(matrix(ids0, 1L), dtype = torch_float32())
save_list$attn <- torch_tensor(matrix(attn, 1L), dtype = torch_float32())
save_list$out <- stacked

safetensors::safe_save_file(save_list, fixture)

cat("state_dict keys (", length(sd), "):\n", sep = "")
cat(paste0("  ", names(sd)), sep = "\n"); cat("\n")
cat(sprintf("fixture: %s (%.2f MB)\n", fixture, file.size(fixture) / 1e6))
cat(sprintf("out shape %s  sd %.4f  range [%.3f, %.3f]\n",
            paste(dim(stacked), collapse = "x"), stacked$std()$item(),
            stacked$min()$item(), stacked$max()$item()))
