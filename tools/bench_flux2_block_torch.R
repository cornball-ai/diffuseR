# Benchmark the torch FLUX.2 single block (diffuseR reference).
# Usage: r tools/bench_flux2_block_torch.R [float32|bfloat16] [S] [cpu|cuda]

suppressMessages(library(torch))

dtype_str <- if (length(argv) >= 1) argv[1] else "float32"
S <- if (length(argv) >= 2) as.integer(argv[2]) else 4608L
dev <- if (length(argv) >= 3) argv[3] else "cuda"
dt <- switch(dtype_str,
             float32 = torch_float32(),
             bfloat16 = torch_bfloat16(),
             stop("dtype?"))

dim <- 3072L
heads <- 24L
head_dim <- 128L

blk <- diffuseR::flux2_single_block(dim, heads, head_dim)
blk$to(device = dev, dtype = dt)
blk$eval()

torch_manual_seed(1)
h <- (torch_randn(1, S, dim) * 0.5)$to(device = dev, dtype = dt)
temb_mod <- (torch_randn(1, dim * 3L) * 0.1)$to(device = dev, dtype = dt)
theta <- (torch_rand(S, head_dim %/% 2L) - 0.5) * (2 * pi)
cos <- torch_repeat_interleave(torch_cos(theta), 2L, dim = -1L)$to(device = dev)
sin <- torch_repeat_interleave(torch_sin(theta), 2L, dim = -1L)$to(device = dev)
freqs <- list(cos, sin)

sync <- function() if (dev == "cuda") cuda_synchronize()

with_no_grad({
  for (i in 1:3) out <- blk(h, temb_mod, image_rotary_emb = freqs)
  sync()
  n <- 20L
  t0 <- Sys.time()
  for (i in seq_len(n)) out <- blk(h, temb_mod, image_rotary_emb = freqs)
  sync()
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
})

peak <- tryCatch(cuda_memory_stats()$allocated_bytes$all$peak / 1e9,
                 error = function(e) NA)
cat(sprintf("RESULT torch %s %s S=%d: %.1f ms/iter (peak %.1f GB)\n",
            dev, dtype_str, S, 1000 * el / n, peak))
