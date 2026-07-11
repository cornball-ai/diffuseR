# Benchmark the anvl FLUX.2 single block (yunque port).
# Usage: r tools/bench_flux2_block_anvl.R [cpu|cuda] [S] [highest|default]
#
# Random weights at Klein-4B dims (timing only; parity is the tinytest).
# First call compiles; it is await()ed before the timed loop starts.
# The steady-state loop queues all calls and awaits the last, mirroring
# the torch script's synchronize-at-end pattern. precision "highest"
# forbids TF32 (honest f32); "default" allows it. On anvl 0.3.0 the
# precision argument is ignored and CUDA dots run TF32 regardless.

suppressMessages(library(anvl))
suppressMessages(library(yunque))

dev <- if (length(argv) >= 1) argv[1] else "cpu"
S <- if (length(argv) >= 2) as.integer(argv[2]) else 4608L
precision <- if (length(argv) >= 3) argv[3] else "highest"

dim <- 3072L
heads <- 24L
head_dim <- 128L
inner <- heads * head_dim
mlp_hidden <- as.integer(dim * 3)

set.seed(1)
nv <- function(data, ...) nv_array(data, dtype = "f32", device = dev, ...)

t0 <- Sys.time()
w_qkv_t <- nv(matrix(rnorm(dim * (3L * inner + 2L * mlp_hidden), sd = 0.02),
                     dim))
w_out_t <- nv(matrix(rnorm((inner + mlp_hidden) * dim, sd = 0.02),
                     inner + mlp_hidden))
norm_q_w <- nv(rep(1, head_dim))
norm_k_w <- nv(rep(1, head_dim))
h <- nv(array(rnorm(S * dim, sd = 0.5), c(1L, S, dim)))
shift <- nv(array(rnorm(dim, sd = 0.1), c(1L, 1L, dim)))
scale <- nv(array(rnorm(dim, sd = 0.1), c(1L, 1L, dim)))
gate <- nv(array(rnorm(dim, sd = 0.1), c(1L, 1L, dim)))
theta <- matrix(runif(S * head_dim / 2, -pi, pi), S)
cos <- nv(cos(theta)[, rep(seq_len(head_dim / 2), each = 2)])
sin <- nv(sin(theta)[, rep(seq_len(head_dim / 2), each = 2)])
t_data <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

f <- jit(yq_flux2_single_block(heads = heads, head_dim = head_dim,
                               precision = precision))

t0 <- Sys.time()
out <- f(h, shift, scale, gate, cos, sin,
         w_qkv_t, w_out_t, norm_q_w, norm_k_w)
await(out)
t_compile <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

n <- 20L
t0 <- Sys.time()
for (i in seq_len(n)) {
  out <- f(h, shift, scale, gate, cos, sin,
           w_qkv_t, w_out_t, norm_q_w, norm_k_w)
}
await(out)
el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

cat(sprintf(
    "RESULT anvl %s %s S=%d: %.1f ms/iter (compile+first %.1fs, data %.1fs)\n",
    dev, precision, S, 1000 * el / n, t_compile, t_data))
