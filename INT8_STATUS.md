# INT8 streamed transformer: measured status (2026-07-31)

Real-model benchmark run on the branch. **Not shippable on a 16 GB card
as implemented.** The quantizer is good; the execution model is what
costs. Read this before picking the work back up.

Hardware: RTX 5060 Ti 16 GB (16.31 GB VRAM), 125 GB host RAM, NVMe.

## Render benchmark (768x512x49, seed 42, identical cached Gemma3 embeds)

| format | pipeline load | cold render | warm render |
|---|---|---|---|
| nf4 | 56.5 s | 38.6 s | **32.3 s** |
| int8 | 61.3 s | 111.9 s | **105.6 s** |

int8 is **3.27x slower end to end**, despite winning every isolated
measurement below.

## Component measurements (transformer_blocks.0.ff.net.0.proj.weight,
16384x4096, dequant cuda-fenced, 30 reps)

| | bits/w | rel. err | dequant |
|---|---|---|---|
| NF4 byte-LUT gather | 4.50 | 0.10999 | 7.77 ms |
| int8 per-channel | 8.01 | **0.02584** | **1.50 ms** |
| write-only floor | - | - | 0.70 ms |

int8 decodes **5.2x faster** with **4.3x lower error**. It is a better
quantizer on both axes. H2D of the same tensor pinned: int8 2.50 ms
(67 MB), nf4 1.30 ms (34 MB), measured bus rate ~27 GB/s.

## Why the render is slow anyway

`int8_ltx23.R` contains **zero references to the jit path**. Only
`dit_ltx23.R`, `jit_ltx23.R` and `nf4_ltx23.R` use it. NF4 executes the
48-block stack as one TorchScript graph call per step; int8 runs
`ltx23_int8_linear$forward` eagerly from R for every cast-set weight.

The artifact holds **3,699 int8 tensors** (confirmed by the pin_memory
call count at load). That is ~3,700 R-level forwards plus 3,700 separate
H2D transfers per step, ~29,600 per 8-step render. The 73.3 s gap over
29,592 layer-steps is ~2.5 ms each: eager dispatch and per-transfer
latency, not bandwidth. Raw transfer is only ~5.5 s of the render
(18.52 GB/step at 27 GB/s).

This is the "eager-path overhead" risk flagged when the branch was
opened. It is real and it dominates.

## Blocker 1: int8 cannot be GPU-resident on 16 GB

Artifact footprint by dtype (both artifacts, 7291 tensors each):

| | NF4 | int8 |
|---|---|---|
| F32 (scales) | 1.18 GB | 0.04 GB |
| BF16 (non-cast) | 9.09 GB | 9.09 GB |
| 4-bit U8 / I8 | 9.26 GB | **18.52 GB** |
| total | 19.52 GB | 27.65 GB |

The int8 cast set alone (18.52 GB) exceeds the whole card (16.31 GB),
before activations and before the 9.09 GB of BF16 non-cast tensors.
There is no configuration where it stays resident on 16 GB.

**On a 24 GB card it fits** (~18.6 GB weights + activations ~= 22 GB) and
could use the jit stack exactly like NF4. That is the natural home for
this work.

## Blocker 2 (LANDMINE): jit_trace bakes closure-captured CPU tensors

Tracing a forward that reads a host-resident weight from the enclosing
scope **constant-folds it into a device constant**. Streaming silently
stops. Verified by mutating the host tensor after tracing:

```r
w_cpu <- torch_ones(c(256,256))$to(dtype=torch_int8())$pin_memory(device=D)
f  <- function(x) nnf_linear(x, w_cpu$to(device=D)$to(dtype=torch_float32()))
tr <- jit_trace(f, x)
tr(x)                      # 256
with_no_grad(w_cpu$mul_(2L))
tr(x)                      # 256  <- host mutation ignored: BAKED IN
```

This does not error. It looks like a ~26% speedup while quietly moving
every weight onto the device, which on a 16 GB card means OOM rather
than a win.

**The fix is to pass weights as graph inputs**, which does stream
correctly:

```r
g  <- function(x, w) nnf_linear(x, w$to(device=D, non_blocking=TRUE)$to(dtype=torch_float32()))
tr <- jit_trace(g, x, w_cpu)
tr(x, w_cpu)               # 256
with_no_grad(w_cpu$mul_(2L))
tr(x, w_cpu)               # 512  <- re-reads host memory: STREAMING
```

## The path forward, if resumed

1. Restructure `ltx23_int8_linear` so int8 weights flow as **graph
   inputs**, not module fields.
2. Extend `jit_ltx23.R` to accept them, probably as `List[Tensor]` given
   there are ~3,700. **Whether R torch's `jit_trace` handles a list of
   tensors as an input is the cheap gate to test first**; if it does not,
   the design is not expressible and int8 is a >16 GB-card feature only.
3. Expected payoff (estimate, not measured): removing most of the ~73 s
   of dispatch lands int8 near 32-35 s, roughly parity with NF4, but with
   4.3x lower error and ~9-10 GB of VRAM freed. That headroom is what
   currently forces tiled decode and caps resolution.

Parity on speed is not exciting on its own. The accuracy and the freed
VRAM are the reasons to do it.

## Other notes

- int8 is a standard safetensors dtype, so the artifact reads on stock
  CRAN safetensors. As of 2026-07-31 fp8 does too (mlverse/safetensors
  #13 merged), so this is no longer a differentiator.
- Both QC renders were visually valid; int8 is functionally correct.
- R torch gotchas hit while measuring: `pin_memory()` requires the
  `device` argument that libtorch deprecates; `cuda_memory_allocated()`
  does not exist, use `cuda_memory_stats()$allocated_bytes$all$current`;
  bf16 tensors need a float32 cast before `as.numeric()`.
