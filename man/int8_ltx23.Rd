% tinyrox says don't edit this manually, but it can't stop you!
\name{int8_ltx23}
\alias{int8_ltx23}
\title{INT8 Weight Storage for the LTX-2.3 Transformer}
\description{
Streamed 8-bit weights: the cast-set linears (the official LTX fp8
cast policy) are stored as int8 with per-output-channel float32
scales, kept CPU-resident (optionally pinned), and shipped to the
compute device inside each forward - 1 byte/param over PCIe, upcast
and rescale on device. Per-channel scales carry more precision than
fp8's per-tensor scale, and int8 is a standard safetensors dtype, so
the artifact reads on stock CRAN safetensors (fp8 needs the fork).
}
