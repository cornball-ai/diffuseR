% tinyrox says don't edit this manually, but it can't stop you!
\name{fp8_ltx23}
\alias{fp8_ltx23}
\title{FP8 Weight Storage for the LTX-2.3 Transformer}
\description{
GPU-poor weight handling: the large attention/FFN linears of the DiT
are stored as float8_e4m3fn with per-tensor scales (the official LTX
quantization policy), kept CPU-resident (optionally pinned), and
dequantized on the compute device inside each forward. Everything
else (norms, embeddings, modulation tables, biases) stays bfloat16.
Requires a safetensors build with F8 dtype support.
}
