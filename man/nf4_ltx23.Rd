% tinyrox says don't edit this manually, but it can't stop you!
\name{nf4_ltx23}
\alias{nf4_ltx23}
\title{NF4 Weight Storage for the LTX-2.3 Transformer}
\description{
4-bit NormalFloat quantization (the QLoRA scheme: per-block absmax
normalization against a 16-level quantile code, two indices packed
per byte). At ~4.5 bits/parameter the 22B transformer fits in about
12.5 GB, small enough to stay resident on a 16 GB GPU: no per-step
PCIe weight streaming, at a small quality cost relative to fp8.
Quantization and dequantization are pure torch ops (bucketize,
index_select) - no custom kernels.
}
