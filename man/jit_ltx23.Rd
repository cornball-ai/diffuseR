% tinyrox says don't edit this manually, but it can't stop you!
\name{jit_ltx23}
\alias{jit_ltx23}
\title{LTX-2.3 JIT Block Stack}
\description{
TorchScript compilation of the 48-block NF4 transformer step (cf.
the torch skill's JIT-decode pattern proven in whisper and
chatterbox). Eager execution crosses R -> lantern per op (~190 us
each) and leaves every intermediate as an R tensor handle that only
dies at gc(); at high resolution that forces a per-block gc()
costing the vast majority of step time. Compiled, the whole block
stack is one crossing: intermediates are freed eagerly by libtorch,
no R garbage accumulates, no per-block gc is needed, and attention
runs through the fused \code{scaled_dot_product_attention} kernel
instead of a materialized score matrix.
}
\details{
Weights are passed per call as a flat \code{List[Tensor]} (borrowed
by reference, no copies) with a fixed per-block layout; the packer
and the TorchScript indices must stay in lockstep (parity-tested).

}
