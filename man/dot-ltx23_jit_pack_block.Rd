% tinyrox says don't edit this manually, but it can't stop you!
\name{.ltx23_jit_pack_block}
\alias{.ltx23_jit_pack_block}
\title{Pack a transformer block's weights for the JIT stack}
\usage{
.ltx23_jit_pack_block(block)
}
\arguments{
\item{block}{An NF4-quantized \code{ltx23_transformer_block}.}
}
\value{
List of 114 tensors.
}
\description{
Returns the block's tensors in the fixed 114-slot layout consumed by
the compiled \code{stack_nf4}/\code{block_nf4} TorchScript functions.
Tensor handles are borrowed, not copied.
}
\keyword{internal}
