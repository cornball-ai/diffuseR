% tinyrox says don't edit this manually, but it can't stop you!
\name{zimage_block}
\alias{zimage_block}
\title{Z-Image transformer block}
\usage{
zimage_block(dim, n_heads, norm_eps = 1e-05, modulation = TRUE)
}
\arguments{
\item{dim}{Integer. Model width.}

\item{n_heads}{Integer. Attention heads; head dim is dim / n_heads.}

\item{norm_eps}{Numeric. RMSNorm epsilon. Default 1e-5.}

\item{modulation}{Logical. Whether the block is timestep-modulated.}
}
\description{
Sandwich-norm residual block shared by the noise refiner, the context
refiner and the main trunk. With \code{modulation = TRUE} the block
carries an adaLN linear producing (scale_msa, gate_msa, scale_mlp,
gate_mlp); the context refiner uses \code{modulation = FALSE} and has
no adaLN weights at all.
}
