% tinyrox says don't edit this manually, but it can't stop you!
\name{anvl_t5}
\alias{anvl_t5}
\title{T5-XXL encoder forward (anvl port of \code{\link{t5_encoder}})}
\description{
anvl re-implementation of diffuseR's T5-v1.1 encoder stack, FLUX.1's
second text encoder. Mirrors the torch reference module tree 1:1 but
as pure functions over an AnvlArray weight pytree. T5 distinctives,
each carried over exactly:
\itemize{
\item \strong{RMS norm} (\code{\link[yunque]{yq_rms_norm}}: scale
only, no bias, no mean subtraction, no +1 on the weight).
\item \strong{No attention scaling} - scores are the raw
\eqn{q k^\top}, never divided by \eqn{\sqrt{d_{kv}}} (folded into
the weights), so \code{\link[yunque]{yq_sdpa}} (which scales) is
NOT used; scores are formed manually.
\item \strong{Learned relative-position bias} added to the unscaled
scores pre-softmax; computed once (host-side,
\code{\link{yq_t5_rel_pos_bias}}) from block 1's bucketed embedding
and shared by every layer.
\item \strong{Gated-GELU (GEGLU) FFN}: \code{wo(gelu(wi_0(x)) *
wi_1(x))} with the tanh-approximation GELU (T5 v1.1 = "gelu_new").
}
The encoder is bidirectional (padding mask only); FLUX passes no mask,
so none is implemented here.
}
