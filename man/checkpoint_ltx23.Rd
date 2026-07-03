% tinyrox says don't edit this manually, but it can't stop you!
\name{checkpoint_ltx23}
\alias{checkpoint_ltx23}
\title{LTX-2.3 Single-File Checkpoint Reader}
\description{
LTX 2.3 checkpoints ship as one safetensors file containing every
component (transformer, connectors, video VAE, audio VAE, vocoder),
with the model version and full component configuration embedded in
the safetensors metadata. These helpers open the file, validate the
version, split the key space by component, and stream tensors into
R torch modules one at a time so the 46 GB file is never fully
materialized in memory.
}
