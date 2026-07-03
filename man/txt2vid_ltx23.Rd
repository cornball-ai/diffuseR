% tinyrox says don't edit this manually, but it can't stop you!
\name{txt2vid_ltx23}
\alias{txt2vid_ltx23}
\title{LTX-2.3 Text-to-Video Pipeline}
\description{
Fresh R port of the LTX-2 text-to-video flow from the diffusers
reference (Apache-2.0, pipelines/ltx2/pipeline_ltx2.py), specialized
for the distilled LTX 2.3 checkpoints: 8-step official sigma schedule,
no classifier-free guidance, joint audio-video denoising with an Euler
velocity step, and audio decoding through the audio VAE and BWE
vocoder to 48 kHz stereo.
}
