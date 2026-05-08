# Converted from PyTorch by pyrotechnics
# Review: indexing (0->1 based), integer literals (add L),
# and block structure (braces may need adjustment)

# import torch
# from einops import rearrange

# from .pixel_shuffle import PixelShuffleND
# from .res_block import ResBlock
# from .spatial_rational_resampler import SpatialRationalResampler
# from ..video_vae import VideoEncoder


latent_upsampler <- nn_module(
  "LatentUpsampler",

  forward = function(latent) {
    b, _, f, _, _ <- latent.shape
    if (self$dims == 2) {
        x <- rearrange(latent, "b c f h w -> (b f) c h w")
        x <- self$initial_conv(x)
        x <- self$initial_norm(x)
        x <- self$initial_activation(x)
        for (block in self$res_blocks) {
            x <- block(x)
        x <- self$upsampler(x)
        for (block in self$post_upsample_res_blocks) {
            x <- block(x)
        x <- self$final_conv(x)
        x <- rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
    } else {
        x <- self$initial_conv(latent)
        x <- self$initial_norm(x)
        x <- self$initial_activation(x)
        for (block in self$res_blocks) {
            x <- block(x)
        if (self$temporal_upsample) {
            x <- self$upsampler(x)
            # remove the first frame after upsampling.
            # This is done because the first frame encodes one pixel frame.
            x <- x[:, :, 1:, :, :]
        } else if (inherits(self$upsampler, "SpatialRationalResampler")) {
            x <- self$upsampler(x)
        } else {
            x <- rearrange(x, "b c f h w -> (b f) c h w")
            x <- self$upsampler(x)
            x <- rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        for (block in self$post_upsample_res_blocks) {
            x <- block(x)
        x <- self$final_conv(x)
    return(x)
  }

)
def upsample_video(latent: torch.Tensor, video_encoder: VideoEncoder, upsampler: "LatentUpsampler") -> torch.Tensor:
    """
    Apply upsampling to the latent representation using the provided upsampler,
    with normalization && un-normalization based on the video encoder's per-channel statistics.
    Args:
        latent: Input latent tensor of shape [B, C, F, H, W].
        video_encoder: VideoEncoder with per_channel_statistics for normalization.
        upsampler: LatentUpsampler module to perform upsampling.
    Returns:
        torch.Tensor: Upsampled && re-normalized latent tensor.
    """
    latent <- video_encoder.per_channel_statistics.un_normalize(latent)
    latent <- upsampler(latent)
    latent <- video_encoder.per_channel_statistics.normalize(latent)
    return(latent)

