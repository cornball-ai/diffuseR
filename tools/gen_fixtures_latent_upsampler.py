# Generate LTX latent upsampler parity fixtures for the R port.
#
# Uses diffusers' LTX2LatentUpsamplerModel (Apache-2.0) with tiny random
# configs in the two layouts the LTX-2.5 release ships: the spatial x2
# upscaler (per-frame Conv2d + 2D pixel shuffle, no rational resampler)
# and the temporal x2 upscaler (Conv3d + 1D pixel shuffle, first frame
# dropped). Run via tools/gen_fixtures.sh.

import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "upstream", "diffusers", "src"))

from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "inst", "tinytest", "fixtures",
                   "ltx_latent_upsampler.safetensors")

torch.manual_seed(31)

tensors = {}
for tag, kwargs in [
    ("spatial", dict(spatial_upsample=True, temporal_upsample=False, use_rational_resampler=False)),
    ("temporal", dict(spatial_upsample=False, temporal_upsample=True)),
]:
    model = LTX2LatentUpsamplerModel(in_channels=8, mid_channels=32, num_blocks_per_stage=1, dims=3, **kwargs).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(0.2 * torch.randn_like(p))
    x = torch.randn(1, 8, 3, 4, 5)
    with torch.no_grad():
        y = model(x)
    print(tag, tuple(x.shape), "->", tuple(y.shape))
    for k, v in model.state_dict().items():
        tensors[f"{tag}.{k}"] = v.contiguous()
    tensors[f"{tag}_x"] = x
    tensors[f"{tag}_out"] = y

save_file(tensors, OUT)
print("wrote", OUT, os.path.getsize(OUT), "bytes")
