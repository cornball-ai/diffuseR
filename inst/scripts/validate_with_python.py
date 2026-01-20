#!/usr/bin/env python3
"""
Validate LTX-2 pipeline with Python diffusers.
Saves intermediate tensors as safetensors for comparison with R.
"""

import torch
from safetensors.torch import save_file
from pathlib import Path
import os

# Output directory
output_dir = Path.home() / "cornball_media" / "python_validation"
output_dir.mkdir(parents=True, exist_ok=True)

print("=== LTX-2 Python Validation ===\n")

# Same parameters as R test
prompt = "A Bigfoot dj transforming into a humanoid robot with lifelike realistic human facial features and hair with rubbery synthetic skin, uncanny valley style exaggerated features"
width = 512
height = 320
num_frames = 17
num_steps = 25
guidance_scale = 4.0
seed = 42

print(f"Prompt: {prompt}")
print(f"Resolution: {width}x{height}, {num_frames} frames, {num_steps} steps")
print(f"Seed: {seed}\n")

# Set seed for reproducibility
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print("Loading pipeline...")
from diffusers import LTXPipeline

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=torch.float16
)
pipe.to("cuda")

print("Generating video...")
output = pipe(
    prompt=prompt,
    negative_prompt="",
    width=width,
    height=height,
    num_frames=num_frames,
    num_inference_steps=num_steps,
    guidance_scale=guidance_scale,
    generator=torch.Generator("cuda").manual_seed(seed),
    output_type="pt",  # Return tensor
)

video_tensor = output.frames  # Shape: [B, F, C, H, W] or similar
print(f"Output shape: {video_tensor.shape}")
print(f"Output dtype: {video_tensor.dtype}")
print(f"Output range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")

# Save video tensor
save_file({"video": video_tensor.cpu().float()}, output_dir / "video_output.safetensors")
print(f"\nSaved video tensor to: {output_dir / 'video_output.safetensors'}")

# Export as video file
from diffusers.utils import export_to_video
video_path = output_dir / "bigfoot_dj_python.mp4"
export_to_video(video_tensor[0], str(video_path), fps=24)
print(f"Saved video to: {video_path}")

print("\n=== Done ===")
