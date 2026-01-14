#!/usr/bin/env python3
"""
Validation script for RoPE (Rotary Position Embeddings).
Generates test cases with known inputs/outputs for comparison with R implementation.
"""

import json
import sys
import torch
import numpy as np

# Add diffusers to path
sys.path.insert(0, "/home/troy/diffusers-ref/src")
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AudioVideoRotaryPosEmbed,
    apply_interleaved_rotary_emb,
)


def generate_test_cases():
    """Generate test cases for R validation."""

    test_cases = {}

    # Test 1: Video coordinate preparation
    print("Test 1: Video coordinate preparation")
    rope_embed = LTX2AudioVideoRotaryPosEmbed(
        dim=2048,
        patch_size=1,
        patch_size_t=1,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        scale_factors=(8, 32, 32),
        theta=10000.0,
        causal_offset=1,
        modality="video",
        rope_type="interleaved",
    )

    coords = rope_embed.prepare_video_coords(
        batch_size=2,
        num_frames=4,
        height=16,
        width=16,
        device=torch.device("cpu"),
        fps=24.0,
    )

    test_cases["video_coords"] = {
        "config": {
            "batch_size": 2,
            "num_frames": 4,
            "height": 16,
            "width": 16,
            "fps": 24.0,
        },
        "coords_shape": list(coords.shape),
        "coords_min": float(coords.min()),
        "coords_max": float(coords.max()),
        "coords_mean": float(coords.mean()),
        # Sample values for specific verification
        "coords_first_patch": coords[0, :, 0, :].tolist(),
        "coords_last_patch": coords[0, :, -1, :].tolist(),
    }

    # Test 2: RoPE forward (frequency computation)
    print("Test 2: RoPE frequency computation")
    cos_freqs, sin_freqs = rope_embed.forward(coords)

    test_cases["rope_freqs"] = {
        "cos_shape": list(cos_freqs.shape),
        "sin_shape": list(sin_freqs.shape),
        "cos_min": float(cos_freqs.min()),
        "cos_max": float(cos_freqs.max()),
        "cos_mean": float(cos_freqs.mean()),
        "sin_min": float(sin_freqs.min()),
        "sin_max": float(sin_freqs.max()),
        "sin_mean": float(sin_freqs.mean()),
    }

    # Test 3: Apply interleaved rotary embedding
    print("Test 3: Apply interleaved rotary embedding")
    torch.manual_seed(42)
    x = torch.randn(2, 1024, 2048)  # [B, S, C]

    rotated = apply_interleaved_rotary_emb(x, (cos_freqs, sin_freqs))

    test_cases["apply_rope"] = {
        "seed": 42,
        "input_shape": list(x.shape),
        "output_shape": list(rotated.shape),
        "output_mean": float(rotated.mean()),
        "output_std": float(rotated.std()),
        "output_min": float(rotated.min()),
        "output_max": float(rotated.max()),
    }

    # Test 4: Different patch sizes
    print("Test 4: Different patch sizes")
    rope_patched = LTX2AudioVideoRotaryPosEmbed(
        dim=2048,
        patch_size=2,
        patch_size_t=2,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        scale_factors=(8, 32, 32),
        theta=10000.0,
        modality="video",
    )

    coords_patched = rope_patched.prepare_video_coords(
        batch_size=1,
        num_frames=8,
        height=32,
        width=32,
        device=torch.device("cpu"),
    )

    test_cases["patched_coords"] = {
        "config": {
            "patch_size": 2,
            "patch_size_t": 2,
            "num_frames": 8,
            "height": 32,
            "width": 32,
        },
        "coords_shape": list(coords_patched.shape),
        "expected_num_patches": (8 // 2) * (32 // 2) * (32 // 2),
    }

    return test_cases


def main():
    print("Generating RoPE test cases...")
    test_cases = generate_test_cases()

    output_path = "/home/troy/diffuseR/inst/validation/rope_test_cases.json"
    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"\nTest cases saved to {output_path}")
    print("\nSummary:")
    for name, data in test_cases.items():
        print(f"  - {name}")

    return test_cases


if __name__ == "__main__":
    main()
