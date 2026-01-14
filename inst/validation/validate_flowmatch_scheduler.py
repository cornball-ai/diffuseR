#!/usr/bin/env python3
"""
Validation script for FlowMatch scheduler.
Generates test cases with known inputs/outputs for comparison with R implementation.
"""

import json
import torch
import numpy as np
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

def generate_test_cases():
    """Generate test cases for R validation."""

    test_cases = {}

    # Test 1: Basic scheduler creation and set_timesteps
    print("Test 1: Basic scheduler with shift=1.0")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_inference_steps=8)

    test_cases["basic_8_steps"] = {
        "config": {
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "num_inference_steps": 8,
        },
        "sigmas": scheduler.sigmas.numpy().tolist(),
        "timesteps": scheduler.timesteps.numpy().tolist(),
    }

    # Test 2: Shifted scheduler
    print("Test 2: Scheduler with shift=3.0")
    scheduler_shifted = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    scheduler_shifted.set_timesteps(num_inference_steps=20)

    test_cases["shifted_20_steps"] = {
        "config": {
            "num_train_timesteps": 1000,
            "shift": 3.0,
            "num_inference_steps": 20,
        },
        "sigmas": scheduler_shifted.sigmas.numpy().tolist(),
        "timesteps": scheduler_shifted.timesteps.numpy().tolist(),
    }

    # Test 3: Step function with deterministic input
    print("Test 3: Scheduler step with deterministic input")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )
    scheduler.set_timesteps(num_inference_steps=8)

    # Use deterministic input
    torch.manual_seed(42)
    sample = torch.randn(1, 4, 8, 16, 16)  # video-like: batch, channels, frames, h, w
    model_output = torch.randn(1, 4, 8, 16, 16)
    timestep = scheduler.timesteps[0]

    result = scheduler.step(model_output, timestep, sample)
    prev_sample = result.prev_sample

    test_cases["step_test"] = {
        "sample_shape": list(sample.shape),
        "sample_seed": 42,
        "timestep": float(timestep),
        "sigma_current": float(scheduler.sigmas[0]),
        "sigma_next": float(scheduler.sigmas[1]),
        "dt": float(scheduler.sigmas[1] - scheduler.sigmas[0]),
        "prev_sample_mean": float(prev_sample.mean()),
        "prev_sample_std": float(prev_sample.std()),
        "prev_sample_min": float(prev_sample.min()),
        "prev_sample_max": float(prev_sample.max()),
    }

    # Test 4: Full denoising loop
    print("Test 4: Full 8-step denoising loop")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )
    scheduler.set_timesteps(num_inference_steps=8)

    torch.manual_seed(123)
    latents = torch.randn(1, 4, 4, 16, 16)  # smaller for faster test

    intermediate_stats = []
    for i, t in enumerate(scheduler.timesteps):
        # Simulate model output (just use latents scaled for testing)
        model_output = latents * 0.1
        result = scheduler.step(model_output, t, latents)
        latents = result.prev_sample

        intermediate_stats.append({
            "step": i,
            "timestep": float(t),
            "latents_mean": float(latents.mean()),
            "latents_std": float(latents.std()),
        })

    test_cases["full_loop"] = {
        "seed": 123,
        "shape": [1, 4, 4, 16, 16],
        "num_steps": 8,
        "intermediate_stats": intermediate_stats,
        "final_mean": float(latents.mean()),
        "final_std": float(latents.std()),
    }

    # Test 5: scale_noise (forward process)
    print("Test 5: scale_noise function")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
    )
    scheduler.set_timesteps(num_inference_steps=8)

    torch.manual_seed(99)
    clean_sample = torch.randn(1, 4, 4, 8, 8)
    noise = torch.randn(1, 4, 4, 8, 8)
    # scale_noise expects a batch of timesteps matching batch_size
    timestep = scheduler.timesteps[2:3].expand(1)  # shape [1]

    noisy = scheduler.scale_noise(clean_sample, timestep, noise)

    test_cases["scale_noise"] = {
        "seed": 99,
        "shape": [1, 4, 4, 8, 8],
        "timestep": float(timestep[0]),
        "noisy_mean": float(noisy.mean()),
        "noisy_std": float(noisy.std()),
    }

    return test_cases


def main():
    print("Generating FlowMatch scheduler test cases...")
    test_cases = generate_test_cases()

    output_path = "flowmatch_test_cases.json"
    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"\nTest cases saved to {output_path}")
    print("\nSummary:")
    for name, data in test_cases.items():
        print(f"  - {name}")

    return test_cases


if __name__ == "__main__":
    main()
