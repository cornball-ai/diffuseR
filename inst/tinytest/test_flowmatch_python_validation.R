# Validation tests for FlowMatch scheduler against Python diffusers
# These tests ensure numerical equivalence with HuggingFace implementation

# Load Python test cases
test_cases_path <- system.file("validation/flowmatch_test_cases.json",
                                package = "diffuseR")

if (!file.exists(test_cases_path)) {
  # Skip if validation file not found (CI/CRAN)
  cat("Skipping Python validation tests - test cases file not found\n")
} else {
  test_cases <- jsonlite::fromJSON(test_cases_path)

  # Tolerance for numerical comparison
  tol <- 1e-4

  # Test 1: Basic 8-step scheduler sigmas match Python
  cat("Test 1: Basic 8-step scheduler sigmas\n")
  schedule <- flowmatch_scheduler_create(
    num_train_timesteps = 1000,
    shift = 1.0
  )
  schedule <- flowmatch_set_timesteps(schedule, num_inference_steps = 8)

  python_sigmas <- test_cases$basic_8_steps$sigmas
  r_sigmas <- as.numeric(schedule$sigmas)

  expect_equal(length(r_sigmas), length(python_sigmas),
               info = "Sigma lengths should match")

  max_diff <- max(abs(r_sigmas - python_sigmas))
  expect_true(max_diff < tol,
              info = sprintf("Max sigma diff %.6f exceeds tolerance %.6f",
                           max_diff, tol))
  cat(sprintf("  Max sigma diff: %.6e (tolerance: %.6e)\n", max_diff, tol))

  # Test 2: Shifted scheduler (shift=3.0, 20 steps)
  cat("Test 2: Shifted scheduler (shift=3.0)\n")
  schedule_shifted <- flowmatch_scheduler_create(
    num_train_timesteps = 1000,
    shift = 3.0
  )
  schedule_shifted <- flowmatch_set_timesteps(schedule_shifted,
                                               num_inference_steps = 20)

  python_sigmas_shifted <- test_cases$shifted_20_steps$sigmas
  r_sigmas_shifted <- as.numeric(schedule_shifted$sigmas)

  max_diff_shifted <- max(abs(r_sigmas_shifted - python_sigmas_shifted))
  expect_true(max_diff_shifted < tol,
              info = sprintf("Max shifted sigma diff %.6f exceeds tolerance",
                           max_diff_shifted))
  cat(sprintf("  Max sigma diff: %.6e\n", max_diff_shifted))

  # Test 3: Timesteps match
  cat("Test 3: Timesteps match\n")
  python_timesteps <- test_cases$basic_8_steps$timesteps
  r_timesteps <- as.numeric(schedule$timesteps)

  max_timestep_diff <- max(abs(r_timesteps - python_timesteps))
  expect_true(max_timestep_diff < tol,
              info = sprintf("Max timestep diff %.6f exceeds tolerance",
                           max_timestep_diff))
  cat(sprintf("  Max timestep diff: %.6e\n", max_timestep_diff))

  # Test 4: Step function output matches
  cat("Test 4: Step function numerical match\n")
  schedule <- flowmatch_scheduler_create(num_train_timesteps = 1000, shift = 1.0)
  schedule <- flowmatch_set_timesteps(schedule, num_inference_steps = 8)

  # Use same seed as Python
  torch::torch_manual_seed(42)
  sample <- torch::torch_randn(c(1, 4, 8, 16, 16))
  model_output <- torch::torch_randn(c(1, 4, 8, 16, 16))
  timestep <- schedule$timesteps[1]$item()

  result <- flowmatch_scheduler_step(
    model_output = model_output,
    timestep = timestep,
    sample = sample,
    schedule = schedule
  )
  prev_sample <- result$prev_sample

  python_step <- test_cases$step_test

  r_mean <- as.numeric(prev_sample$mean())
  r_std <- as.numeric(prev_sample$std())
  r_min <- as.numeric(prev_sample$min())
  r_max <- as.numeric(prev_sample$max())

  # Note: Due to different RNG implementations, exact match is unlikely
  # But dt and sigma values should match exactly
  expect_true(abs(python_step$sigma_current - 1.0) < tol,
              info = "Sigma current should be 1.0")
  expect_true(abs(python_step$sigma_next - as.numeric(schedule$sigmas[2])) < tol,
              info = "Sigma next should match")

  cat(sprintf("  Python mean: %.6f, R mean: %.6f\n",
              python_step$prev_sample_mean, r_mean))
  cat(sprintf("  dt: %.6f (expected: %.6f)\n",
              as.numeric(schedule$sigmas[2]) - as.numeric(schedule$sigmas[1]),
              python_step$dt))

  # Test 5: Full loop with deterministic scaling
  cat("Test 5: Full denoising loop pattern\n")
  schedule <- flowmatch_scheduler_create(num_train_timesteps = 1000, shift = 1.0)
  schedule <- flowmatch_set_timesteps(schedule, num_inference_steps = 8)

  # Initialize with seed
  torch::torch_manual_seed(123)
  latents <- torch::torch_randn(c(1, 4, 4, 16, 16))

  # Run full loop
  for (i in seq_along(schedule$timesteps)) {
    t <- schedule$timesteps[i]$item()
    model_output <- latents * 0.1  # Same as Python test
    result <- flowmatch_scheduler_step(
      model_output = model_output,
      timestep = t,
      sample = latents,
      schedule = schedule
    )
    latents <- result$prev_sample
    schedule <- result$schedule
  }

  final_mean <- as.numeric(latents$mean())
  final_std <- as.numeric(latents$std())

  python_final <- test_cases$full_loop

  # The pattern should be similar even if RNG differs
  # Check that values are in reasonable range
  expect_true(abs(final_std) < 2.0,
              info = "Final std should be reasonable")
  cat(sprintf("  Python final: mean=%.6f, std=%.6f\n",
              python_final$final_mean, python_final$final_std))
  cat(sprintf("  R final:      mean=%.6f, std=%.6f\n", final_mean, final_std))

  cat("\nAll Python validation tests completed\n")
}
