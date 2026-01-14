# Tests for FlowMatch Euler Discrete Scheduler

# Test scheduler creation
expect_silent({
  schedule <- flowmatch_scheduler_create()
})

expect_equal(schedule$config$num_train_timesteps, 1000L)
expect_equal(schedule$config$shift, 1.0)
expect_false(schedule$config$use_dynamic_shifting)
expect_equal(length(schedule$sigmas), 1000)

# Test with custom parameters
schedule2 <- flowmatch_scheduler_create(
  num_train_timesteps = 500,
  shift = 2.0,
  use_dynamic_shifting = FALSE
)
expect_equal(schedule2$config$num_train_timesteps, 500)
expect_equal(schedule2$config$shift, 2.0)

# Test calculate_shift
mu <- flowmatch_calculate_shift(
  seq_len = 1024,
  base_seq_len = 256,
  max_seq_len = 4096,
  base_shift = 0.5,
  max_shift = 1.15
)
expect_true(is.numeric(mu))
expect_true(mu > 0.5 && mu < 1.15)

# Test set_timesteps
schedule <- flowmatch_scheduler_create()
schedule <- flowmatch_set_timesteps(schedule, num_inference_steps = 8)

expect_equal(schedule$num_inference_steps, 8)
expect_true(inherits(schedule$sigmas, "torch_tensor"))
expect_true(inherits(schedule$timesteps, "torch_tensor"))
# Should have num_inference_steps + 1 sigmas (with terminal)
expect_equal(as.numeric(schedule$sigmas$shape[1]), 9)

# Test scheduler step
if (torch::cuda_is_available()) {
  device <- "cuda"
} else {
  device <- "cpu"
}

schedule <- flowmatch_scheduler_create()
schedule <- flowmatch_set_timesteps(schedule, num_inference_steps = 8, device = device)

# Create dummy sample and model output
sample <- torch::torch_randn(c(1, 4, 8, 16, 16), device = device)
model_output <- torch::torch_randn(c(1, 4, 8, 16, 16), device = device)
timestep <- schedule$timesteps[1]$item()

# Perform step
result <- flowmatch_scheduler_step(
  model_output = model_output,
  timestep = timestep,
  sample = sample,
  schedule = schedule
)
prev_sample <- result$prev_sample
schedule <- result$schedule

expect_true(inherits(prev_sample, "torch_tensor"))
expect_equal(as.numeric(prev_sample$shape), c(1, 4, 8, 16, 16))

# Test that step_index advances
expect_equal(schedule$step_index, 2L)

# Test scale_noise (forward process)
clean_sample <- torch::torch_randn(c(1, 4, 8, 16, 16), device = device)
noise <- torch::torch_randn(c(1, 4, 8, 16, 16), device = device)
timestep_tensor <- torch::torch_tensor(schedule$timesteps[1]$item(), device = device)

noisy_sample <- flowmatch_scale_noise(
  sample = clean_sample,
  timestep = timestep_tensor,
  noise = noise,
  schedule = schedule
)

expect_true(inherits(noisy_sample, "torch_tensor"))
expect_equal(as.numeric(noisy_sample$shape), c(1, 4, 8, 16, 16))

# Test that shifted schedules differ from unshifted
schedule_shifted <- flowmatch_scheduler_create(shift = 3.0)
schedule_unshifted <- flowmatch_scheduler_create(shift = 1.0)

expect_false(all(schedule_shifted$sigmas == schedule_unshifted$sigmas))

cat("All FlowMatch scheduler tests passed\n")
