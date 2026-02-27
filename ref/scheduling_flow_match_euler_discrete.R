# Converted from PyTorch by pyrotechnics
# Review: indexing (0->1 based), integer literals (add L),
# and block structure (braces may need adjustment)

# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import math
# from dataclasses import dataclass
# from typing import List, Optional, Tuple, Union

# import numpy as np
# import torch

# from ..configuration_utils import ConfigMixin, register_to_config
# from ..utils import BaseOutput, is_scipy_available, logging
# from .scheduling_utils import SchedulerMixin


if (is_scipy_available()) {
# import scipy.stats

logger <- logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] && [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading && saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        use_dynamic_shifting (`bool`, defaults to FALSE):
            Whether to apply timestep shifting on-the-fly based on the image resolution.
        base_shift (`float`, defaults to 0.5):
            Value to stabilize image generation. Increasing `base_shift` reduces variation && image is more consistent
            with desired output.
        max_shift (`float`, defaults to 1.15):
            Value change allowed to latent vectors. Increasing `max_shift` encourages more variation && image may be
            more exaggerated || stylized.
        base_image_seq_len (`int`, defaults to 256):
            The base image sequence length.
        max_image_seq_len (`int`, defaults to 4096):
            The maximum image sequence length.
        invert_sigmas (`bool`, defaults to FALSE):
            Whether to invert the sigmas.
        shift_terminal (`float`, defaults to NULL):
            The end value of the shifted timestep schedule.
        use_karras_sigmas (`bool`, defaults to FALSE):
            Whether to use Karras sigmas for step sizes in the noise schedule during sampling.
        use_exponential_sigmas (`bool`, defaults to FALSE):
            Whether to use exponential sigmas for step sizes in the noise schedule during sampling.
        use_beta_sigmas (`bool`, defaults to FALSE):
            Whether to use beta sigmas for step sizes in the noise schedule during sampling.
        time_shift_type (`str`, defaults to "exponential"):
            The type of dynamic resolution-dependent timestep shifting to apply. Either "exponential" || "linear".
        stochastic_sampling (`bool`, defaults to FALSE):
            Whether to use stochastic sampling.
    """

    _compatibles <- []
    order <- 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps <- 1000,
        shift <- 1.0,
        use_dynamic_shifting <- FALSE,
        base_shift <- 0.5,
        max_shift <- 1.15,
        base_image_seq_len <- 256,
        max_image_seq_len <- 4096,
        invert_sigmas <- FALSE,
        shift_terminal <- NULL,
        use_karras_sigmas <- FALSE,
        use_exponential_sigmas <- FALSE,
        use_beta_sigmas <- FALSE,
        time_shift_type <- "exponential",
        stochastic_sampling <- FALSE,
    ):
        if (self$config.use_beta_sigmas && ! is_scipy_available()) {
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if (sum([self$config.use_beta_sigmas, self$config.use_exponential_sigmas, self$config.use_karras_sigmas]) > 1) {
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if (time_shift_type ! in {"exponential", "linear"}) {
            raise ValueError("`time_shift_type` must either be 'exponential' || 'linear'.")

        timesteps <- np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps <- torch_from_numpy(timesteps)$to(dtype=torch.float32)

        sigmas <- timesteps / num_train_timesteps
        if (! use_dynamic_shifting) {
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas <- shift * sigmas / (1 + (shift - 1) * sigmas)

        self$timesteps <- sigmas * num_train_timesteps

        self$_step_index <- NULL
        self$_begin_index <- NULL

        self$_shift <- shift

        self$sigmas <- sigmas$to("cpu")  # to avoid too much CPU/GPU communication
        self$sigma_min <- self$sigmas[-1].item()
        self$sigma_max <- self$sigmas[0].item()

    @property
    def shift(self):
        """
        The value used for shifting.
        """
        return(self$_shift)

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return(self$_step_index)

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return(self$_begin_index)

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index= 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`, defaults to `0`):
                The begin index for the scheduler.
        """
        self$_begin_index <- begin_index

    def set_shift(self, shift):
        self$_shift <- shift

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: torch.FloatTensor,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`torch.FloatTensor`):
                The current timestep in the diffusion chain.
            noise (`torch.FloatTensor`):
                The noise tensor.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas <- self$sigmas$to(device=sample$device, dtype=sample$dtype)

        if (sample$device.type == "mps" && torch_is_floating_point(timestep)) {
            # mps does not support float64
            schedule_timesteps <- self$timesteps$to(sample$device, dtype=torch.float32)
            timestep <- timestep$to(sample$device, dtype=torch.float32)
        } else {
            schedule_timesteps <- self$timesteps$to(sample$device)
            timestep <- timestep$to(sample$device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if (self$is.null(begin_index)) {
            step_indices <- [self$index_for_timestep(t, schedule_timesteps) for t in timestep]
        } else if (self$!is.null(step_index)) {
            # add_noise is called after first denoising step (for inpainting)
            step_indices <- [self$step_index] * timestep.shape[0]
        } else {
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices <- [self$begin_index] * timestep.shape[0]

        sigma <- sigmas[step_indices].flatten()
        while length(sigma.shape) < length(sample.shape):
            sigma <- sigma$unsqueeze(-1)

        sample <- sigma * noise + (1.0 - sigma) * sample

        return(sample)

    def _sigma_to_t(self, sigma):
        return(sigma * self$config.num_train_timesteps)

    def time_shift(self, mu, sigma, t: torch.Tensor):
        if (self$config.time_shift_type == "exponential") {
            return(self$_time_shift_exponential(mu, sigma, t))
        } else if (self$config.time_shift_type == "linear") {
            return(self$_time_shift_linear(mu, sigma, t))

    def stretch_shift_to_terminal(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Stretches && shifts the timestep schedule to ensure it terminates at the configured `shift_terminal` config
        value.

        Reference:
        https:%/%github.com/Lightricks/LTX-Video/blob/a01a171f8fe3d99dce2728d60a73fecf4d4238ae/ltx_video/schedulers/rf.py#L51

        Args:
            t (`torch.Tensor`):
                A tensor of timesteps to be stretched && shifted.

        Returns:
            `torch.Tensor`:
                A tensor of adjusted timesteps such that the final value equals `self$config.shift_terminal`.
        """
        one_minus_z <- 1 - t
        scale_factor <- one_minus_z[-1] / (1 - self$config.shift_terminal)
        stretched_t <- 1 - (one_minus_z / scale_factor)
        return(stretched_t)

    def set_timesteps(
        self,
        num_inference_steps <- NULL,
        device: Union[str, torch$device] = NULL,
        sigmas <- NULL,
        mu <- NULL,
        timesteps <- NULL,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` || `torch$device`, *optional*):
                The device to which the timesteps should be moved to. If `NULL`, the timesteps are ! moved.
            sigmas (`List[float]`, *optional*):
                Custom values for sigmas to be used for each diffusion step. If `NULL`, the sigmas are computed
                automatically.
            mu (`float`, *optional*):
                Determines the amount of shifting applied to sigmas when performing resolution-dependent timestep
                shifting.
            timesteps (`List[float]`, *optional*):
                Custom values for timesteps to be used for each diffusion step. If `NULL`, the timesteps are computed
                automatically.
        """
        if (self$config.use_dynamic_shifting && is.null(mu)) {
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is set to be `TRUE`")

        if (!is.null(sigmas) && !is.null(timesteps)) {
            if (length(sigmas) != length(timesteps)) {
                raise ValueError("`sigmas` && `timesteps` should have the same length")

        if (!is.null(num_inference_steps)) {
            if (!is.null(sigmas) && length(sigmas) != num_inference_steps) || (
                !is.null(timesteps) && length(timesteps) != num_inference_steps
            ):
                raise ValueError(
                    "`sigmas` && `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided"
                )
        } else {
            num_inference_steps <- length(sigmas) if !is.null(sigmas) else length(timesteps)

        self$num_inference_steps <- num_inference_steps

        # 1. Prepare default sigmas
        is_timesteps_provided <- !is.null(timesteps)

        if (is_timesteps_provided) {
            timesteps <- np.array(timesteps).astype(np.float32)

        if (is.null(sigmas)) {
            if (is.null(timesteps)) {
                timesteps <- np.linspace(
                    self$_sigma_to_t(self$sigma_max), self$_sigma_to_t(self$sigma_min), num_inference_steps
                )
            sigmas <- timesteps / self$config.num_train_timesteps
        } else {
            sigmas <- np.array(sigmas).astype(np.float32)
            num_inference_steps <- length(sigmas)

        # 2. Perform timestep shifting. Either no shifting is applied, or resolution-dependent shifting of
        #    "exponential" or "linear" type is applied
        if (self$config.use_dynamic_shifting) {
            sigmas <- self$time_shift(mu, 1.0, sigmas)
        } else {
            sigmas <- self$shift * sigmas / (1 + (self$shift - 1) * sigmas)

        # 3. If required, stretch the sigmas schedule to terminate at the configured `shift_terminal` value
        if (self$config.shift_terminal) {
            sigmas <- self$stretch_shift_to_terminal(sigmas)

        # 4. If required, convert sigmas to one of karras, exponential, or beta sigma schedules
        if (self$config.use_karras_sigmas) {
            sigmas <- self$_convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        } else if (self$config.use_exponential_sigmas) {
            sigmas <- self$_convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
        } else if (self$config.use_beta_sigmas) {
            sigmas <- self$_convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        # 5. Convert sigmas and timesteps to tensors and move to specified device
        sigmas <- torch_from_numpy(sigmas)$to(dtype=torch.float32, device=device)
        if (! is_timesteps_provided) {
            timesteps <- sigmas * self$config.num_train_timesteps
        } else {
            timesteps <- torch_from_numpy(timesteps)$to(dtype=torch.float32, device=device)

        # 6. Append the terminal sigma value.
        #    If a model requires inverted sigma schedule for denoising but timesteps without inversion, the
        #    `invert_sigmas` flag can be set to `True`. This case is only required in Mochi
        if (self$config.invert_sigmas) {
            sigmas <- 1.0 - sigmas
            timesteps <- sigmas * self$config.num_train_timesteps
            sigmas <- torch_cat([sigmas, torch_ones(1, device=sigmas$device)])
        } else {
            sigmas <- torch_cat([sigmas, torch_zeros(1, device=sigmas$device)])

        self$timesteps <- timesteps
        self$sigmas <- sigmas
        self$_step_index <- NULL
        self$_begin_index <- NULL

    def index_for_timestep(self, timestep, schedule_timesteps=NULL):
        if (is.null(schedule_timesteps)) {
            schedule_timesteps <- self$timesteps

        indices <- (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos <- 1 if length(indices) > 1 else 0

        return(indices[pos].item())

    def _init_step_index(self, timestep):
        if (self$is.null(begin_index)) {
            if (isinstance(timestep, torch.Tensor)) {
                timestep <- timestep$to(self$timesteps$device)
            self$_step_index <- self$index_for_timestep(timestep)
        } else {
            self$_step_index <- self$_begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn <- 0.0,
        s_tmin <- 0.0,
        s_tmax <- float("inf"),
        s_noise <- 1.0,
        generator <- NULL,
        per_token_timesteps <- NULL,
        return_dict <- TRUE,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether || ! to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] || tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] || `tuple`:
                If return_dict is `TRUE`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            inherits(timestep, "int")
            || isinstance(timestep, torch.IntTensor)
            || isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `seq_along(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is ! supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if (self$is.null(step_index)) {
            self$_init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample <- sample$to(torch.float32)

        if (!is.null(per_token_timesteps)) {
            per_token_sigmas <- per_token_timesteps / self$config.num_train_timesteps

            sigmas <- self$sigmas[:, NULL, NULL]
            lower_mask <- sigmas < per_token_sigmas[NULL] - 1e-6
            lower_sigmas <- lower_mask * sigmas
            lower_sigmas, _ <- lower_sigmas$max(dim=0)

            current_sigma <- per_token_sigmas[..., NULL]
            next_sigma <- lower_sigmas[..., NULL]
            dt <- current_sigma - next_sigma
        } else {
            sigma_idx <- self$step_index
            sigma <- self$sigmas[sigma_idx]
            sigma_next <- self$sigmas[sigma_idx + 1]

            current_sigma <- sigma
            next_sigma <- sigma_next
            dt <- sigma_next - sigma

        if (self$config.stochastic_sampling) {
            x0 <- sample - current_sigma * model_output
            noise <- torch_randn_like(sample)
            prev_sample <- (1.0 - next_sigma) * x0 + next_sigma * noise
        } else {
            prev_sample <- sample + dt * model_output

        # upon completion increase step index by one
        self$_step_index += 1
        if (is.null(per_token_timesteps)) {
            # Cast sample back to model compatible dtype
            prev_sample <- prev_sample$to(model_output$dtype)

        if (! return_dict) {
            return(list(prev_sample,))

        return(FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample))

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """
        Construct the noise schedule as proposed in [Elucidating the Design Space of Diffusion-Based Generative
        Models](https:%/%huggingface.co/papers/2206.00364).

        Args:
            in_sigmas (`torch.Tensor`):
                The input sigma values to be converted.
            num_inference_steps (`int`):
                The number of inference steps to generate the noise schedule for.

        Returns:
            `torch.Tensor`:
                The converted sigma values following the Karras noise schedule.
        """

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if (hasattr(self$config, "sigma_min")) {
            sigma_min <- self$config.sigma_min
        } else {
            sigma_min <- NULL

        if (hasattr(self$config, "sigma_max")) {
            sigma_max <- self$config.sigma_max
        } else {
            sigma_max <- NULL

        sigma_min <- sigma_min if !is.null(sigma_min) else in_sigmas[-1].item()
        sigma_max <- sigma_max if !is.null(sigma_max) else in_sigmas[0].item()

        rho <- 7.0  # 7.0 is the value used in the paper
        ramp <- np.linspace(0, 1, num_inference_steps)
        min_inv_rho <- sigma_min ^ (1 / rho)
        max_inv_rho <- sigma_max ^ (1 / rho)
        sigmas <- (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ^ rho
        return(sigmas)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_exponential
    def _convert_to_exponential(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """
        Construct an exponential noise schedule.

        Args:
            in_sigmas (`torch.Tensor`):
                The input sigma values to be converted.
            num_inference_steps (`int`):
                The number of inference steps to generate the noise schedule for.

        Returns:
            `torch.Tensor`:
                The converted sigma values following an exponential schedule.
        """

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if (hasattr(self$config, "sigma_min")) {
            sigma_min <- self$config.sigma_min
        } else {
            sigma_min <- NULL

        if (hasattr(self$config, "sigma_max")) {
            sigma_max <- self$config.sigma_max
        } else {
            sigma_max <- NULL

        sigma_min <- sigma_min if !is.null(sigma_min) else in_sigmas[-1].item()
        sigma_max <- sigma_max if !is.null(sigma_max) else in_sigmas[0].item()

        sigmas <- np$exp(np.linspace(math$log(sigma_max), math$log(sigma_min), num_inference_steps))
        return(sigmas)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_beta
    def _convert_to_beta(
        self, in_sigmas: torch.Tensor, num_inference_steps, alpha <- 0.6, beta= 0.6
    ) -> torch.Tensor:
        """
        Construct a beta noise schedule as proposed in [Beta Sampling is All You
        Need](https:%/%huggingface.co/papers/2407.12173).

        Args:
            in_sigmas (`torch.Tensor`):
                The input sigma values to be converted.
            num_inference_steps (`int`):
                The number of inference steps to generate the noise schedule for.
            alpha (`float`, *optional*, defaults to `0.6`):
                The alpha parameter for the beta distribution.
            beta (`float`, *optional*, defaults to `0.6`):
                The beta parameter for the beta distribution.

        Returns:
            `torch.Tensor`:
                The converted sigma values following a beta distribution schedule.
        """

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if (hasattr(self$config, "sigma_min")) {
            sigma_min <- self$config.sigma_min
        } else {
            sigma_min <- NULL

        if (hasattr(self$config, "sigma_max")) {
            sigma_max <- self$config.sigma_max
        } else {
            sigma_max <- NULL

        sigma_min <- sigma_min if !is.null(sigma_min) else in_sigmas[-1].item()
        sigma_max <- sigma_max if !is.null(sigma_max) else in_sigmas[0].item()

        sigmas <- np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return(sigmas)

    def _time_shift_exponential(self, mu, sigma, t):
        return(math$exp(mu) / (math$exp(mu) + (1 / t - 1) ^ sigma))

    def _time_shift_linear(self, mu, sigma, t):
        return(mu / (mu + (1 / t - 1) ^ sigma))

    def __len__(self):
        return(self$config.num_train_timesteps)

