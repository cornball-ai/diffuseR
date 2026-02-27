# Converted from PyTorch by pyrotechnics
# Review: indexing (0->1 based), integer literals (add L),
# and block structure (braces may need adjustment)

# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
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

# import copy
# import inspect
# from typing import Any, Callable, Dict, List, Optional, Union

# import numpy as np
# import torch
# from transformers import Gemma3ForConditionalGeneration, GemmaTokenizer, GemmaTokenizerFast

# from ...callbacks import MultiPipelineCallbacks, PipelineCallback
# from ...loaders import FromSingleFileMixin, LTX2LoraLoaderMixin
# from ...models.autoencoders import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video
# from ...models.transformers import LTX2VideoTransformer3DModel
# from ...schedulers import FlowMatchEulerDiscreteScheduler
# from ...utils import is_torch_xla_available, logging, replace_example_docstring
# from ...utils.torch_utils import randn_tensor
# from ...video_processor import VideoProcessor
# from ..pipeline_utils import DiffusionPipeline
# from .connectors import LTX2TextConnectors
# from .pipeline_output import LTX2PipelineOutput
# from .vocoder import LTX2Vocoder


if (is_torch_xla_available()) {
# import torch_xla.core.xla_model as xm

    XLA_AVAILABLE <- TRUE
} else {
    XLA_AVAILABLE <- FALSE

logger <- logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING <- """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import LTX2Pipeline
        >>> from diffusers.pipelines.ltx2.export_utils import encode_video

        >>> pipe <- LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()

        >>> prompt <- "A woman with long brown hair && light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket && has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm && natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage"
        >>> negative_prompt <- "worst quality, inconsistent motion, blurry, jittery, distorted"

        >>> frame_rate <- 24.0
        >>> video, audio <- pipe(
        ...     prompt <- prompt,
        ...     negative_prompt <- negative_prompt,
        ...     width <- 768,
        ...     height <- 512,
        ...     num_frames <- 121,
        ...     frame_rate <- frame_rate,
        ...     num_inference_steps <- 40,
        ...     guidance_scale <- 4.0,
        ...     output_type <- "np",
        ...     return_dict <- FALSE,
        ... )
        >>> video <- (video * 255).round().astype("uint8")
        >>> video <- torch_from_numpy(video)

        >>> encode_video(
        ...     video[0],
        ...     fps <- frame_rate,
        ...     audio <- audio[0].float()$cpu(),
        ...     audio_sample_rate <- pipe.vocoder.config.output_sampling_rate,  # should be 24000
        ...     output_path <- "video.mp4",
        ... )
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len <- 256,
    max_seq_len <- 4096,
    base_shift <- 0.5,
    max_shift <- 1.15,
):
    m <- (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b <- base_shift - m * base_seq_len
    mu <- image_seq_len * m + b
    return(mu)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps <- NULL,
    device <- NULL,
    timesteps <- NULL,
    sigmas <- NULL,
    ^kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method && retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `NULL`.
        device (`str` || `torch$device`, *optional*):
            The device to which the timesteps should be moved to. If `NULL`, the timesteps are ! moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` && `sigmas` must be `NULL`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` && `timesteps` must be `NULL`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler && the
        second element is the number of inference steps.
    """
    if (!is.null(timesteps) && !is.null(sigmas)) {
        raise ValueError("Only one of `timesteps` || `sigmas` can be passed. Please choose one to set custom values")
    if (!is.null(timesteps)) {
        accepts_timesteps <- "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if (! accepts_timesteps) {
            raise ValueError(
                sprintf("The current scheduler class {scheduler.__class__}'s `set_timesteps` does ! support custom")
                sprintf(" timestep schedules. Please check whether you are using the correct scheduler.")
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, ^kwargs)
        timesteps <- scheduler.timesteps
        num_inference_steps <- length(timesteps)
    } else if (!is.null(sigmas)) {
        accept_sigmas <- "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if (! accept_sigmas) {
            raise ValueError(
                sprintf("The current scheduler class {scheduler.__class__}'s `set_timesteps` does ! support custom")
                sprintf(" sigmas schedules. Please check whether you are using the correct scheduler.")
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, ^kwargs)
        timesteps <- scheduler.timesteps
        num_inference_steps <- length(timesteps)
    } else {
        scheduler.set_timesteps(num_inference_steps, device=device, ^kwargs)
        timesteps <- scheduler.timesteps
    return(list(timesteps, num_inference_steps))


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality && fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules && Sample Steps are
    Flawed](https:%/%huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text <- noise_pred_text$std(dim=list(range(1, noise_pred_text.ndim)), keepdim=TRUE)
    std_cfg <- noise_cfg$std(dim=list(range(1, noise_cfg.ndim)), keepdim=TRUE)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled <- noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg <- guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return(noise_cfg)


class LTX2Pipeline(DiffusionPipeline, FromSingleFileMixin, LTX2LoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation.

    Reference: https:%/%github.com/Lightricks/LTX-Video

    Args:
        transformer ([`LTXVideoTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLLTXVideo`]):
            Variational Auto-Encoder (VAE) Model to encode && decode images to && from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https:%/%huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https:%/%huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https:%/%huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https:%/%huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
        connectors ([`LTX2TextConnectors`]):
            Text connector stack used to adapt text encoder hidden states for the video && audio branches.
    """

    model_cpu_offload_seq <- "text_encoder->connectors->transformer->vae->audio_vae->vocoder"
    _optional_components <- []
    _callback_tensor_inputs <- ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLLTX2Video,
        audio_vae: AutoencoderKLLTX2Audio,
        text_encoder: Gemma3ForConditionalGeneration,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        connectors: LTX2TextConnectors,
        transformer: LTX2VideoTransformer3DModel,
        vocoder: LTX2Vocoder,
    ):
        super().__init__()

        self$register_modules(
            vae <- vae,
            audio_vae <- audio_vae,
            text_encoder <- text_encoder,
            tokenizer <- tokenizer,
            connectors <- connectors,
            transformer <- transformer,
            vocoder <- vocoder,
            scheduler <- scheduler,
        )

        self$vae_spatial_compression_ratio <- (
            self$vae.spatial_compression_ratio if getattr(self, "vae", NULL) !is.null else 32
        )
        self$vae_temporal_compression_ratio <- (
            self$vae.temporal_compression_ratio if getattr(self, "vae", NULL) !is.null else 8
        )
        # TODO: check whether the MEL compression ratio logic here is corrct
        self$audio_vae_mel_compression_ratio <- (
            self$audio_vae.mel_compression_ratio if getattr(self, "audio_vae", NULL) !is.null else 4
        )
        self$audio_vae_temporal_compression_ratio <- (
            self$audio_vae.temporal_compression_ratio if getattr(self, "audio_vae", NULL) !is.null else 4
        )
        self$transformer_spatial_patch_size <- (
            self$transformer.config.patch_size if getattr(self, "transformer", NULL) !is.null else 1
        )
        self$transformer_temporal_patch_size <- (
            self$transformer.config.patch_size_t if getattr(self, "transformer") !is.null else 1
        )

        self$audio_sampling_rate <- (
            self$audio_vae.config.sample_rate if getattr(self, "audio_vae", NULL) !is.null else 16000
        )
        self$audio_hop_length <- (
            self$audio_vae.config.mel_hop_length if getattr(self, "audio_vae", NULL) !is.null else 160
        )

        self$video_processor <- VideoProcessor(vae_scale_factor=self$vae_spatial_compression_ratio)
        self$tokenizer_max_length <- (
            self$tokenizer.model_max_length if getattr(self, "tokenizer", NULL) !is.null else 1024
        )

    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        device: Union[str, torch$device],
        padding_side <- "left",
        scale_factor <- 8,
        eps <- 1e-6,
    ) -> torch.Tensor:
        """
        Packs && normalizes text encoder hidden states, respecting padding. Normalization is performed per-batch &&
        per-layer in a masked fashion (only over non-padded positions).

        Args:
            text_hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_dim, num_layers)`):
                Per-layer hidden_states from a text encoder (e.g. `Gemma3ForConditionalGeneration`).
            sequence_lengths (`torch.Tensor of shape `(batch_size,)`):
                The number of valid (non-padded) tokens for each batch instance.
            device: (`str` || `torch$device`, *optional*):
                torch device to place the resulting embeddings on
            padding_side: (`str`, *optional*, defaults to `"left"`):
                Whether the text tokenizer performs padding on the `"left"` || `"right"`.
            scale_factor (`int`, *optional*, defaults to `8`):
                Scaling factor to multiply the normalized hidden states by.
            eps (`float`, *optional*, defaults to `1e-6`):
                A small positive value for numerical stability when performing normalization.

        Returns:
            `torch.Tensor` of shape `(batch_size, seq_len, hidden_dim * num_layers)`:
                Normed && flattened text encoder hidden states.
        """
        batch_size, seq_len, hidden_dim, num_layers <- text_hidden_states.shape
        original_dtype <- text_hidden_states$dtype

        # Create padding mask
        token_indices <- torch_arange(seq_len, device=device)$unsqueeze(0)
        if (padding_side == "right") {
            # For right padding, valid tokens are from 0 to sequence_length-1
            mask <- token_indices < sequence_lengths[:, NULL]  # [batch_size, seq_len]
        } else if (padding_side == "left") {
            # For left padding, valid tokens are from (T - sequence_length) to T-1
            start_indices <- seq_len - sequence_lengths[:, NULL]  # [batch_size, 1]
            mask <- token_indices >= start_indices  # [B, T]
        } else {
            raise ValueError(sprintf("padding_side must be 'left' || 'right', got {padding_side}"))
        mask <- mask[:, :, NULL, NULL]  # [batch_size, seq_len] --> [batch_size, seq_len, 1, 1]

        # Compute masked mean over non-padding positions of shape (batch_size, 1, 1, seq_len)
        masked_text_hidden_states <- text_hidden_states$masked_fill(~mask, 0.0)
        num_valid_positions <- (sequence_lengths * hidden_dim)$view(batch_size, 1, 1, 1)
        masked_mean <- masked_text_hidden_states$sum(dim=(1, 2), keepdim=TRUE) / (num_valid_positions + eps)

        # Compute min/max over non-padding positions of shape (batch_size, 1, 1 seq_len)
        x_min <- text_hidden_states$masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=TRUE)
        x_max <- text_hidden_states$masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=TRUE)

        # Normalization
        normalized_hidden_states <- (text_hidden_states - masked_mean) / (x_max - x_min + eps)
        normalized_hidden_states <- normalized_hidden_states * scale_factor

        # Pack the hidden states to a 3D tensor (batch_size, seq_len, hidden_dim * num_layers)
        normalized_hidden_states <- normalized_hidden_states$flatten(2)
        mask_flat <- mask$squeeze(-1)$expand(-1, -1, hidden_dim * num_layers)
        normalized_hidden_states <- normalized_hidden_states$masked_fill(~mask_flat, 0.0)
        normalized_hidden_states <- normalized_hidden_states$to(dtype=original_dtype)
        return(normalized_hidden_states)

    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt <- 1,
        max_sequence_length <- 1024,
        scale_factor <- 8,
        device <- NULL,
        dtype <- NULL,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` || `List[str]`, *optional*):
                prompt to be encoded
            device: (`str` || `torch$device`):
                torch device to place the resulting embeddings on
            dtype: (`torch$dtype`):
                torch dtype to cast the prompt embeds to
            max_sequence_length (`int`, defaults to 1024): Maximum sequence length to use for the prompt.
        """
        device <- device || self$_execution_device
        dtype <- dtype || self$text_encoder$dtype

        prompt <- [prompt] if inherits(prompt, "str") else prompt
        batch_size <- length(prompt)

        if (getattr(self, "tokenizer", NULL) !is.null) {
            # Gemma expects left padding for chat-style prompts
            self$tokenizer.padding_side <- "left"
            if (self$tokenizer.is.null(pad_token)) {
                self$tokenizer.pad_token <- self$tokenizer.eos_token

        prompt <- [p.strip() for p in prompt]
        text_inputs <- self$tokenizer(
            prompt,
            padding <- "max_length",
            max_length <- max_sequence_length,
            truncation <- TRUE,
            add_special_tokens <- TRUE,
            return_tensors <- "pt",
        )
        text_input_ids <- text_inputs.input_ids
        prompt_attention_mask <- text_inputs.attention_mask
        text_input_ids <- text_input_ids$to(device)
        prompt_attention_mask <- prompt_attention_mask$to(device)

        text_encoder_outputs <- self$text_encoder(
            input_ids <- text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=TRUE
        )
        text_encoder_hidden_states <- text_encoder_outputs.hidden_states
        text_encoder_hidden_states <- torch_stack(text_encoder_hidden_states, dim=-1)
        sequence_lengths <- prompt_attention_mask$sum(dim=-1)

        prompt_embeds <- self$_pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device <- device,
            padding_side <- self$tokenizer.padding_side,
            scale_factor <- scale_factor,
        )
        prompt_embeds <- prompt_embeds$to(dtype=dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ <- prompt_embeds.shape
        prompt_embeds <- prompt_embeds$repeat(1, num_videos_per_prompt, 1)
        prompt_embeds <- prompt_embeds$view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask <- prompt_attention_mask$view(batch_size, -1)
        prompt_attention_mask <- prompt_attention_mask$repeat(num_videos_per_prompt, 1)

        return(list(prompt_embeds, prompt_attention_mask))

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt <- NULL,
        do_classifier_free_guidance <- TRUE,
        num_videos_per_prompt <- 1,
        prompt_embeds <- NULL,
        negative_prompt_embeds <- NULL,
        prompt_attention_mask <- NULL,
        negative_prompt_attention_mask <- NULL,
        max_sequence_length <- 1024,
        scale_factor <- 8,
        device <- NULL,
        dtype <- NULL,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` || `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` || `List[str]`, *optional*):
                The prompt || prompts ! to guide the image generation. If ! defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when ! using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `TRUE`):
                Whether to use classifier free guidance || !.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If !
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If ! provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch$device`, *optional*):
                torch device
            dtype: (`torch$dtype`, *optional*):
                torch dtype
        """
        device <- device || self$_execution_device

        prompt <- [prompt] if inherits(prompt, "str") else prompt
        if (!is.null(prompt)) {
            batch_size <- length(prompt)
        } else {
            batch_size <- prompt_embeds.shape[0]

        if (is.null(prompt_embeds)) {
            prompt_embeds, prompt_attention_mask <- self$_get_gemma_prompt_embeds(
                prompt <- prompt,
                num_videos_per_prompt <- num_videos_per_prompt,
                max_sequence_length <- max_sequence_length,
                scale_factor <- scale_factor,
                device <- device,
                dtype <- dtype,
            )

        if (do_classifier_free_guidance && is.null(negative_prompt_embeds)) {
            negative_prompt <- negative_prompt || ""
            negative_prompt <- batch_size * [negative_prompt] if inherits(negative_prompt, "str") else negative_prompt

            if (!is.null(prompt) && type(prompt) is ! type(negative_prompt)) {
                raise TypeError(
                    sprintf("`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=")
                    sprintf(" {type(prompt)}.")
                )
            } else if (batch_size != length(negative_prompt)) {
                raise ValueError(
                    sprintf("`negative_prompt`: {negative_prompt} has batch size {length(negative_prompt)}, but `prompt`:")
                    sprintf(" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches")
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask <- self$_get_gemma_prompt_embeds(
                prompt <- negative_prompt,
                num_videos_per_prompt <- num_videos_per_prompt,
                max_sequence_length <- max_sequence_length,
                scale_factor <- scale_factor,
                device <- device,
                dtype <- dtype,
            )

        return(list(prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask))

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs <- NULL,
        prompt_embeds <- NULL,
        negative_prompt_embeds <- NULL,
        prompt_attention_mask <- NULL,
        negative_prompt_attention_mask <- NULL,
    ):
        if (height % 32 != 0 || width % 32 != 0) {
            raise ValueError(sprintf("`height` && `width` have to be divisible by 32 but are {height} && {width}."))

        if !is.null(callback_on_step_end_tensor_inputs) && ! all(
            k in self$_callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                sprintf("`callback_on_step_end_tensor_inputs` has to be in {self$_callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k ! in self$_callback_tensor_inputs]}")
            )

        if (!is.null(prompt) && !is.null(prompt_embeds)) {
            raise ValueError(
                sprintf("Cannot forward both `prompt`: {prompt} && `prompt_embeds`: {prompt_embeds}. Please make sure to")
                " only forward one of the two."
            )
        } else if (is.null(prompt) && is.null(prompt_embeds)) {
            raise ValueError(
                "Provide either `prompt` || `prompt_embeds`. Cannot leave both `prompt` && `prompt_embeds` undefined."
            )
        } else if (!is.null(prompt) && (! inherits(prompt, "str") && ! inherits(prompt, "list"))) {
            raise ValueError(sprintf("`prompt` has to be of type `str` || `list` but is {type(prompt)}"))

        if (!is.null(prompt_embeds) && is.null(prompt_attention_mask)) {
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if (!is.null(negative_prompt_embeds) && is.null(negative_prompt_attention_mask)) {
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if (!is.null(prompt_embeds) && !is.null(negative_prompt_embeds)) {
            if (prompt_embeds.shape != negative_prompt_embeds.shape) {
                raise ValueError(
                    "`prompt_embeds` && `negative_prompt_embeds` must have the same shape when passed directly, but"
                    sprintf(" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`")
                    sprintf(" {negative_prompt_embeds.shape}.")
                )
            if (prompt_attention_mask.shape != negative_prompt_attention_mask.shape) {
                raise ValueError(
                    "`prompt_attention_mask` && `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    sprintf(" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`")
                    sprintf(" {negative_prompt_attention_mask.shape}.")
                )

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size= 1, patch_size_t= 1) -> torch.Tensor:
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width <- latents.shape
        post_patch_num_frames <- num_frames %/% patch_size_t
        post_patch_height <- height %/% patch_size
        post_patch_width <- width %/% patch_size
        latents <- latents$reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents <- latents$permute(0, 2, 4, 6, 1, 3, 5, 7)$flatten(4, 7)$flatten(1, 3)
        return(latents)

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, num_frames, height, width, patch_size <- 1, patch_size_t= 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size <- latents$size(0)
        latents <- latents$reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents <- latents$permute(0, 4, 1, 5, 2, 6, 3, 7)$flatten(6, 7)$flatten(4, 5)$flatten(2, 3)
        return(latents)

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor <- 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean <- latents_mean$view(1, -1, 1, 1, 1)$to(latents$device, latents$dtype)
        latents_std <- latents_std$view(1, -1, 1, 1, 1)$to(latents$device, latents$dtype)
        latents <- latents * latents_std / scaling_factor + latents_mean
        return(latents)

    @staticmethod
    def _denormalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean <- latents_mean$to(latents$device, latents$dtype)
        latents_std <- latents_std$to(latents$device, latents$dtype)
        return((latents * latents_std) + latents_mean)

    @staticmethod
    def _pack_audio_latents(
        latents: torch.Tensor, patch_size <- NULL, patch_size_t= NULL
    ) -> torch.Tensor:
        # Audio latents shape: [B, C, L, M], where L is the latent audio length and M is the number of mel bins
        if (!is.null(patch_size) && !is.null(patch_size_t)) {
            # Packs the latents into a patch sequence of shape [B, L // p_t * M // p, C * p_t * p] (a ndim=3 tnesor).
            # dim=1 is the effective audio sequence length and dim=2 is the effective audio input feature size.
            batch_size, num_channels, latent_length, latent_mel_bins <- latents.shape
            post_patch_latent_length <- latent_length / patch_size_t
            post_patch_mel_bins <- latent_mel_bins / patch_size
            latents <- latents$reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
            )
            latents <- latents$permute(0, 2, 4, 1, 3, 5)$flatten(3, 5)$flatten(1, 2)
        } else {
            # Packs the latents into a patch sequence of shape [B, L, C * M]. This implicitly assumes a (mel)
            # patch_size of M (all mel bins constitutes a single patch) and a patch_size_t of 1.
            latents <- latents$transpose(1, 2)$flatten(2, 3)  # [B, C, L, M] --> [B, L, C * M]
        return(latents)

    @staticmethod
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length,
        num_mel_bins,
        patch_size <- NULL,
        patch_size_t <- NULL,
    ) -> torch.Tensor:
        # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor of shape [B, C, L, M],
        # where L is the latent audio length and M is the number of mel bins.
        if (!is.null(patch_size) && !is.null(patch_size_t)) {
            batch_size <- latents$size(0)
            latents <- latents$reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
            latents <- latents$permute(0, 3, 1, 4, 2, 5)$flatten(4, 5)$flatten(2, 3)
        } else {
            # Assume [B, S, D] = [B, L, C * M], which implies that patch_size = M and patch_size_t = 1.
            latents <- latents.unflatten(2, (-1, num_mel_bins))$transpose(1, 2)
        return(latents)

    def prepare_latents(
        self,
        batch_size <- 1,
        num_channels_latents <- 128,
        height <- 512,
        width <- 768,
        num_frames <- 121,
        dtype <- NULL,
        device <- NULL,
        generator <- NULL,
        latents <- NULL,
    ) -> torch.Tensor:
        if (!is.null(latents)) {
            return(latents$to(device=device, dtype=dtype))

        height <- height %/% self$vae_spatial_compression_ratio
        width <- width %/% self$vae_spatial_compression_ratio
        num_frames <- (num_frames - 1) %/% self$vae_temporal_compression_ratio + 1

        shape <- (batch_size, num_channels_latents, num_frames, height, width)

        if (inherits(generator, "list") && length(generator) != batch_size) {
            raise ValueError(
                sprintf("You have passed a list of generators of length {length(generator)}, but requested an effective batch")
                sprintf(" size of {batch_size}. Make sure the batch size matches the length of the generators.")
            )

        latents <- randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents <- self$_pack_latents(
            latents, self$transformer_spatial_patch_size, self$transformer_temporal_patch_size
        )
        return(latents)

    def prepare_audio_latents(
        self,
        batch_size <- 1,
        num_channels_latents <- 8,
        num_mel_bins <- 64,
        num_frames <- 121,
        frame_rate <- 25.0,
        sampling_rate <- 16000,
        hop_length <- 160,
        dtype <- NULL,
        device <- NULL,
        generator <- NULL,
        latents <- NULL,
    ) -> torch.Tensor:
        duration_s <- num_frames / frame_rate
        latents_per_second <- (
            float(sampling_rate) / float(hop_length) / float(self$audio_vae_temporal_compression_ratio)
        )
        latent_length <- round(duration_s * latents_per_second)

        if (!is.null(latents)) {
            return(latents$to(device=device, dtype=dtype), latent_length)

        # TODO: confirm whether this logic is correct
        latent_mel_bins <- num_mel_bins %/% self$audio_vae_mel_compression_ratio

        shape <- (batch_size, num_channels_latents, latent_length, latent_mel_bins)

        if (inherits(generator, "list") && length(generator) != batch_size) {
            raise ValueError(
                sprintf("You have passed a list of generators of length {length(generator)}, but requested an effective batch")
                sprintf(" size of {batch_size}. Make sure the batch size matches the length of the generators.")
            )

        latents <- randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents <- self$_pack_audio_latents(latents)
        return(list(latents, latent_length))

    @property
    def guidance_scale(self):
        return(self$_guidance_scale)

    @property
    def guidance_rescale(self):
        return(self$_guidance_rescale)

    @property
    def do_classifier_free_guidance(self):
        return(self$_guidance_scale > 1.0)

    @property
    def num_timesteps(self):
        return(self$_num_timesteps)

    @property
    def current_timestep(self):
        return(self$_current_timestep)

    @property
    def attention_kwargs(self):
        return(self$_attention_kwargs)

    @property
    def interrupt(self):
        return(self$_interrupt)

# NOTE: wrap body in with_no_grad({ ... })
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = NULL,
        negative_prompt <- NULL,
        height <- 512,
        width <- 768,
        num_frames <- 121,
        frame_rate <- 24.0,
        num_inference_steps <- 40,
        timesteps <- NULL,
        guidance_scale <- 4.0,
        guidance_rescale <- 0.0,
        num_videos_per_prompt <- 1,
        generator <- NULL,
        latents <- NULL,
        audio_latents <- NULL,
        prompt_embeds <- NULL,
        prompt_attention_mask <- NULL,
        negative_prompt_embeds <- NULL,
        negative_prompt_attention_mask <- NULL,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale <- NULL,
        output_type <- "pil",
        return_dict <- TRUE,
        attention_kwargs <- NULL,
        callback_on_step_end, NULL]] = NULL,
        callback_on_step_end_tensor_inputs <- ["latents"],
        max_sequence_length <- 1024,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` || `List[str]`, *optional*):
                The prompt || prompts to guide the image generation. If ! defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to `512`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to `768`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, *optional*, defaults to `121`):
                The number of video frames to generate
            frame_rate (`float`, *optional*, defaults to `24.0`):
                The frames per second (FPS) of the generated video.
            num_inference_steps (`int`, *optional*, defaults to 40):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If ! defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to `4.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https:%/%huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https:%/%huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules && Sample Steps are
                Flawed](https:%/%huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules && Sample Steps are
                Flawed](https:%/%huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` || `List[torch.Generator]`, *optional*):
                One || a list of [torch generator(s)](https:%/%pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If ! provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            audio_latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for audio
                generation. Can be used to tweak the same generation with different prompts. If ! provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If !
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If !
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            decode_timestep (`float`, defaults to `0.0`):
                The timestep at which generated video is decoded.
            decode_noise_scale (`float`, defaults to `NULL`):
                The interpolation factor between random noise && denoised latents at the decode timestep.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https:%/%pillow.readthedocs.io/en/stable/): `PIL.Image.Image` || `np.array`.
            return_dict (`bool`, *optional*, defaults to `TRUE`):
                Whether || ! to return a [`~pipelines.ltx.LTX2PipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self$processor` in
                [diffusers.models.attention_processor](https:%/%github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step, timestep,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*, defaults to `["latents"]`):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, *optional*, defaults to `1024`):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ltx.LTX2PipelineOutput`] || `tuple`:
                If `return_dict` is `TRUE`, [`~pipelines.ltx.LTX2PipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """

        if (isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks))) {
            callback_on_step_end_tensor_inputs <- callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self$check_inputs(
            prompt <- prompt,
            height <- height,
            width <- width,
            callback_on_step_end_tensor_inputs <- callback_on_step_end_tensor_inputs,
            prompt_embeds <- prompt_embeds,
            negative_prompt_embeds <- negative_prompt_embeds,
            prompt_attention_mask <- prompt_attention_mask,
            negative_prompt_attention_mask <- negative_prompt_attention_mask,
        )

        self$_guidance_scale <- guidance_scale
        self$_guidance_rescale <- guidance_rescale
        self$_attention_kwargs <- attention_kwargs
        self$_interrupt <- FALSE
        self$_current_timestep <- NULL

        # 2. Define call parameters
        if (!is.null(prompt) && inherits(prompt, "str")) {
            batch_size <- 1
        } else if (!is.null(prompt) && inherits(prompt, "list")) {
            batch_size <- length(prompt)
        } else {
            batch_size <- prompt_embeds.shape[0]

        device <- self$_execution_device

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self$encode_prompt(
            prompt <- prompt,
            negative_prompt <- negative_prompt,
            do_classifier_free_guidance <- self$do_classifier_free_guidance,
            num_videos_per_prompt <- num_videos_per_prompt,
            prompt_embeds <- prompt_embeds,
            negative_prompt_embeds <- negative_prompt_embeds,
            prompt_attention_mask <- prompt_attention_mask,
            negative_prompt_attention_mask <- negative_prompt_attention_mask,
            max_sequence_length <- max_sequence_length,
            device <- device,
        )
        if (self$do_classifier_free_guidance) {
            prompt_embeds <- torch_cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask <- torch_cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        additive_attention_mask <- (1 - prompt_attention_mask$to(prompt_embeds$dtype)) * -1000000.0
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask <- self$connectors(
            prompt_embeds, additive_attention_mask, additive_mask <- TRUE
        )

        # 4. Prepare latent variables
        latent_num_frames <- (num_frames - 1) %/% self$vae_temporal_compression_ratio + 1
        latent_height <- height %/% self$vae_spatial_compression_ratio
        latent_width <- width %/% self$vae_spatial_compression_ratio
        video_sequence_length <- latent_num_frames * latent_height * latent_width

        num_channels_latents <- self$transformer.config.in_channels
        latents <- self$prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        num_mel_bins <- self$audio_vae.config.mel_bins if getattr(self, "audio_vae", NULL) !is.null else 64
        latent_mel_bins <- num_mel_bins %/% self$audio_vae_mel_compression_ratio

        num_channels_latents_audio <- (
            self$audio_vae.config.latent_channels if getattr(self, "audio_vae", NULL) !is.null else 8
        )
        audio_latents, audio_num_frames <- self$prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents <- num_channels_latents_audio,
            num_mel_bins <- num_mel_bins,
            num_frames <- num_frames,  # Video frames, audio frames will be calculated from this
            frame_rate <- frame_rate,
            sampling_rate <- self$audio_sampling_rate,
            hop_length <- self$audio_hop_length,
            dtype <- torch.float32,
            device <- device,
            generator <- generator,
            latents <- audio_latents,
        )

        # 5. Prepare timesteps
        sigmas <- np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu <- calculate_shift(
            video_sequence_length,
            self$scheduler.config.get("base_image_seq_len", 1024),
            self$scheduler.config.get("max_image_seq_len", 4096),
            self$scheduler.config.get("base_shift", 0.95),
            self$scheduler.config.get("max_shift", 2.05),
        )
        # For now, duplicate the scheduler for use with the audio latents
        audio_scheduler <- copy.deepcopy(self$scheduler)
        _, _ <- retrieve_timesteps(
            audio_scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas <- sigmas,
            mu <- mu,
        )
        timesteps, num_inference_steps <- retrieve_timesteps(
            self$scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas <- sigmas,
            mu <- mu,
        )
        num_warmup_steps <- max(length(timesteps) - num_inference_steps * self$scheduler.order, 0)
        self$_num_timesteps <- length(timesteps)

        # 6. Prepare micro-conditions
        rope_interpolation_scale <- (
            self$vae_temporal_compression_ratio / frame_rate,
            self$vae_spatial_compression_ratio,
            self$vae_spatial_compression_ratio,
        )
        # Pre-compute video and audio positional ids as they will be the same at each step of the denoising loop
        video_coords <- self$transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents$device, fps <- frame_rate
        )
        audio_coords <- self$transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents$device
        )

        # 7. Denoising loop
        with self$progress_bar(total=num_inference_steps) as progress_bar:
            # TODO: tuple unpacking in for loop
            for (            for i, t in enumerate(timesteps): in for i, t in seq_along(timesteps):) {
                if (self$interrupt) {
                    continue

                self$_current_timestep <- t

                latent_model_input <- torch_cat(c(latents) * 2) if self$do_classifier_free_guidance else latents
                latent_model_input <- latent_model_input$to(prompt_embeds$dtype)
                audio_latent_model_input <- (
                    torch_cat(c(audio_latents) * 2) if self$do_classifier_free_guidance else audio_latents
                )
                audio_latent_model_input <- audio_latent_model_input$to(prompt_embeds$dtype)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep <- t$expand(latent_model_input.shape[0])

                with self$transformer.cache_context("cond_uncond"):
                    noise_pred_video, noise_pred_audio <- self$transformer(
                        hidden_states <- latent_model_input,
                        audio_hidden_states <- audio_latent_model_input,
                        encoder_hidden_states <- connector_prompt_embeds,
                        audio_encoder_hidden_states <- connector_audio_prompt_embeds,
                        timestep <- timestep,
                        encoder_attention_mask <- connector_attention_mask,
                        audio_encoder_attention_mask <- connector_attention_mask,
                        num_frames <- latent_num_frames,
                        height <- latent_height,
                        width <- latent_width,
                        fps <- frame_rate,
                        audio_num_frames <- audio_num_frames,
                        video_coords <- video_coords,
                        audio_coords <- audio_coords,
                        # rope_interpolation_scale=rope_interpolation_scale,
                        attention_kwargs <- attention_kwargs,
                        return_dict <- FALSE,
                    )
                noise_pred_video <- noise_pred_video$float()
                noise_pred_audio <- noise_pred_audio$float()

                if (self$do_classifier_free_guidance) {
                    noise_pred_video_uncond, noise_pred_video_text <- noise_pred_video$chunk(2)
                    noise_pred_video <- noise_pred_video_uncond + self$guidance_scale * (
                        noise_pred_video_text - noise_pred_video_uncond
                    )

                    noise_pred_audio_uncond, noise_pred_audio_text <- noise_pred_audio$chunk(2)
                    noise_pred_audio <- noise_pred_audio_uncond + self$guidance_scale * (
                        noise_pred_audio_text - noise_pred_audio_uncond
                    )

                    if (self$guidance_rescale > 0) {
                        # Based on 3.4. in https://huggingface.co/papers/2305.08891
                        noise_pred_video <- rescale_noise_cfg(
                            noise_pred_video, noise_pred_video_text, guidance_rescale <- self$guidance_rescale
                        )
                        noise_pred_audio <- rescale_noise_cfg(
                            noise_pred_audio, noise_pred_audio_text, guidance_rescale <- self$guidance_rescale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents <- self$scheduler.step(noise_pred_video, t, latents, return_dict=FALSE)[0]
                # NOTE: for now duplicate scheduler for audio latents in case self.scheduler sets internal state in
                # the step method (such as _step_index)
                audio_latents <- audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=FALSE)[0]

                if (!is.null(callback_on_step_end)) {
                    callback_kwargs <- {}
                    for (k in callback_on_step_end_tensor_inputs) {
                        callback_kwargs[k] = locals()[k]
                    callback_outputs <- callback_on_step_end(self, i, t, callback_kwargs)

                    latents <- callback_outputs.pop("latents", latents)
                    prompt_embeds <- callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if (i == length(timesteps) - 1 || ((i + 1) > num_warmup_steps && (i + 1) % self$scheduler.order == 0)) {
                    progress_bar.update()

                if (XLA_AVAILABLE) {
                    xm.mark_step()

        latents <- self$_unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self$transformer_spatial_patch_size,
            self$transformer_temporal_patch_size,
        )
        latents <- self$_denormalize_latents(
            latents, self$vae.latents_mean, self$vae.latents_std, self$vae.config.scaling_factor
        )

        audio_latents <- self$_denormalize_audio_latents(
            audio_latents, self$audio_vae.latents_mean, self$audio_vae.latents_std
        )
        audio_latents <- self$_unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)

        if (output_type == "latent") {
            video <- latents
            audio <- audio_latents
        } else {
            latents <- latents$to(prompt_embeds$dtype)

            if (! self$vae.config.timestep_conditioning) {
                timestep <- NULL
            } else {
                noise <- randn_tensor(latents.shape, generator=generator, device=device, dtype=latents$dtype)
                if (! inherits(decode_timestep, "list")) {
                    decode_timestep <- [decode_timestep] * batch_size
                if (is.null(decode_noise_scale)) {
                    decode_noise_scale <- decode_timestep
                } else if (! inherits(decode_noise_scale, "list")) {
                    decode_noise_scale <- [decode_noise_scale] * batch_size

                timestep <- torch_tensor(decode_timestep, device=device, dtype=latents$dtype)
                decode_noise_scale <- torch_tensor(decode_noise_scale, device=device, dtype=latents$dtype)[
                    :, NULL, NULL, NULL, NULL
                ]
                latents <- (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            latents <- latents$to(self$vae$dtype)
            video <- self$vae.decode(latents, timestep, return_dict=FALSE)[0]
            video <- self$video_processor.postprocess_video(video, output_type=output_type)

            audio_latents <- audio_latents$to(self$audio_vae$dtype)
            generated_mel_spectrograms <- self$audio_vae.decode(audio_latents, return_dict=FALSE)[0]
            audio <- self$vocoder(generated_mel_spectrograms)

        # Offload all models
        self$maybe_free_model_hooks()

        if (! return_dict) {
            return(list(video, audio))

        return(LTX2PipelineOutput(frames=video, audio=audio))

