"""Z-Image Pipeline."""

import inspect
from typing import List, Optional, Union

from loguru import logger
import torch

from config import (
    BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT,
    DEFAULT_CFG_TRUNCATION,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_WIDTH,
    MAX_IMAGE_SEQ_LEN,
    MAX_SHIFT,
)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The scheduler does not support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(f"The scheduler does not support custom sigmas schedules.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.no_grad()
def generate(
    transformer,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    prompt: Union[str, List[str]],
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    cfg_normalization: bool = False,
    cfg_truncation: float = DEFAULT_CFG_TRUNCATION,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    output_type: str = "pil",
):
    device = next(transformer.parameters()).device

    if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    else:
        vae_scale_factor = 8
    vae_scale = vae_scale_factor * 2

    if height % vae_scale != 0:
        raise ValueError(f"Height must be divisible by {vae_scale} (got {height}).")
    if width % vae_scale != 0:
        raise ValueError(f"Width must be divisible by {vae_scale} (got {width}).")

    if isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]
    else:
        batch_size = len(prompt)

    do_classifier_free_guidance = guidance_scale > 1.0
    logger.info(
        f"Generating image: {height}x{width}, steps={num_inference_steps}, cfg={guidance_scale}"
    )

    formatted_prompts = []
    for p in prompt:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted_prompts.append(formatted_prompt)

    text_inputs = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_list = []
    for i in range(len(prompt_embeds)):
        prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])

    negative_prompt_embeds_list = []
    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt = ["" for _ in prompt]
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        neg_formatted = []
        for p in negative_prompt:
            messages = [{"role": "user", "content": p}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            neg_formatted.append(formatted_prompt)

        neg_inputs = tokenizer(
            neg_formatted,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        neg_input_ids = neg_inputs.input_ids.to(device)
        neg_masks = neg_inputs.attention_mask.to(device).bool()

        neg_embeds = text_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        for i in range(len(neg_embeds)):
            negative_prompt_embeds_list.append(neg_embeds[i][neg_masks[i]])

    if num_images_per_prompt > 1:
        prompt_embeds_list = [
            pe for pe in prompt_embeds_list for _ in range(num_images_per_prompt)
        ]
        if do_classifier_free_guidance:
            negative_prompt_embeds_list = [
                npe
                for npe in negative_prompt_embeds_list
                for _ in range(num_images_per_prompt)
            ]

    height_latent = 2 * (int(height) // vae_scale)
    width_latent = 2 * (int(width) // vae_scale)
    shape = (
        batch_size * num_images_per_prompt,
        transformer.in_channels,
        height_latent,
        width_latent,
    )

    latents = torch.randn(
        shape, generator=generator, device=device, dtype=torch.float32
    )

    actual_batch_size = batch_size * num_images_per_prompt
    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    scheduler_kwargs = {"mu": mu}
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=None,
        **scheduler_kwargs,
    )

    logger.info(f"Sampling loop start: {num_inference_steps} steps")

    from tqdm import tqdm

    # Denoising loop with progress bar
    for i, t in enumerate(tqdm(timesteps, desc="Denoising", total=len(timesteps))):
        # If current t is 0 and it's the last step, skip computation
        if t == 0 and i == len(timesteps) - 1:
            logger.debug(
                f"Step {i+1}/{num_inference_steps} | t: {t.item():.2f} | Skipping last step"
            )
            continue

        timestep = t.expand(latents.shape[0])
        timestep = (1000 - timestep) / 1000
        t_norm = timestep[0].item()

        current_guidance_scale = guidance_scale
        if (
            do_classifier_free_guidance
            and cfg_truncation is not None
            and float(cfg_truncation) <= 1
        ):
            if t_norm > cfg_truncation:
                current_guidance_scale = 0.0

        apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

        if apply_cfg:
            latents_typed = latents.to(
                transformer.dtype
                if hasattr(transformer, "dtype")
                else next(transformer.parameters()).dtype
            )
            latent_model_input = latents_typed.repeat(2, 1, 1, 1)
            prompt_embeds_model_input = prompt_embeds_list + negative_prompt_embeds_list
            timestep_model_input = timestep.repeat(2)
        else:
            latent_model_input = latents.to(next(transformer.parameters()).dtype)
            prompt_embeds_model_input = prompt_embeds_list
            timestep_model_input = timestep

        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        model_out_list = transformer(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
        )[0]

        if apply_cfg:
            pos_out = model_out_list[:actual_batch_size]
            neg_out = model_out_list[actual_batch_size:]
            noise_pred = []
            for j in range(actual_batch_size):
                pos = pos_out[j].float()
                neg = neg_out[j].float()
                pred = pos + current_guidance_scale * (pos - neg)

                if cfg_normalization and float(cfg_normalization) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(pos)
                    new_pos_norm = torch.linalg.vector_norm(pred)
                    max_new_norm = ori_pos_norm * float(cfg_normalization)
                    if new_pos_norm > max_new_norm:
                        pred = pred * (max_new_norm / new_pos_norm)
                noise_pred.append(pred)
            noise_pred = torch.stack(noise_pred, dim=0)
        else:
            noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

        noise_pred = -noise_pred.squeeze(2)
        latents = scheduler.step(
            noise_pred.to(torch.float32), t, latents, return_dict=False
        )[0]
        assert latents.dtype == torch.float32

    if output_type == "latent":
        return latents

    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor
    image = vae.decode(latents, return_dict=False)[0]

    if output_type == "pil":
        from PIL import Image

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(img) for img in image]

    return image
