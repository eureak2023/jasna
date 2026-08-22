"""Still-image mosaic removal with an arbitrary SD 1.5 *inpainting* checkpoint.

Third still-image engine, alongside :mod:`jasna.image_restore` (the supporter
``sd-15-jav`` model) and :mod:`jasna.image_restore_video` (BasicVSR++).

Why a separate engine instead of pointing ``sd-15-jav`` at another checkpoint:
that path forces ``prediction_type="v_prediction"`` with ``rescale_betas_zero_snr``
and feeds the UNet a hand-built 9-channel tensor with a precomputed null
embedding. Community checkpoints are epsilon-prediction, so they would decode to
noise there. Here we let ``from_single_file`` bring the checkpoint's own
scheduler, VAE and text encoder, and drive it through the stock diffusers
inpainting pipeline — so anything that works in A1111/ComfyUI works here.

The checkpoint must be a real **inpainting** variant: its UNet takes 9 input
channels (4 latent + 1 mask + 4 masked-image latents). A regular 4-channel model
loads and then fails on the first step. ``check_inpainting_checkpoint`` reports
that up front instead.

Detection, crop geometry and compositing are shared with the supporter path
(``prepare_image_restore`` / ``paste_back``), so only the restoration step
differs between the two diffusion engines.

Unlike BasicVSR++ this *generates* what it cannot see: expect a sharp, plausible
result that is not what was actually behind the mosaic.
"""

from __future__ import annotations

import json
import logging
import struct
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = ""
DEFAULT_NEGATIVE_PROMPT = "blurry, mosaic, pixelated, censored, low quality, watermark, text"
DEFAULT_GUIDANCE = 7.0
# jasna's supporter model runs at 512; the crop helpers are shared, so match it.
SD15_RESTORATION_SIZE = 512


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def check_inpainting_checkpoint(path: Path) -> int | None:
    """Return the UNet's input-channel count, or None if it cannot be read.

    9 = inpainting checkpoint (usable), 4 = ordinary text-to-image model.
    Reads only the safetensors header, so it costs nothing on a 4 GB file.
    """
    path = Path(path)
    if path.suffix.lower() != ".safetensors":
        return None
    try:
        header = _safetensors_header(path)
    except Exception:  # noqa: BLE001 - a malformed header is simply "unknown"
        return None
    for key, meta in header.items():
        if key.endswith("diffusion_model.input_blocks.0.0.weight") or key == "conv_in.weight":
            shape = meta.get("shape") or []
            if len(shape) >= 2:
                return int(shape[1])
    return None


def _load_lora(pipe, lora_path: Path, lora_scale: float) -> None:
    """Fuse a LoRA, tolerating text-encoder halves diffusers cannot map.

    Slider LoRAs often carry ``lora_te_*`` weights in a layout that trips
    diffusers' text-encoder loader ("list index out of range"). The UNet half is
    what matters for inpainting, so retry with the text-encoder keys dropped
    rather than failing the whole run.
    """
    if not lora_path.exists():
        raise FileNotFoundError(str(lora_path))
    try:
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
    except Exception as exc:  # noqa: BLE001 - any loader error triggers the fallback
        from safetensors.torch import load_file

        state = load_file(str(lora_path))
        unet_only = {k: v for k, v in state.items() if not k.startswith("lora_te")}
        if len(unet_only) == len(state) or not unet_only:
            raise
        logger.warning(
            "LoRA %s failed to load in full (%s); retrying with the %d UNet tensors only",
            lora_path.name, type(exc).__name__, len(unet_only),
        )
        pipe.load_lora_weights(unet_only)
    pipe.fuse_lora(lora_scale=lora_scale)
    logger.info("Fused LoRA %s (scale %.2f)", lora_path.name, lora_scale)


def load_inpaint_pipeline(
    model_path: Path,
    device: torch.device,
    fp16: bool,
    *,
    lora_path: Path | None = None,
    lora_scale: float = 1.0,
):
    from diffusers import StableDiffusionInpaintPipeline

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    channels = check_inpainting_checkpoint(model_path)
    if channels == 4:
        raise ValueError(
            f"{model_path.name} is a regular 4-channel SD 1.5 model, not an inpainting one. "
            "Pick a checkpoint whose UNet takes 9 input channels (usually named '-inpainting')."
        )
    if channels not in (9, None):
        raise ValueError(f"{model_path.name}: unexpected UNet input channels {channels}, need 9")

    dtype = torch.float16 if fp16 else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_single_file(
        str(model_path),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    if lora_path is not None:
        _load_lora(pipe, Path(lora_path), float(lora_scale))

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # The VAE is the memory hog at 512; slicing costs ~nothing and keeps 4K inputs safe.
    pipe.vae.enable_slicing()
    logger.info(
        "SD 1.5 inpainting pipeline loaded from %s (fp16=%s, scheduler=%s)",
        model_path.name, fp16, type(pipe.scheduler).__name__,
    )
    return pipe


def restore_prepared_sd15(
    prepared,
    pipe,
    *,
    steps: int,
    strength: float,
    guidance: float,
    seed: int,
    prompt: str,
    negative_prompt: str,
) -> np.ndarray:
    """Inpaint every detected group of ``prepared`` and composite back.

    ``prepared`` is a :class:`jasna.image_restore.PreparedImageRestore`, so the
    crop geometry is identical to the supporter model's.
    """
    from jasna.sd15_crop_utils import paste_back

    if not prepared.groups:
        return prepared.img_chw_u8.copy()

    device = pipe.device
    result = prepared.img_rgb.copy()
    for group in prepared.groups:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=group.mosaic_01.to(device=device, dtype=torch.float32),
            mask_image=group.mask_01.to(device=device, dtype=torch.float32),
            height=SD15_RESTORATION_SIZE,
            width=SD15_RESTORATION_SIZE,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            strength=float(strength),
            generator=generator,
            output_type="pt",
        ).images[0]
        restored_rgb = (
            out.float().clamp(0, 1).mul(255.0).round().to(torch.uint8)
            .permute(1, 2, 0).cpu().numpy()
        )
        paste_back(
            result, restored_rgb, group.crop_bbox,
            group.content_h, group.content_w, group.group_mask,
        )
    return np.ascontiguousarray(result.transpose(2, 0, 1))


def run_image_jobs_sd15_custom(args, jobs: list[tuple[Path, Path]], progress_callback=None) -> None:
    """Batch entry point mirroring ``image_restore._run_image_jobs``."""
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.image_restore import prepare_image_restore, variant_output_paths
    from jasna.media import image_io
    from jasna.mosaic.detection_registry import (
        build_detection_model,
        coerce_detection_model_name,
        recommended_score_threshold,
        require_detection_model_weights,
    )

    if not jobs:
        return

    model_path = str(getattr(args, "image_model_path", "") or "").strip()
    if not model_path:
        raise ValueError(
            "--image-restoration-model-name sd15-custom requires --image-model-path "
            "<checkpoint.safetensors>"
        )

    device = torch.device(str(args.device))
    fp16 = bool(args.fp16)
    batch_size = int(args.batch_size)

    detection_model_name = coerce_detection_model_name(str(args.detection_model))
    score_threshold = float(
        recommended_score_threshold(detection_model_name)
        if args.detection_score_threshold is None
        else args.detection_score_threshold
    )
    has_explicit_path = bool(str(args.detection_model_path).strip())
    detection_model_path = (
        Path(str(args.detection_model_path))
        if has_explicit_path
        else require_detection_model_weights(detection_model_name)
    )
    if not detection_model_path.exists():
        raise FileNotFoundError(str(detection_model_path))

    ensure_engines_compiled(EngineCompilationRequest(
        device=str(device),
        fp16=fp16,
        detection=True,
        detection_model_name=detection_model_name,
        detection_model_path=str(detection_model_path),
        detection_batch_size=batch_size,
    ))

    detector = build_detection_model(
        detection_model_name,
        detection_model_path,
        batch_size=batch_size,
        device=device,
        score_threshold=score_threshold,
        fp16=fp16,
    )
    lora = str(getattr(args, "image_lora_path", "") or "").strip()
    pipe = load_inpaint_pipeline(
        Path(model_path), device, fp16,
        lora_path=Path(lora) if lora else None,
        lora_scale=float(getattr(args, "image_lora_scale", 1.0)),
    )

    num_variants = max(1, int(args.sd15_variants))
    try:
        for i, (input_path, output_base) in enumerate(jobs, start=1):
            if progress_callback is not None:
                progress_callback(i, input_path, output_base)
            img = image_io.read_image_rgb_chw(input_path)
            with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                prepared = prepare_image_restore(img, detector, device=device, fp16=fp16)
                outputs = [
                    restore_prepared_sd15(
                        prepared, pipe,
                        steps=int(args.sd15_steps),
                        strength=float(args.sd15_strength),
                        guidance=float(getattr(args, "sd15_guidance", DEFAULT_GUIDANCE)),
                        seed=int(args.sd15_seed) + v,
                        prompt=str(getattr(args, "image_prompt", DEFAULT_PROMPT)),
                        negative_prompt=str(
                            getattr(args, "image_negative_prompt", DEFAULT_NEGATIVE_PROMPT)
                        ),
                    )
                    for v in range(num_variants)
                ]
            for path, out in zip(variant_output_paths(output_base, num_variants), outputs):
                image_io.write_image_rgb_chw(path, out)
                logger.info("Wrote %s", path)
    finally:
        detector.close()
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()
