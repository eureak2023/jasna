"""Still-image mosaic removal with the *video* restoration model.

Companion to :mod:`jasna.image_restore` (SD 1.5 inpainting). Instead of
generating new content with a diffusion model, this path runs the very same
BasicVSR++ mosaic restorer the video pipeline uses, on a one-image "clip".

Why this works at all: BasicVSR++ is recurrent over time, but nothing in the
crop/restore/blend *geometry* is temporal — ``extract_crop``,
``prepare_crops_for_restoration`` and the compositing in
``BlendBuffer._apply_blend`` are all per-frame. The only missing piece for a
still image is the clip itself, and a clip of N identical crops is exactly what
the model sees for a motionless scene, i.e. in distribution. The *middle*
restored frame is the one kept: the ends of a clip receive the least propagated
context, the same reason the video pipeline discards clip margins.

Trade-off vs. the video path: a single frame carries no extra information to
fuse, so this is closer to a learned deblur of the mosaic than to the
multi-frame reconstruction you get on video. In exchange it needs no extra
download and no license — it reuses the restoration weights already on disk,
unlike the ~6.9 GB supporter-gated SD 1.5 bundle.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from jasna.crop_buffer import (
    RESTORATION_SIZE,
    extract_crop,
    prepare_crops_for_restoration,
    scale_offsets,
)
from jasna.restorer.denoise import DenoiseStrength, apply_denoise
from jasna.tracking.blending import create_bbox_blend_mask

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_CLIP_SIZE = 5


def restore_image_video_model(
    img_chw_u8: np.ndarray,
    detector,
    restorer,
    *,
    device: torch.device,
    clip_size: int = DEFAULT_IMAGE_CLIP_SIZE,
    denoise_strength: DenoiseStrength = DenoiseStrength.NONE,
    secondary_restorer=None,
) -> np.ndarray:
    """Detect mosaics, restore each one with BasicVSR++, composite back.

    Returns a ``(3, H, W)`` uint8 RGB array. Zero detections -> a copy of the
    input unchanged (same contract as ``image_restore.restore_image``).
    """
    frame_cpu = torch.from_numpy(img_chw_u8)
    _, frame_h, frame_w = img_chw_u8.shape

    # The detector engine has a fixed batch size; the video pipeline always
    # feeds it a full batch, so replicate the single image rather than build a
    # second engine for batch=1.
    batch = frame_cpu.unsqueeze(0).expand(detector.batch_size, -1, -1, -1).contiguous()
    detections = detector(batch, target_hw=(frame_h, frame_w))
    boxes = detections.boxes_xyxy[0]
    masks = detections.masks[0]

    if len(boxes) == 0:
        logger.info("No mosaics detected; writing input unchanged")
        return img_chw_u8.copy()

    clip_size = max(1, int(clip_size))
    keep_index = clip_size // 2

    frame = frame_cpu.to(device)
    blended = frame.clone()
    secondary_used = 0

    for i in range(len(boxes)):
        raw_crop = extract_crop(frame, boxes[i], frame_h, frame_w)
        crop_h, crop_w = raw_crop.crop_shape
        if crop_h <= 0 or crop_w <= 0:
            continue

        resized_crops, pad_offsets, resize_shapes = prepare_crops_for_restoration(
            [raw_crop], restorer.device, restorer.input_dtype
        )
        # One resize, N references: raw_process only stacks the list, so the
        # repeated entries cost nothing beyond the model's own temporal work.
        primary = restorer.raw_process(resized_crops * clip_size)
        primary = apply_denoise(primary, denoise_strength)
        kept = primary[keep_index:keep_index + 1]
        # The crop is resized into a 256x256 canvas before restoration. A region
        # bigger than that lost resolution on the way in, and the secondary
        # upscaler wins some of it back. A smaller region was never downscaled, so
        # a 256 -> 1024 -> region round trip only softens it — measured on real
        # frames: 1.61x sharper at 362x406, but 0.68x (i.e. worse) at 193x235.
        # So apply the secondary only where it can actually help.
        upscaled_from_256 = max(crop_h, crop_w) > RESTORATION_SIZE
        if secondary_restorer is not None and upscaled_from_256:
            # scale_offsets below reads the patch's own size, so the blend geometry
            # follows the larger patch automatically.
            patch = secondary_restorer.restore(kept, keep_start=0, keep_end=1)[0].to(device)
            secondary_used += 1
        else:
            patch = (
                kept[0]
                .clamp(0, 1)
                .mul(255.0)
                .round()
                .clamp(0, 255)
                .to(dtype=torch.uint8)
            )

        # Same un-pad / resize-back / masked lerp as BlendBuffer._apply_blend.
        pad_offset, resize_shape = scale_offsets(patch, pad_offsets[0], resize_shapes[0])
        pad_left, pad_top = pad_offset
        resize_h, resize_w = resize_shape
        x1, y1, x2, y2 = raw_crop.enlarged_bbox

        unpadded = patch[:, pad_top:pad_top + resize_h, pad_left:pad_left + resize_w]
        resized_back = F.interpolate(
            unpadded.unsqueeze(0).float(),
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        blend_mask = create_bbox_blend_mask(
            masks[i].to(device), (x1, y1, x2, y2), (frame_h, frame_w)
        )
        target = blended[:, y1:y2, x1:x2].float()
        target.lerp_(resized_back, blend_mask.unsqueeze(0)).round_().clamp_(0, 255)
        blended[:, y1:y2, x1:x2] = target.to(blended.dtype)

    if secondary_restorer is not None:
        logger.info(
            "Restored %d mosaic region(s) with the video model (%d via %s, %d already <= %dpx)",
            len(boxes), secondary_used, secondary_restorer.name,
            len(boxes) - secondary_used, RESTORATION_SIZE,
        )
    else:
        logger.info("Restored %d mosaic region(s) with the video model", len(boxes))
    return np.ascontiguousarray(blended.cpu().numpy())


def run_image_jobs_video_model(args, jobs: list[tuple[Path, Path]], progress_callback=None) -> None:
    """Batch entry point mirroring ``image_restore._run_image_jobs``.

    Loads the detector and BasicVSR++ once for the whole job list.
    """
    from jasna.accelerator import is_amd_device
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.media import image_io
    from jasna.mosaic.detection_registry import (
        build_detection_model,
        coerce_detection_model_name,
        recommended_score_threshold,
        require_detection_model_weights,
    )
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.session_factory import build_secondary_restorer

    if not jobs:
        return

    device = torch.device(str(args.device))
    fp16 = bool(args.fp16)
    batch_size = int(args.batch_size)
    clip_size = max(1, int(getattr(args, "image_clip_size", DEFAULT_IMAGE_CLIP_SIZE)))
    denoise_strength = DenoiseStrength(str(args.denoise).lower())

    restoration_model_path = Path(str(args.restoration_model_path))
    if not restoration_model_path.exists():
        raise FileNotFoundError(str(restoration_model_path))

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

    secondary_name = str(args.secondary_restoration).lower()
    if is_amd_device(device) and secondary_name != "none":
        raise ValueError(
            f"Secondary restoration '{secondary_name}' is not available in the AMD build yet"
        )

    compile_result = ensure_engines_compiled(EngineCompilationRequest(
        device=str(device),
        fp16=fp16,
        basicvsrpp=bool(args.compile_basicvsrpp) and not is_amd_device(device),
        basicvsrpp_model_path=str(restoration_model_path),
        detection=True,
        detection_model_name=detection_model_name,
        detection_model_path=str(detection_model_path),
        detection_batch_size=batch_size,
        unet4x=(secondary_name == "unet-4x"),
    ))

    detector = build_detection_model(
        detection_model_name,
        detection_model_path,
        batch_size=batch_size,
        device=device,
        score_threshold=score_threshold,
        fp16=fp16,
    )
    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(restoration_model_path),
        device=device,
        max_clip_size=clip_size,
        use_tensorrt=compile_result.use_basicvsrpp_tensorrt,
        fp16=fp16,
    )
    # The video pipeline's secondary upscalers work per 256x256 patch, so they apply
    # unchanged here; --secondary-restoration selects one (default "none").
    secondary_restorer = build_secondary_restorer(
        SimpleNamespace(
            secondary_restoration=secondary_name,
            tvai_ffmpeg_path=str(args.tvai_ffmpeg_path),
            tvai_model=str(args.tvai_model),
            tvai_scale=int(args.tvai_scale),
            tvai_args=str(args.tvai_args),
            tvai_workers=int(args.tvai_workers),
            tvai_denoise=bool(args.tvai_denoise),
            rtx_scale=int(args.rtx_scale),
            rtx_quality=str(args.rtx_quality).lower(),
            rtx_denoise=str(args.rtx_denoise).lower(),
            rtx_deblur=str(args.rtx_deblur).lower(),
            fp16=fp16,
        ),
        device,
    )
    try:
        for i, (input_path, output_path) in enumerate(jobs, start=1):
            logger.info("[%d/%d] Processing %s", i, len(jobs), input_path.name)
            if progress_callback is not None:
                progress_callback(i, input_path, output_path)
            img = image_io.read_image_rgb_chw(input_path)
            with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                out = restore_image_video_model(
                    img, detector, restorer,
                    device=device,
                    clip_size=clip_size,
                    denoise_strength=denoise_strength,
                    secondary_restorer=secondary_restorer,
                )
            image_io.write_image_rgb_chw(output_path, out)
            logger.info("Wrote %s", output_path)
    finally:
        detector.close()
        restorer.close()
        if secondary_restorer is not None and hasattr(secondary_restorer, "close"):
            secondary_restorer.close()
