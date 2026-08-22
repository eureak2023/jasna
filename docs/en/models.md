# Choosing Models

## Detection model

The detection model finds mosaics in each frame.

- **Use the latest RF-DETR model** (`rfdetr-v6`) — it's the default, fast, and
  the best all-rounder. Bundled with Jasna.
- **`rfdetr-v6-large`** is a higher-quality, slower variant, and can be the
  better pick for 4K videos. It is an optional
  separate download — pick the file for your graphics card, drop it into
  `model_weights/`, and Jasna detects it automatically:
  - NVIDIA:
    [rfdetr-v6-large.onnx](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.onnx)
  - AMD:
    [rfdetr-v6-large.pt](https://github.com/Kruk2/jasna/releases/download/0.1/rfdetr-v6-large.pt)
- **Lada YOLO** models can work better for 2D animations.
- **rfdetr-vr-v1** (bundled) is the RF-DETR VR180 detection model — best for VR180 videos.
- **zelefans-vr-yolo-v2** (optional download) is an alternative VR180 detector.
- **On AMD**, RF-DETR runs on your graphics card and uses the `.pt` model
  files (NVIDIA uses `.onnx`). It is slower than on NVIDIA, so pick
  `lada-yolo-v4` when speed matters more than detection quality.

Each model applies its own recommended detection threshold by default
(`rfdetr-v6`: 0.35, `rfdetr-v6-large`: 0.40); override with
`--detection-score-threshold`.

The legacy `rfdetr-v5` model remains supported.

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v6
```

You can also set a different detection model per video inside the
[segment editor](segments.md).

## Secondary restoration

Jasna restores a 256x256 crop of each mosaic region. Large mosaic regions,
close-ups, and 4K videos can therefore look blurry after the primary
restoration. A secondary model upscales the restored crop to 512x512 or
1024x1024 before blending it back, making it noticeably sharper.

- **unet-4x**: supporter model. Faster than TVAI with similar quality in
  current testing. Trained on an in-domain JAV dataset and visually close to
  TVAI `iris-2`. See
  [examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260).
  Unlock it with a supporter key — see
  [Supporting the project](../../README.md#supporting-the-project).
- **RTX Super Resolution**: very fast, free, and needs nothing extra.
  Quality is okay. Some videos may flicker, so test on a short clip first.
- **TVAI**: better than RTX Super Resolution and comparable to unet-4x, but
  very slow. Requires [Topaz Video](https://www.topazlabs.com/topaz-video),
  which is paid and Windows-only. Recommended model: `iris-2`.

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

For TVAI, set the `TVAI_MODEL_DATA_DIR` and `TVAI_MODEL_DIR` environment
variables to your Topaz Video model folders, as shown below
(`--tvai-args` can further customize the Topaz model parameters):

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

### Speed and VRAM comparison

| Secondary type           | CAWD 1080p        | KV-109 1080p      |
| ------------------------ | -----------------:| -----------------:|
| No secondary             | 22s / 10.0 GB VRAM | 11s / 10.7 GB VRAM |
| unet-4x                  | 29s / 12.5 GB VRAM | 14s / 12.6 GB VRAM |
| RTX Super-Res            | 25s / 11.7 GB VRAM | 13s / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52s / 12.1 GB VRAM | 24s / 12.4 GB VRAM |

## Still-image restoration

For still images Jasna does not run the video pipeline. Just add an image to the
GUI queue (or pass it on the CLI) — image jobs route to the still-image path
automatically:

```bash
jasna --input photo.png --output restored.png
```

Two engines are available, picked with `--image-restoration-model-name`
(**Engine** in the GUI's Image Restoration section):

| Engine | Needs | Approach |
| ------ | ----- | -------- |
| `basicvsrpp` (default) | nothing extra | Runs the video restoration model on a static clip built from the image. |
| `sd-15-jav` | 6.9 GB download + supporter licence | Inpaints the mosaic with a fine-tuned diffusion model — invents plausible detail. |

### Video model (BasicVSR++)

```bash
jasna --input photo.png --output restored.png
```

Detection, crop geometry and blending are exactly the video pipeline's; only the
clip is synthetic — `--image-clip-size` copies of the image (default 5), of which
the middle restored frame is kept. It reuses the weights already in
`model_weights/`, so there is nothing to download and no licence to enter, and it
finishes in well under a second per image.

Because a still frame has no neighbouring frames to fuse, this is closer to a
learned de-mosaic of that one image than to what BasicVSR++ achieves on video.
In exchange it stays faithful to the source instead of generating new content.
Raising `--image-clip-size` gives the recurrent network a few more refinement
passes; the difference flattens out past about 5.

### SD 1.5

- The model is **not bundled** and is about **6.9 GB**. Jasna asks before
  downloading it from
  [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav).
- It is currently available only to supporters and uses the same key as
  unet-4x — see
  [Supporting the project](../../README.md#supporting-the-project).
- Expect about **7 GB VRAM** during inference, a bit more for large 4K
  images.

The SD 1.5 path is experimental. Results vary by scene, but some images can
work very well. Generate several variants and keep the best one:

```bash
jasna --input photo.png --output restored.png --sd15-variants 4
```

Every knob (`--sd15-steps`, `--sd15-strength`, `--sd15-seed`, ...) is listed
in the [CLI reference](cli.md#sd-15-sd-15-jav).

Examples:
[SD 1.5 examples on SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106)
and [more](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990).
