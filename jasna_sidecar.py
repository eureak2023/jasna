"""jasna_sidecar — restore mosaics with the jasna engine, speaking ffplay-fsr1's lada
sidecar protocol (raw RGB24 frames over stdout). Unlike jasna_bridge.py (which drove
jasna's HLS server), this runs jasna's restoration PIPELINE in-process from source and
emits the restored frames directly — no H.264 re-encode, no HLS, full engine speed.

Wire format (identical to lada_sidecar.py):
  HEADER = struct("<4sIiidi") = magic 'LADA' | gen u32 | w i32 | h i32 | pts_sec f64 | len i32
  then len bytes RGB24 (stride w*3). w<0 => READY marker. len==0 => EOF for that gen.
stdin commands: OPEN\t<target>\t<gen>\t<path> | SEEK\t<target>\t<gen> | STOP | QUIT

Runs in jasna's from-source venv (see docs/en/development.md). HW (NVDEC) decode needs a
custom PyAV (current_ctx API) not on PyPI, so we force SOFTWARE decode (decode is not the
bottleneck: ~769 fps at 1080p) via a NvidiaVideoReader monkeypatch — stock PyAV 18 works.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Queue, Empty

HEADER = struct.Struct("<4sIiidi")
MAGIC = b"LADA"

# Our binary frame stream needs a private, uncorrupted stdout. TensorRT / torch / CUDA write
# C-level log lines to fd 1 while loading engines, which would interleave into and corrupt
# it. So (sidecar mode ONLY) _init_frame_io() dups the real stdout to a private fd for our
# frames and points fd 1 at stderr, sending all that chatter to stderr. The jasna-CLI
# passthrough (which needs a normal stdout) does NOT call it, so its stdout is untouched.
_rawfd = None
_out_lock = threading.Lock()
_logf = None


def _init_frame_io():
    global _rawfd
    _rawfd = os.dup(1)
    try:
        os.dup2(2, 1)
    except Exception:
        pass
    try:
        import msvcrt
        msvcrt.setmode(_rawfd, os.O_BINARY)
    except Exception:
        pass


def _write_all(data):
    view = memoryview(data)
    while view:
        n = os.write(_rawfd, view)
        view = view[n:]


def log(m):
    print(f"[jasna-sidecar] {m}", file=sys.stderr, flush=True)
    if _logf:
        try:
            _logf.write(f"[jasna-sidecar] {m}\n"); _logf.flush()
        except Exception:
            pass


def emit(gen, w, h, pts_sec, payload):
    hdr = HEADER.pack(MAGIC, gen & 0xFFFFFFFF, w, h, pts_sec, len(payload))
    with _out_lock:
        _write_all(hdr if not payload else hdr + payload)


# Running from source, this script sits at the repo root next to the jasna/ checkout dir,
# whose bare 'jasna' folder would shadow the installed jasna package as an (empty) namespace
# package and hide jasna.__version__. Drop the repo root from sys.path so the real editable-
# installed package resolves. (No-op when frozen: the bundle has no such dir on sys.path.)
if not getattr(sys, "frozen", False):
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here in sys.path and os.path.isdir(os.path.join(_here, "jasna")):
        sys.path.remove(_here)

def _pin_bundled_cudnn():
    """Make torch's own cuDNN win over any CUDA Toolkit on PATH. MUST run before `import torch`.

    cudnn64_9.dll does not link its sublibraries (cudnn_graph/ops/cnn/adv/heuristic/
    engines_*64_9.dll) - it LoadLibrary()s them lazily, by BARE NAME. A bare-name load
    searches the exe dir, system32 and PATH, but NOT torch/lib: torch registers that dir
    with os.add_dll_directory(), which only applies to LoadLibraryEx with the USER_DIRS
    flag, so it is invisible here. Any CUDA Toolkit on PATH therefore hijacks the load
    (seen here: v12.8 ships cuDNN 9.8, v13.3 ships 9.24, our torch bundles 9.20), and the
    first restore that needs one of those sublibraries dies with
    "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH". The sidecar itself survives (the failure is
    caught per-pass), so it just stops emitting frames - which used to strand the player in
    a permanent "BUFFER" pause.

    Fix: put torch/lib FIRST on PATH and drop every other directory that carries a cuDNN,
    so the bare-name search can only find the matching set."""
    import importlib.util
    libdir = ""
    try:
        spec = importlib.util.find_spec("torch")   # does not execute torch/__init__.py
        libdir = os.path.join(spec.submodule_search_locations[0], "lib")
    except Exception:
        pass
    if not os.path.isfile(os.path.join(libdir, "cudnn64_9.dll")):
        # PyInstaller may hand back a frozen-importer path; torch/lib is under _MEIPASS.
        libdir = os.path.join(getattr(sys, "_MEIPASS", ""), "torch", "lib")
    if not os.path.isfile(os.path.join(libdir, "cudnn64_9.dll")):
        return                                     # CPU-only torch: nothing to pin
    try:
        os.add_dll_directory(libdir)
    except Exception:
        pass
    keep, dropped = [], []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            foreign = (not os.path.samefile(entry, libdir)) and \
                      os.path.isfile(os.path.join(entry, "cudnn64_9.dll"))
        except OSError:                            # unreadable / bogus PATH entry
            foreign = False
        (dropped if foreign else keep).append(entry)
    os.environ["PATH"] = os.pathsep.join([libdir] + keep)
    for d in dropped:
        log(f"ignoring foreign cuDNN on PATH: {d}")


_pin_bundled_cudnn()

# --- jasna imports (resolved from the from-source venv) ----------------------
import torch  # noqa: E402
import av  # noqa: E402
from av.video.reformatter import ColorRange as AvColorRange  # noqa: E402

from jasna.media import get_video_meta_data  # noqa: E402
from jasna.media.video_decoder import NvidiaVideoReader  # noqa: E402
from jasna.blend_buffer import BlendBuffer  # noqa: E402
from jasna.crop_buffer import CropBuffer  # noqa: E402
from jasna.frame_queue import FrameQueue  # noqa: E402
from jasna.pipeline_items import FrameMeta  # noqa: E402
from jasna.vram_offloader import VramOffloader  # noqa: E402
from jasna.pipeline_threads import (  # noqa: E402
    decode_detect_loop, primary_restore_loop, secondary_restore_loop, blend_encode_loop,
)


# --- force software decode (avoids the custom-PyAV current_ctx requirement) ---
def _sw_enter(self):
    self._decoder_ctx = None
    self._amd_hardware_decode = False
    self.container = av.open(self.file)               # stock PyAV, software decode
    self.video_stream = self.container.streams.video[0]
    ctx = self.video_stream.codec_context
    ctx.thread_type = "AUTO"
    self.width = ctx.width
    self.height = ctx.height
    self._full_range = (
        ctx.color_range == int(AvColorRange.JPEG)
        or self.metadata.color_range == AvColorRange.JPEG
    )
    self._raw_stream = None
    return self


# NOTE: the _sw_enter monkeypatch is applied only in run_sidecar() (sidecar mode), so the
# jasna-CLI passthrough keeps jasna's normal hardware-decode path.


# --- fix intermittent frame corruption: force the SW decoders onto the default stream -------
# Intermittent full-frame garbage / colored-dots-on-black (~1% of frames, in bursts of
# consecutive frames) was traced to PyTorch's caching allocator handing a decoder batch's
# still-in-use VRAM to another reader's in-flight CUDA work. It is unique to our sidecar:
# _sw_enter forces BOTH the detection AND blend readers onto the software->CUDA upload path
# (jasna normally decodes on NVDEC), so two NvidiaVideoReader instances run `_frames_software`
# concurrently, each creating its OWN CUDA stream (jasna.media.video_decoder.new_stream) to
# convert YUV->RGB while the batch is allocated/consumed on the default stream. The allocator
# has no cross-stream reuse guard here (no record_stream), so a freed batch's memory can be
# handed to the other reader's stream mid-conversion -> a burst of garbage frames. Proven:
# 0.96% corruption with the caching allocator vs 0.00% with PYTORCH_NO_CUDA_MEMORY_CACHING=1
# (but disabling caching costs ~37% throughput; record_stream on the yielded batch barely
# helped). The clean fix: make new_stream() return the *default* stream so both readers convert
# on it -- then allocation, conversion and consumption are all one stream and the allocator's
# normal same-stream ordering makes reuse safe. new_stream is only used by the two decoders in
# sidecar mode (the encoder is replaced by RawFrameWriter), and YUV conversion is cheap
# (~769fps) so serializing it onto the default stream costs almost nothing. Patch the name as
# bound inside video_decoder (it did `from jasna.accelerator import new_stream`).
import jasna.media.video_decoder as _vd  # noqa: E402


def _default_stream(device):
    return torch.cuda.current_stream(torch.device(device))


class RawFrameWriter:
    """Sink the pipeline calls per restored frame: frame is a (3,H,W) uint8 CUDA RGB
    tensor; we ship it as packed RGB24 with its absolute pts."""
    def __init__(self, gen, time_base):
        self.gen = gen
        self.tb = float(time_base)
        self.n = 0
        self.t0 = time.time()

    def write(self, frame, pts, *, apply_lut=True):
        # `frame` is a (3,H,W) CUDA tensor from the restore/blend pipeline. With the SW
        # decoders pinned to the default stream (see the new_stream patch below) the blend and
        # this device->host copy are ordered on that one stream, so the .to("cpu") sees a fully
        # written frame -- no extra fence needed (an earlier full-device sync here was a
        # redundant leftover from before the stream fix and only cost throughput).
        arr = frame.permute(1, 2, 0).contiguous().to("cpu", torch.uint8).numpy()
        h, w = arr.shape[0], arr.shape[1]
        emit(self.gen, w, h, float(pts) * self.tb, arr.tobytes())
        self.n += 1
        if self.n == 1:
            log(f"gen={self.gen} first frame pts={float(pts)*self.tb:.2f} {w}x{h} "
                f"({time.time()-self.t0:.1f}s)")

    def after_write(self, n):
        pass

    def close(self):
        pass


def build_pipeline(device_str, model_weights, max_clip_size, batch_size, temporal_overlap, fp16):
    from jasna.accelerator import device_context
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.pipeline import Pipeline

    device = torch.device(device_str)
    mw = Path(model_weights)
    detection_model_name = "rfdetr-v5"
    detection_model_path = mw / "rfdetr-v5.onnx"
    restoration_model_path = mw / "lada_mosaic_restoration_model_generic_v1.2.pth"

    log("compiling/loading engines (reuses cached engines in model_weights) ...")
    cr = ensure_engines_compiled(EngineCompilationRequest(
        device=str(device), fp16=fp16, basicvsrpp=True,
        basicvsrpp_model_path=str(restoration_model_path),
        basicvsrpp_max_clip_size=max_clip_size,
        detection=True, detection_model_name=detection_model_name,
        detection_model_path=str(detection_model_path),
        detection_batch_size=batch_size, unet4x=False,
    ))
    with device_context(device):
        restoration_pipeline = RestorationPipeline(
            restorer=BasicvsrppMosaicRestorer(
                checkpoint_path=str(restoration_model_path), device=device,
                max_clip_size=max_clip_size, use_tensorrt=cr.use_basicvsrpp_tensorrt, fp16=fp16),
            secondary_restorer=None,
            denoise_strength=DenoiseStrength("none"), denoise_step=DenoiseStep("after_primary"),
        )
        pipeline = Pipeline(
            input_video=Path("__none__"), output_video=Path("__none___out"),
            detection_model_name=detection_model_name, detection_model_path=detection_model_path,
            detection_score_threshold=0.25, restoration_pipeline=restoration_pipeline,
            codec="h264", encoder_settings={}, batch_size=batch_size, device=device,
            max_clip_size=max_clip_size, temporal_overlap=temporal_overlap, enable_crossfade=True,
            vr_mode="off", fp16=fp16, disable_progress=True, lut_path=None,
            retarget_high_fps=False, segments=None, splice_plan=None, working_dir=None,
        )
    log("models loaded")
    return pipeline, device


def run_pass(pipeline, device, metadata, gen, seek_ts, cancel_event):
    """One restoration pass from seek_ts to EOF (or until cancel_event), emitting frames
    tagged `gen`. Mirrors streaming_pipeline._run_streaming_pass minus the HLS server."""
    secondary_workers = max(1, int(pipeline.restoration_pipeline.secondary_num_workers))
    clip_queue = FrameQueue(max_frames=pipeline.max_clip_size)
    secondary_queue = FrameQueue(max_frames=pipeline.max_clip_size * secondary_workers)
    encode_queue = FrameQueue(max_frames=pipeline.max_clip_size)
    metadata_queue: "Queue" = Queue(maxsize=pipeline.max_clip_size * 5)

    error_holder = []
    blend_buffer = BlendBuffer(device=device)
    crop_buffers: dict = {}
    crop_lock = threading.Lock()
    primary_idle_event = threading.Event()
    frame_shape = []

    vram_offloader = VramOffloader(
        device=device, blend_buffer=blend_buffer, crop_buffers=crop_buffers, crop_lock=crop_lock)
    vram_offloader.set_pipeline_queues(clip_queue, secondary_queue, encode_queue, metadata_queue)

    frame_writer = RawFrameWriter(gen, metadata.time_base)
    st = seek_ts if seek_ts and seek_ts > 0 else None

    threads = [
        threading.Thread(target=lambda: decode_detect_loop(
            input_video=str(pipeline.input_video), batch_size=pipeline.batch_size, device=device,
            metadata=metadata, detection_model=pipeline._job_detection_model,
            max_clip_size=pipeline.max_clip_size, temporal_overlap=pipeline.temporal_overlap,
            enable_crossfade=pipeline.enable_crossfade, blend_buffer=blend_buffer,
            crop_buffers=crop_buffers, clip_queue=clip_queue, metadata_queue=metadata_queue,
            error_holder=error_holder, frame_shape=frame_shape, cancel_event=cancel_event,
            seek_ts=st, vr_mode=pipeline._vr_resolution.resolved, vr_projector=pipeline._vr_projector,
        ), name="DecodeDetect", daemon=True),
        threading.Thread(target=lambda: primary_restore_loop(
            device=device, restoration_pipeline=pipeline.restoration_pipeline,
            clip_queue=clip_queue, secondary_queue=secondary_queue, error_holder=error_holder,
            primary_idle_event=primary_idle_event, cancel_event=cancel_event,
        ), name="PrimaryRestore", daemon=True),
        threading.Thread(target=lambda: secondary_restore_loop(
            device=device, restoration_pipeline=pipeline.restoration_pipeline,
            secondary_queue=secondary_queue, encode_queue=encode_queue, error_holder=error_holder,
            cancel_event=cancel_event,
        ), name="SecondaryRestore", daemon=True),
        threading.Thread(target=lambda: blend_encode_loop(
            input_video=str(pipeline.input_video), batch_size=pipeline.batch_size, device=device,
            metadata=metadata, blend_buffer=blend_buffer, encode_queue=encode_queue,
            metadata_queue=metadata_queue, error_holder=error_holder, frame_writer=frame_writer,
            cancel_event=cancel_event, seek_ts=st, vram_offloader=vram_offloader,
            vr_projector=pipeline._vr_projector,
        ), name="BlendEncode", daemon=True),
    ]
    vram_offloader.start()
    for t in threads:
        t.start()

    all_queues = [clip_queue, secondary_queue, encode_queue, metadata_queue]

    def _drain():
        for q in all_queues:
            try:
                while True:
                    q.get_nowait()
            except Empty:
                pass

    for t in threads:
        while t.is_alive():
            if cancel_event.is_set():
                _drain()
            t.join(timeout=0.05)
    vram_offloader.stop()

    if not cancel_event.is_set():
        emit(gen, 0, 0, 0.0, b"")
        log(f"gen={gen} pass done ({frame_writer.n} frames)")
    if error_holder and not cancel_event.is_set():
        log(f"gen={gen} pass error: {error_holder[0]!r}")


class Sidecar:
    def __init__(self, model_weights, device, max_clip_size, batch_size, temporal_overlap, fp16):
        self.pipeline, self.device = build_pipeline(
            device, model_weights, max_clip_size, batch_size, temporal_overlap, fp16)
        self._path = None
        self._meta = None
        self._cancel = None
        self._worker = None

    def _stop_pass(self):
        if self._cancel:
            self._cancel.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=15)
        self._worker = None

    def _start_pass(self, target, gen, path):
        self._stop_pass()
        if path != self._path:
            self.pipeline.input_video = Path(path)
            self._meta = get_video_meta_data(path)
            self.pipeline.configure_vr(self._meta)
            self._path = path
        self._cancel = threading.Event()
        cancel = self._cancel
        self._worker = threading.Thread(
            target=lambda: run_pass(self.pipeline, self.device, self._meta, gen, target, cancel),
            daemon=True)
        self._worker.start()

    def run(self):
        emit(0, -1, -1, 0.0, b"")   # READY
        log("READY")
        stdin = sys.stdin.buffer
        while True:
            raw = stdin.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            cmd = parts[0]
            try:
                if cmd == "OPEN" and len(parts) >= 4:
                    target, gen, path = float(parts[1]), int(parts[2]), parts[3]
                    log(f"OPEN target={target:.3f} gen={gen} {os.path.basename(path)}")
                    self._start_pass(target, gen, path)
                elif cmd == "SEEK" and len(parts) >= 3:
                    target, gen = float(parts[1]), int(parts[2])
                    if self._path:
                        log(f"SEEK target={target:.3f} gen={gen}")
                        self._start_pass(target, gen, self._path)
                elif cmd == "STOP":
                    log("STOP")
                    self._stop_pass()
                elif cmd == "QUIT":
                    break
                else:
                    log(f"unknown command: {line!r}")
            except Exception as e:
                log(f"command error ({line!r}): {e}\n{traceback.format_exc()}")
        self._stop_pass()
        log("exiting")


def run_sidecar():
    global _logf
    # Optional diagnostics: set JASNA_SIDECAR_LOGLEVEL=INFO (or DEBUG) to surface jasna's own
    # logging on stderr -- in particular the per-thread LoopTimer summaries logged at pass end
    # ([timing] decode/detect/restore/blend/write/queue-wait), used to profile the bottleneck.
    _lvl = os.environ.get("JASNA_SIDECAR_LOGLEVEL", "").upper()
    if _lvl:
        import logging as _logging
        _logging.basicConfig(level=getattr(_logging, _lvl, _logging.WARNING),
                             stream=sys.stderr, format="%(name)s %(levelname)s: %(message)s")
    _init_frame_io()                              # private frame stdout (see above)
    # Decode backend. Default "hw": jasna's native NVDEC path (needs the custom PyAV's
    # current_ctx). Measured +30% throughput at 1080p (50->65 fps) AND zero corruption --
    # NVDEC uses independent decoders so the dual-SW-reader allocator aliasing simply does not
    # arise. jasna auto-falls-back to software per file for codecs NVDEC can't handle. We keep
    # the new_stream->default patch applied ALWAYS: it only affects the SW conversion path
    # (_frames_software), so it is a no-op under NVDEC but still guards any per-file SW
    # fallback against the allocator aliasing. "sw" (JASNA_SIDECAR_DECODE=sw) forces the old
    # stock-PyAV software path for troubleshooting.
    _vd.new_stream = _default_stream               # SW-decode conversions share the default stream (see above)
    if os.environ.get("JASNA_SIDECAR_DECODE", "hw").lower() == "sw":
        NvidiaVideoReader.__enter__ = _sw_enter    # force software decode (stock PyAV)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-weights", required=True, help="dir with the jasna model weights + engines")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-clip-size", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--temporal-overlap", type=int, default=8)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    if args.log:
        try:
            _logf = open(args.log, "a", encoding="utf-8")
        except Exception:
            _logf = None
    try:
        sc = Sidecar(args.model_weights, args.device, args.max_clip_size,
                     args.batch_size, args.temporal_overlap, not args.no_fp16)
        sc.run()
    except Exception:
        log("FATAL:\n" + traceback.format_exc())
        raise


def main():
    # One binary, two roles: with --model-weights we are ffplay's raw-frame restore sidecar;
    # otherwise we hand off to the normal jasna CLI/GUI (jasna.main), so a single bundle
    # serves both the player sidecar AND standalone jasna (offline / --stream / GUI).
    if "--model-weights" in sys.argv[1:]:
        run_sidecar()
    else:
        # The jasna CLI shells out to ffmpeg/ffprobe (must be v8). The bundle ships v8 in
        # tools\ next to the exe; prepend it so the CLI never picks up a wrong-version
        # ffprobe from the host PATH.
        if getattr(sys, "frozen", False):
            # datas land under _internal (sys._MEIPASS) in a onedir build.
            for _base in (getattr(sys, "_MEIPASS", None), os.path.dirname(sys.executable)):
                if not _base:
                    continue
                _tools = os.path.join(_base, "tools")
                if os.path.isdir(_tools):
                    os.environ["PATH"] = _tools + os.pathsep + os.environ.get("PATH", "")
                    break
        # The editable install makes `jasna` resolve as a namespace package once frozen, so
        # its __init__ (which defines __version__) never runs and `from jasna import
        # __version__` in jasna.main fails. Submodules still import fine; only this one
        # __init__ attribute needs a fallback. (Keep in sync with jasna/__init__.py.)
        import jasna
        if not getattr(jasna, "__version__", None):
            jasna.__version__ = "0.8.1"
        from jasna.main import main as jasna_main
        jasna_main()


if __name__ == "__main__":
    main()
