"""DLSS 5 neural rendering (NVIDIA NGX "dlssnr") as a finishing pass.

The model resynthesises fine detail a low-bitrate encode threw away - it puts
texture back rather than sharpening what survived. It runs on the *finished*
frame, after detection and restoration, so nothing it does can change what the
detector finds or what BasicVSR++ rebuilds; the restoration models see the
stream exactly as they did before.

None of this is reachable from Python directly. The snippet is undocumented,
has no entry in the driver's own NGX dispatcher, only works through D3D12, and
refuses callers whose module path does not contain "nvngx.dll". All of that
lives in the two DLLs built by ``native/dlssnr/build.sh``; this module is only
the ctypes seam. See ``native/dlssnr/jasna_ngx.h`` for the C API.

Everything degrades quietly: no RTX 50 series GPU, no snippet DLL, no DLLs
built - :func:`load` returns None and the caller passes frames through.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_STRENGTH = 60      # percent, matching ffplay's default
MAX_STRENGTH = 150


class _Tune(ctypes.Structure):
    """Mirror of FFNRTune in native/dlssnr/jasna_ngx.h."""

    _fields_ = [
        ("intensity", ctypes.c_float),
        ("local_structure", ctypes.c_float),
        ("local_tone", ctypes.c_float),
        ("skin_structure", ctypes.c_float),
        ("style", ctypes.c_int),
        ("auto_mask", ctypes.c_int),
        ("reset", ctypes.c_int),
    ]


def _candidate_dirs() -> list[Path]:
    """Where jasna_dlssnr.dll might be, frozen or not."""
    here = Path(__file__).resolve()
    dirs = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        dirs += [exe_dir, exe_dir / "_internal", Path(getattr(sys, "_MEIPASS", exe_dir))]
    dirs += [
        here.parent.parent / "native" / "dlssnr",   # source tree
        here.parent.parent,
    ]
    return dirs


class DlssNr:
    """One model instance. Not thread-safe: a single worker must own it."""

    def __init__(self, lib: ctypes.CDLL, strength: int, colour: float = 0.0):
        self._lib = lib
        self._w = self._h = 0
        self._out: np.ndarray | None = None
        self.strength = strength
        self.colour = float(colour)
        self._prev_src: torch.Tensor | None = None
        self._prev_out: torch.Tensor | None = None
        self._held = 0
        # The model's own defaults, exactly as ffplay uses them. skin_structure
        # -1 means "follow the local structure term", which is the model's
        # default and is not the same as a strength of 0.
        self._tune = _Tune(1.0, 1.0, 1.0, -1.0, 2, 1, 1)

    @property
    def _error(self) -> str:
        return self._lib.jnr_last_error().decode("ascii", "replace")

    def reset_history(self) -> None:
        """Drop temporal state. A seek or a cut must call this."""
        self._tune.reset = 1

    def process(self, frame: torch.Tensor) -> torch.Tensor:
        """Enhance one uint8 RGB frame, returning the same layout and device.

        Accepts (3, H, W) or (H, W, 3). Any failure returns the input
        unchanged - a frame that cannot be enhanced still has to be shown.
        """
        if frame.dtype != torch.uint8 or frame.ndim != 3:
            return frame
        chw = frame.shape[0] == 3 and frame.shape[2] != 3
        hwc = frame.permute(1, 2, 0) if chw else frame
        h, w = int(hwc.shape[0]), int(hwc.shape[1])

        if (w, h) != (self._w, self._h):
            if self._lib.jnr_ensure(w, h) != 0:
                log.warning("DLSS-NR: disabled at %dx%d: %s", w, h, self._error)
                return frame
            self._w, self._h = w, h
            self._out = np.empty((h, w, 3), dtype=np.uint8)
            self._tune.reset = 1
            self._prev_src = self._prev_out = None
            log.info("DLSS-NR: model up at %dx%d", w, h)

        # Hold the previous answer while the picture is not moving.
        #
        # The model is recurrent and does not settle: feeding it one frame
        # twelve times over produces twelve different pictures, drifting by
        # ~0.6 mean levels and up to 18 in places across 2-3% of the frame,
        # with no sign of converging. That is invisible while the scene moves
        # and very visible when it does not - a still shot, a held pose, or the
        # duplicated frames telecined content is full of - because everything
        # around the shimmer is frozen. Reusing the last answer costs nothing
        # (it also skips the model) and leaves a still picture actually still.
        if self._prev_src is not None and self._prev_out is not None:
            if self._frame_delta(hwc, self._prev_src) < self._STATIC_DELTA:
                self._held += 1
                out = self._prev_out
                return out.permute(2, 0, 1).contiguous() if chw else out

        src = np.ascontiguousarray(hwc.cpu().numpy())
        rc = self._lib.jnr_process(
            src.ctypes.data_as(ctypes.c_char_p),
            self._out.ctypes.data_as(ctypes.c_char_p),
            w, h, ctypes.byref(self._tune),
        )
        self._tune.reset = 0
        if rc != 0:
            log.warning("DLSS-NR: %s; passing frames through", self._error)
            self._w = self._h = 0
            return frame

        model = torch.from_numpy(self._out).to(frame.device, non_blocking=True)
        out = self._compose(hwc, model)
        self._prev_src = hwc.clone()
        self._prev_out = out
        return out.permute(2, 0, 1).contiguous() if chw else out

    # Mean absolute difference, in 0-255 levels, below which two frames count
    # as the same picture. Moving content on a 1080p encode sits around 1.0
    # here; a duplicated frame around 0.03. 0.3 separates them with room to
    # spare, and erring low only means the model runs on a frame it need not
    # have - which is what it did before this existed.
    _STATIC_DELTA = 0.3

    @staticmethod
    def _frame_delta(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.shape != b.shape:
            return float("inf")
        return float((a.float() - b.float()).abs_().mean())

    # Ported verbatim from ffplay's nr_compose_src / nr_stat_src fragment
    # shaders (ffplay_fsr.c). Skipping it is what makes the picture flicker:
    # the model returns a whole picture rather than a correction and its
    # absolute luminance is arbitrary - about 1.05x low on a bright scene and
    # 1.40x low on a dark one - so straight blending pumps the brightness up
    # and down from frame to frame as the scene changes.
    #
    # So: scale the model's answer by the ratio of the two frame-mean lumas,
    # then read it as a per-pixel *brightness verdict* on the original rather
    # than as pixels. At colour 0 only that verdict is taken and the source hue
    # survives untouched, which is what keeps the pass from tinting the film;
    # at 1 the model's own colour comes through too. max_ratio caps brightening
    # so a highlight cannot turn into a cluster of coloured cells.
    _LUMA = (0.2126, 0.7152, 0.0722)
    _MAX_RATIO = 2.0

    def _compose(self, src_u8: torch.Tensor, model_u8: torch.Tensor) -> torch.Tensor:
        luma = torch.tensor(self._LUMA, device=src_u8.device, dtype=torch.float32)
        o = src_u8.float().mul_(1.0 / 255.0)
        m = model_u8.float().mul_(1.0 / 255.0)

        lo = o @ luma                       # per-pixel luma of the original
        lm = m @ luma                       # ... and of the model's answer
        mean_o, mean_m = lo.mean(), lm.mean()
        gain = (mean_o / mean_m).clamp(0.25, 4.0) if mean_m > 1e-5 else \
            torch.ones((), device=o.device)
        m = m * gain
        lm = lm * gain

        ratio = torch.where(lo > 1.0 / 255.0, lm / lo.clamp_min(1e-6),
                            torch.ones_like(lo)).clamp_(0.0, self._MAX_RATIO)
        blend = o * ratio.unsqueeze(-1)
        if self.colour > 0.0:
            blend = blend.lerp_(m, self.colour)
        out = torch.lerp(o, blend, self.strength / 100.0).clamp_(0.0, 1.0)
        return out.mul_(255.0).round_().to(torch.uint8)

    def close(self) -> None:
        self._lib.jnr_shutdown()


def load(strength: int = DEFAULT_STRENGTH, colour: float = 0.0,
         snippet_path: str | None = None) -> DlssNr | None:
    """Bring the model up, or return None with the reason logged.

    Returning None is an ordinary outcome, not an error: the sidecar has to
    keep working on machines without an RTX 50 series GPU.
    """
    strength = max(0, min(MAX_STRENGTH, int(strength)))
    if strength == 0:
        return None
    if snippet_path:
        os.environ["JASNA_DLSSNR_DLL"] = str(snippet_path)

    lib = None
    for d in _candidate_dirs():
        dll = d / "jasna_dlssnr.dll"
        if not dll.exists():
            continue
        try:
            # So the loader finds nvngx.dll_jasna.dll beside it.
            os.add_dll_directory(str(d))
        except (OSError, AttributeError):
            pass
        try:
            lib = ctypes.CDLL(str(dll))
            break
        except OSError as exc:
            log.warning("DLSS-NR: cannot load %s: %s", dll, exc)
    if lib is None:
        log.info("DLSS-NR: jasna_dlssnr.dll not found; neural rendering off")
        return None

    lib.jnr_last_error.restype = ctypes.c_char_p
    lib.jnr_process.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int,
                                ctypes.POINTER(_Tune)]
    if lib.jnr_init() != 0:
        log.info("DLSS-NR: unavailable (%s); neural rendering off",
                 lib.jnr_last_error().decode("ascii", "replace"))
        return None

    colour = max(0.0, min(1.0, float(colour)))
    print(f"DLSS neural rendering: on at {strength}%"
          + (f", colour {colour:g}" if colour else ""))
    return DlssNr(lib, strength, colour)
