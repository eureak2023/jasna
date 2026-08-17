# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir bundle of jasna_sidecar.py (jasna raw-frame restore sidecar for
# ffplay-fsr1), so -jasna works without a Python venv. Build:
#   .venv/Scripts/pyinstaller.exe --noconfirm jasna_sidecar.spec
from PyInstaller.utils.hooks import collect_submodules, collect_all

datas = []
binaries = []
hiddenimports = []

# jasna resolves its precompiled CUDA kernels next to sys.executable when frozen
# (jasna/media/cuda_kernel.py::fatbin_path), so they must sit at the bundle ROOT, not just
# under _internal/jasna/media where collect_all('jasna') would place them. PyInstaller 6
# puts every `datas` entry under _internal regardless of the dest we ask for, so the root
# copies are made after COLLECT at the bottom of this file. 0.10.0 grew from one kernel to
# six (cas/denoise/lut/resize_normalize/rgb_to_yuv/yuv_to_rgb); ship them all - the sidecar
# only exercises a few, but they are ~0.2-1.5 MB each and a missing one is a hard failure.
import glob as _glob
_FATBINS = sorted(_glob.glob(r'jasna\media\*.fatbin'))
datas += [(_f, '.') for _f in _FATBINS]

# The jasna CLI (offline --output, --stream) shells out to ffmpeg/ffprobe and requires
# major version 8. Ship BtbN's ffmpeg 8.1 next to the exe under tools\ so the frozen CLI
# finds a v8 ffprobe regardless of the host PATH (jasna_sidecar.py prepends it when frozen).
import os as _os
_ff = r'D:\Source_AI\ffplay-fsr1\pyav_build\ffmpeg8\bin'
for _exe in ('ffmpeg.exe', 'ffprobe.exe'):
    _p = _os.path.join(_ff, _exe)
    if _os.path.isfile(_p):
        datas += [(_p, 'tools')]

hiddenimports += collect_submodules('mmengine')
hiddenimports += collect_submodules('jasna')

# customtkinter/tkinterdnd2 let the same bundle also run the jasna CLI/GUI (jasna.main),
# so one bundle serves both the ffplay sidecar and standalone jasna (no separate jasna_cli).
for pkg in ('torch', 'torchvision', 'ultralytics', 'cv2', 'av', 'jasna',
            'tensorrt', 'tensorrt_libs', 'tensorrt_bindings', 'torch_tensorrt',
            'onnxruntime', 'customtkinter', 'tkinterdnd2'):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

a = Analysis(
    [r'D:\Source_AI\ffplay-fsr1\jasna\jasna_sidecar.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='jasna_sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='jasna_sidecar',
)

# Root copies of the CUDA kernels (see the _FATBINS note above): fatbin_path() looks beside
# sys.executable when frozen, and COLLECT has just finished writing the bundle, so do it here
# rather than leaving it as a manual post-build step that gets forgotten on the next rebuild.
import shutil as _shutil
_root = _os.path.join(DISTPATH, 'jasna_sidecar')
for _f in _FATBINS:
    _shutil.copy2(_f, _os.path.join(_root, _os.path.basename(_f)))
print(f"spec: copied {len(_FATBINS)} fatbin(s) to the bundle root {_root}")
