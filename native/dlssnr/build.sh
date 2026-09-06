#!/bin/sh
# Build the DLSS neural-rendering pair for the jasna sidecar (MSYS2 mingw64).
#
#   nvngx.dll_jasna.dll  - the NGX forwarder. Its name is load-bearing: the
#                          snippet refuses callers whose module path does not
#                          contain "nvngx.dll".
#   jasna_dlssnr.dll     - the D3D12 host that ctypes talks to.
#
# Both are plain Win32 DLLs with no runtime dependencies beyond the system, so
# they are copied into the PyInstaller bundle as data files.
#
# Run from an MSYS2 shell with mingw64 first on PATH. TMPDIR must be a *mixed*
# path (C:/msys64/tmp) that both MSYS tools and the native gcc accept: a plain
# /c/... one makes gcc fall back to C:\WINDOWS and fail with "Permission
# denied". Same gotcha as the ffplay build; see the repo CLAUDE.md.
set -e

cd "$(dirname "$0")"
OUT="${1:-.}"
CC="${CC:-gcc}"
CFLAGS="-O2 -Wall -municode -DUNICODE -D_UNICODE"

: "${TMPDIR:=C:/msys64/tmp}"
export TMPDIR TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p /c/msys64/tmp 2>/dev/null || true

mkdir -p "$OUT"

# -fno-optimize-sibling-calls: the snippet walks the return address back to the
# calling module, and a tail call would hand it the wrong one.
$CC $CFLAGS -fno-optimize-sibling-calls -shared -static -static-libgcc \
    -o "$OUT/nvngx.dll_jasna.dll" jasna_ngxshim.c -lkernel32

# -ldxguid carries the D3D12/DXGI interface GUIDs; libuuid does not have them.
$CC $CFLAGS -shared -static -static-libgcc \
    -o "$OUT/jasna_dlssnr.dll" jasna_dlssnr.c -lkernel32 -lole32 -ldxguid -luuid

echo "built:"
ls -l "$OUT/nvngx.dll_jasna.dll" "$OUT/jasna_dlssnr.dll"
