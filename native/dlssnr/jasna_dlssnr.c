/*
 * DLSS 5 Neural Rendering host for the jasna sidecar - D3D12 only.
 *
 * Ported from ffplay_ngx.c in this repository. The player already owns a D3D11
 * device full of decoded frames, so its version bridges D3D11 and D3D12 with
 * shared textures and a shared fence. jasna has CPU frames and nothing else,
 * so all of that disappears: an UPLOAD buffer feeds the model, a READBACK
 * buffer drains it, and the only fence is D3D12's own.
 *
 * The model is called exactly as the player calls it, through the same
 * forwarder, because the snippet checks its caller's module path and accepts
 * nothing but a module named for nvngx.dll (see jasna_ngxshim.c).
 *
 * Single-threaded by contract: one worker owns the device.
 */

#define COBJMACROS
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <stdio.h>
#include <string.h>

#define JNR_BUILD
#include "jasna_ngx.h"

#define NGX_OK 0x1u

/* Two allocators so the next frame can be recorded while the previous one is
 * still in flight. The readback makes every frame synchronous anyway, but the
 * cost is two objects and it keeps the door open. */
#define JNR_FRAMES 2

/* D3D12_TEXTURE_DATA_PITCH_ALIGNMENT; spelled out because some mingw headers
 * do not define it. */
#define JNR_PITCH_ALIGN 256

static struct {
    int state;                  /* 0 = untried, 1 = ready, -1 = unavailable */

    HMODULE shim;
    PFN_ffnr_load     f_load;
    PFN_ffnr_init     f_init;
    PFN_ffnr_create   f_create;
    PFN_ffnr_evaluate f_evaluate;
    PFN_ffnr_release  f_release;
    PFN_ffnr_shutdown f_shutdown;

    ID3D12Device              *dev;
    ID3D12CommandQueue        *queue;
    ID3D12CommandAllocator    *alloc[JNR_FRAMES];
    ID3D12GraphicsCommandList *list;
    UINT64                     alloc_val[JNR_FRAMES];
    int                        frame;

    ID3D12Fence *fence;
    HANDLE       fence_event;
    UINT64       fence_val;

    /* Per size. */
    ID3D12Resource *in_tex, *out_tex;
    ID3D12Resource *upload, *readback;
    UINT            row_pitch;          /* aligned, for both staging buffers */
    void           *feature;
    int             w, h;
} nr;

static char jnr_err[512] = "";

static int jnr_fail(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(jnr_err, sizeof(jnr_err), fmt, ap);
    va_end(ap);
    return -1;
}

JNR_API const char *jnr_last_error(void) { return jnr_err; }
JNR_API int jnr_available(void) { return nr.state == 1 && nr.feature != NULL; }

/* ---- locating our companions ----
 *
 * Both the forwarder and the 158 MB snippet blob sit on disk next to us rather
 * than being embedded: the sidecar is a PyInstaller bundle, so "next to the
 * exe" and "next to this DLL" are different directories and neither is fixed
 * at build time. Hence the search rather than a single path. */
static int jnr_self_dir(wchar_t *out, size_t out_len)
{
    HMODULE self = NULL;
    wchar_t *slash;

    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)(void *)jnr_self_dir, &self) || !self)
        return -1;
    if (!GetModuleFileNameW(self, out, (DWORD)out_len))
        return -1;
    slash = wcsrchr(out, L'\\');
    if (!slash)
        return -1;
    *slash = 0;
    return 0;
}

static int jnr_exists(const wchar_t *path)
{
    return GetFileAttributesW(path) != INVALID_FILE_ATTRIBUTES;
}

/* Look for name in: $JASNA_DLSSNR_DLL (exact file, snippet only), this DLL's
 * directory, its parent, its parent's "bin", and the same three relative to
 * the running exe. In this repo the sidecar lives in bin\jasna_sidecar\ and
 * nvngx_dlssnr.dll in bin\, so the parent hop is the one that finds it. */
static int jnr_find(const wchar_t *name, const wchar_t *env, wchar_t *out,
                    size_t out_len)
{
    wchar_t base[MAX_PATH], *slash;
    int origin;

    if (env) {
        DWORD n = GetEnvironmentVariableW(env, out, (DWORD)out_len);

        if (n && n < out_len)
            return jnr_exists(out) ? 0 : jnr_fail("%ls points at a missing file", env);
    }

    for (origin = 0; origin < 2; origin++) {
        if (origin == 0) {
            if (jnr_self_dir(base, MAX_PATH) < 0)
                continue;
        } else {
            if (!GetModuleFileNameW(NULL, base, MAX_PATH))
                continue;
            slash = wcsrchr(base, L'\\');
            if (!slash)
                continue;
            *slash = 0;
        }
        for (int up = 0; up < 3; up++) {
            static const wchar_t *suffix[3] = { L"", L"\\..", L"\\..\\bin" };

            _snwprintf(out, out_len, L"%ls%ls\\%ls", base, suffix[up], name);
            out[out_len - 1] = 0;
            if (jnr_exists(out))
                return 0;
        }
    }
    return jnr_fail("%ls not found beside the sidecar", name);
}

/* ---- fence ---- */

static void jnr_cpu_wait(UINT64 value)
{
    if (!value || ID3D12Fence_GetCompletedValue(nr.fence) >= value)
        return;
    if (SUCCEEDED(ID3D12Fence_SetEventOnCompletion(nr.fence, value, nr.fence_event)))
        WaitForSingleObject(nr.fence_event, 5000);
}

static void jnr_barrier(ID3D12Resource *r, D3D12_RESOURCE_STATES from,
                        D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b;

    memset(&b, 0, sizeof(b));
    b.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource   = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter  = to;
    ID3D12GraphicsCommandList_ResourceBarrier(nr.list, 1, &b);
}

/* ---- resources ---- */

static ID3D12Resource *jnr_tex(int w, int h, D3D12_RESOURCE_FLAGS flags,
                               D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES hp;
    D3D12_RESOURCE_DESC rd;
    ID3D12Resource *res = NULL;

    memset(&hp, 0, sizeof(hp));
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width            = (UINT64)w;
    rd.Height           = (UINT)h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    /* The model writes through a UAV and B8G8R8A8 cannot be one; R8G8B8A8 can,
     * and it reads RGBA just as happily (the player verified both). */
    rd.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.SampleDesc.Count   = 1;
    rd.Layout             = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags              = flags;
    if (FAILED(ID3D12Device_CreateCommittedResource(nr.dev, &hp,
            D3D12_HEAP_FLAG_NONE, &rd, state, NULL, &IID_ID3D12Resource,
            (void **)&res)))
        return NULL;
    return res;
}

static ID3D12Resource *jnr_buffer(UINT64 bytes, D3D12_HEAP_TYPE type,
                                  D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES hp;
    D3D12_RESOURCE_DESC rd;
    ID3D12Resource *res = NULL;

    memset(&hp, 0, sizeof(hp));
    hp.Type = type;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(ID3D12Device_CreateCommittedResource(nr.dev, &hp,
            D3D12_HEAP_FLAG_NONE, &rd, state, NULL, &IID_ID3D12Resource,
            (void **)&res)))
        return NULL;
    return res;
}

/* The footprint a single-subresource RGBA8 2D texture would get from
 * GetCopyableFootprints. Built by hand on purpose: GetDesc returns a struct by
 * value, which is the one part of the D3D12 C bindings mingw gets wrong. */
static void jnr_footprint(D3D12_PLACED_SUBRESOURCE_FOOTPRINT *fp, int w, int h)
{
    memset(fp, 0, sizeof(*fp));
    fp->Offset             = 0;
    fp->Footprint.Format   = DXGI_FORMAT_R8G8B8A8_UNORM;
    fp->Footprint.Width    = (UINT)w;
    fp->Footprint.Height   = (UINT)h;
    fp->Footprint.Depth    = 1;
    fp->Footprint.RowPitch = nr.row_pitch;
}

JNR_API void jnr_free_size(void)
{
    if (nr.feature) {
        jnr_cpu_wait(nr.fence_val);
        if (nr.f_release)
            nr.f_release(nr.feature);
        nr.feature = NULL;
    }
    if (nr.in_tex)   { ID3D12Resource_Release(nr.in_tex);   nr.in_tex   = NULL; }
    if (nr.out_tex)  { ID3D12Resource_Release(nr.out_tex);  nr.out_tex  = NULL; }
    if (nr.upload)   { ID3D12Resource_Release(nr.upload);   nr.upload   = NULL; }
    if (nr.readback) { ID3D12Resource_Release(nr.readback); nr.readback = NULL; }
    nr.w = nr.h = 0;
    nr.row_pitch = 0;
}

/* ---- setup ---- */

JNR_API int jnr_init(void)
{
    typedef HRESULT (WINAPI *PFN_D3D12CreateDevice)(IUnknown *, D3D_FEATURE_LEVEL,
                                                    REFIID, void **);
    typedef HRESULT (WINAPI *PFN_CreateDXGIFactory1)(REFIID, void **);
    PFN_D3D12CreateDevice create12;
    PFN_CreateDXGIFactory1 create_factory;
    D3D12_COMMAND_QUEUE_DESC qd;
    IDXGIFactory1 *factory = NULL;
    IDXGIAdapter1 *adapter = NULL;
    HMODULE d3d12, dxgi;
    wchar_t snippet[MAX_PATH], shim[MAX_PATH], data_dir[MAX_PATH];
    unsigned res;

    if (nr.state == 1)
        return 0;
    if (nr.state)
        return -1;
    nr.state = -1;                      /* every failure below is permanent */

    if (jnr_find(L"nvngx_dlssnr.dll", L"JASNA_DLSSNR_DLL", snippet, MAX_PATH) < 0)
        return -1;
    if (jnr_find(L"nvngx.dll_jasna.dll", NULL, shim, MAX_PATH) < 0)
        return -1;

    nr.shim = LoadLibraryW(shim);
    if (!nr.shim)
        return jnr_fail("LoadLibrary(%ls) failed with %lu", shim,
                        (unsigned long)GetLastError());
    nr.f_load     = (PFN_ffnr_load)    (void *)GetProcAddress(nr.shim, "ffnr_load");
    nr.f_init     = (PFN_ffnr_init)    (void *)GetProcAddress(nr.shim, "ffnr_init");
    nr.f_create   = (PFN_ffnr_create)  (void *)GetProcAddress(nr.shim, "ffnr_create");
    nr.f_evaluate = (PFN_ffnr_evaluate)(void *)GetProcAddress(nr.shim, "ffnr_evaluate");
    nr.f_release  = (PFN_ffnr_release) (void *)GetProcAddress(nr.shim, "ffnr_release");
    nr.f_shutdown = (PFN_ffnr_shutdown)(void *)GetProcAddress(nr.shim, "ffnr_shutdown");
    if (!nr.f_load || !nr.f_init || !nr.f_create || !nr.f_evaluate)
        return jnr_fail("%ls is missing its entry points", shim);
    if (nr.f_load(snippet) != 0)
        return jnr_fail("%ls could not be loaded", snippet);

    /* Loaded by hand so the import table stays free of d3d12.dll - the sidecar
     * has to keep starting on machines without an RTX 50 series GPU. */
    dxgi  = LoadLibraryA("dxgi.dll");
    d3d12 = GetModuleHandleA("d3d12.dll");
    if (!d3d12)
        d3d12 = LoadLibraryA("d3d12.dll");
    create12 = d3d12 ? (PFN_D3D12CreateDevice)(void *)
                       GetProcAddress(d3d12, "D3D12CreateDevice") : NULL;
    create_factory = dxgi ? (PFN_CreateDXGIFactory1)(void *)
                            GetProcAddress(dxgi, "CreateDXGIFactory1") : NULL;
    if (!create12 || !create_factory)
        return jnr_fail("d3d12.dll/dxgi.dll unavailable");

    /* First adapter that gives a device. Torch picks its own CUDA device and
     * we cannot see that choice from here, but the frames come through system
     * memory either way, so a mismatch costs a copy, not correctness. */
    if (FAILED(create_factory(&IID_IDXGIFactory1, (void **)&factory)))
        return jnr_fail("CreateDXGIFactory1 failed");
    for (UINT i = 0; SUCCEEDED(IDXGIFactory1_EnumAdapters1(factory, i, &adapter)); i++) {
        DXGI_ADAPTER_DESC1 ad;

        if (SUCCEEDED(IDXGIAdapter1_GetDesc1(adapter, &ad)) &&
            !(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(create12((IUnknown *)adapter, D3D_FEATURE_LEVEL_11_0,
                               &IID_ID3D12Device, (void **)&nr.dev)))
            break;
        IDXGIAdapter1_Release(adapter);
        adapter = NULL;
    }
    if (adapter)
        IDXGIAdapter1_Release(adapter);
    IDXGIFactory1_Release(factory);
    if (!nr.dev)
        return jnr_fail("no D3D12 device on any adapter");

    memset(&qd, 0, sizeof(qd));
    qd.Type  = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(ID3D12Device_CreateCommandQueue(nr.dev, &qd, &IID_ID3D12CommandQueue,
                                               (void **)&nr.queue)))
        return jnr_fail("CreateCommandQueue failed");
    for (int i = 0; i < JNR_FRAMES; i++)
        if (FAILED(ID3D12Device_CreateCommandAllocator(nr.dev,
                D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator,
                (void **)&nr.alloc[i])))
            return jnr_fail("CreateCommandAllocator failed");
    if (FAILED(ID3D12Device_CreateCommandList(nr.dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                              nr.alloc[0], NULL,
                                              &IID_ID3D12GraphicsCommandList,
                                              (void **)&nr.list)))
        return jnr_fail("CreateCommandList failed");
    ID3D12GraphicsCommandList_Close(nr.list);

    if (FAILED(ID3D12Device_CreateFence(nr.dev, 0, D3D12_FENCE_FLAG_NONE,
                                        &IID_ID3D12Fence, (void **)&nr.fence)))
        return jnr_fail("CreateFence failed");
    nr.fence_event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!nr.fence_event)
        return jnr_fail("CreateEvent failed");

    /* The snippet writes its own logs here when a driver-side log level is
     * set; it must be a directory it can create files in. */
    if (!GetTempPathW(MAX_PATH - 16, data_dir))
        return jnr_fail("GetTempPath failed");
    wcscat(data_dir, L"jasna-ngx");
    CreateDirectoryW(data_dir, NULL);

    res = nr.f_init(nr.dev, data_dir);
    if (res != NGX_OK)
        /* 0xBAD00001 is FeatureNotSupported - a pre-Blackwell GPU, or a driver
         * without the model. */
        return jnr_fail("NGX init refused (0x%X)", res);

    nr.state = 1;
    jnr_err[0] = 0;
    return 0;
}

JNR_API int jnr_ensure(int w, int h)
{
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    unsigned res;

    if (nr.state != 1)
        return -1;
    if (w <= 0 || h <= 0)
        return jnr_fail("bad size %dx%d", w, h);
    if (nr.feature && nr.w == w && nr.h == h)
        return 0;
    jnr_free_size();

    nr.row_pitch = (UINT)(((UINT64)w * 4 + JNR_PITCH_ALIGN - 1) &
                          ~(UINT64)(JNR_PITCH_ALIGN - 1));

    nr.in_tex  = jnr_tex(w, h, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_COMMON);
    nr.out_tex = jnr_tex(w, h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON);
    nr.upload   = jnr_buffer((UINT64)nr.row_pitch * h, D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
    nr.readback = jnr_buffer((UINT64)nr.row_pitch * h, D3D12_HEAP_TYPE_READBACK,
                             D3D12_RESOURCE_STATE_COPY_DEST);
    if (!nr.in_tex || !nr.out_tex || !nr.upload || !nr.readback) {
        jnr_free_size();
        return jnr_fail("resource creation failed at %dx%d", w, h);
    }
    jnr_footprint(&fp, w, h);

    /* CreateFeature records its own setup work, so it needs an open list and a
     * submission before the first evaluate. */
    jnr_cpu_wait(nr.alloc_val[0]);
    ID3D12CommandAllocator_Reset(nr.alloc[0]);
    ID3D12GraphicsCommandList_Reset(nr.list, nr.alloc[0], NULL);
    res = nr.f_create(nr.list, w, h, 0 /* preset: model default */, 2, &nr.feature);
    ID3D12GraphicsCommandList_Close(nr.list);
    if (res != NGX_OK || !nr.feature) {
        nr.feature = NULL;
        jnr_free_size();
        return jnr_fail("CreateFeature failed (0x%X) at %dx%d", res, w, h);
    }
    ID3D12CommandQueue_ExecuteCommandLists(nr.queue, 1,
                                           (ID3D12CommandList *const *)&nr.list);
    ID3D12CommandQueue_Signal(nr.queue, nr.fence, ++nr.fence_val);
    nr.alloc_val[0] = nr.fence_val;
    jnr_cpu_wait(nr.fence_val);

    nr.w = w;
    nr.h = h;
    nr.frame = 0;
    jnr_err[0] = 0;
    return 0;
}

/* ---- per frame ---- */

JNR_API int jnr_process(const unsigned char *in_rgb, unsigned char *out_rgb,
                        int w, int h, const FFNRTune *tune)
{
    D3D12_TEXTURE_COPY_LOCATION src, dst;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    D3D12_RANGE range;
    unsigned char *map = NULL;
    int slot = nr.frame % JNR_FRAMES;
    unsigned res;

    if (nr.state != 1 || !nr.feature)
        return jnr_fail("not initialised");
    if (w != nr.w || h != nr.h)
        return jnr_fail("size %dx%d does not match the model's %dx%d",
                        w, h, nr.w, nr.h);
    if (!in_rgb || !out_rgb || !tune)
        return jnr_fail("null argument");

    jnr_footprint(&fp, w, h);

    /* RGB24 -> RGBA into the upload heap, straight at the aligned row pitch so
     * there is no second staging copy. Alpha is set to opaque; the model does
     * not read it, but leaving it uninitialised makes the readback harder to
     * eyeball when something goes wrong. */
    range.Begin = range.End = 0;                 /* nothing to read back in */
    if (FAILED(ID3D12Resource_Map(nr.upload, 0, &range, (void **)&map)) || !map)
        return jnr_fail("upload map failed");
    for (int y = 0; y < h; y++) {
        const unsigned char *s = in_rgb + (size_t)y * w * 3;
        unsigned char *d = map + (size_t)y * nr.row_pitch;

        for (int x = 0; x < w; x++, s += 3, d += 4) {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
            d[3] = 0xFF;
        }
    }
    ID3D12Resource_Unmap(nr.upload, 0, NULL);

    jnr_cpu_wait(nr.alloc_val[slot]);
    ID3D12CommandAllocator_Reset(nr.alloc[slot]);
    ID3D12GraphicsCommandList_Reset(nr.list, nr.alloc[slot], NULL);

    /* upload buffer -> in_tex */
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    src.pResource       = nr.upload;
    src.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = fp;
    dst.pResource        = nr.in_tex;
    dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    jnr_barrier(nr.in_tex, D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12GraphicsCommandList_CopyTextureRegion(nr.list, &dst, 0, 0, 0, &src, NULL);

    /* the model */
    jnr_barrier(nr.in_tex,  D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    jnr_barrier(nr.out_tex, D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    res = nr.f_evaluate(nr.list, nr.feature, nr.in_tex, nr.out_tex, w, h, tune);
    jnr_barrier(nr.in_tex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COMMON);
    jnr_barrier(nr.out_tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE);

    /* out_tex -> readback buffer */
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    src.pResource        = nr.out_tex;
    src.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    dst.pResource       = nr.readback;
    dst.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    ID3D12GraphicsCommandList_CopyTextureRegion(nr.list, &dst, 0, 0, 0, &src, NULL);
    jnr_barrier(nr.out_tex, D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12_RESOURCE_STATE_COMMON);
    ID3D12GraphicsCommandList_Close(nr.list);

    if (res != NGX_OK) {
        jnr_free_size();
        return jnr_fail("evaluate failed (0x%X)", res);
    }

    ID3D12CommandQueue_ExecuteCommandLists(nr.queue, 1,
                                           (ID3D12CommandList *const *)&nr.list);
    ID3D12CommandQueue_Signal(nr.queue, nr.fence, ++nr.fence_val);
    nr.alloc_val[slot] = nr.fence_val;
    nr.frame++;
    jnr_cpu_wait(nr.fence_val);

    range.Begin = 0;
    range.End   = (SIZE_T)nr.row_pitch * h;
    map = NULL;
    if (FAILED(ID3D12Resource_Map(nr.readback, 0, &range, (void **)&map)) || !map)
        return jnr_fail("readback map failed");
    for (int y = 0; y < h; y++) {
        const unsigned char *s = map + (size_t)y * nr.row_pitch;
        unsigned char *d = out_rgb + (size_t)y * w * 3;

        for (int x = 0; x < w; x++, s += 4, d += 3) {
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    }
    range.Begin = range.End = 0;                 /* nothing written back out */
    ID3D12Resource_Unmap(nr.readback, 0, &range);
    return 0;
}

JNR_API void jnr_shutdown(void)
{
    jnr_free_size();
    if (nr.f_shutdown && nr.state == 1)
        nr.f_shutdown();
    if (nr.list)  { ID3D12GraphicsCommandList_Release(nr.list); nr.list = NULL; }
    for (int i = 0; i < JNR_FRAMES; i++)
        if (nr.alloc[i]) { ID3D12CommandAllocator_Release(nr.alloc[i]); nr.alloc[i] = NULL; }
    if (nr.queue) { ID3D12CommandQueue_Release(nr.queue); nr.queue = NULL; }
    if (nr.fence) { ID3D12Fence_Release(nr.fence); nr.fence = NULL; }
    if (nr.fence_event) { CloseHandle(nr.fence_event); nr.fence_event = NULL; }
    if (nr.dev)   { ID3D12Device_Release(nr.dev); nr.dev = NULL; }
    /* nr.shim stays loaded: the snippet holds CUDA state that does not survive
     * being unloaded and reloaded in the same process. */
    if (nr.state == 1)
        nr.state = 0;
}
