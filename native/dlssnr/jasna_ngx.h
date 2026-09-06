/*
 * DLSS 5 Neural Rendering (NVIDIA NGX "dlssnr") for the jasna sidecar.
 *
 * Ported from ffplay's ffplay_ngx.h / ffplay_ngx.c in this same repository.
 * The player drives the model on D3D11 textures it already owns; jasna has
 * only CPU frames, so the host below is pure D3D12 with an upload buffer in
 * front of the model and a readback buffer behind it. That drops the D3D11
 * device, the cross-API shared textures and the shared fence entirely, which
 * is why this is shorter than the player's version rather than longer.
 *
 * Everything about how the snippet has to be called is unchanged, and the
 * forwarder (jasna_ngxshim.c) is a verbatim copy of the player's - see its
 * header comment for why a separately named module is required at all.
 */

#ifndef JASNA_NGX_H
#define JASNA_NGX_H

/* Model knobs. The snippet reads these on every evaluate except style, which
 * is baked in when the feature is created. Defaults are what the model itself
 * uses; none of it is documented by NVIDIA. */
typedef struct FFNRTune {
    float intensity;        /* DLSSNR.Intensity, 1.0                        */
    float local_structure;  /* DLSSNR.LocalStructureStrength, 1.0           */
    float local_tone;       /* DLSSNR.LocalToneStrength, 1.0                */
    float skin_structure;   /* DLSSNR.SkinStructureStrength; -1 = follow the
                             * local structure term, the model's own default,
                             * which is not the same as a strength of 0      */
    int   style;            /* DLSSNR.Style                                  */
    int   auto_mask;        /* DLSSNR.UseAutoMask                            */
    int   reset;            /* DLSSNR.Reset - drop temporal history (cut/seek)*/
} FFNRTune;

/* Forwarder entry points, resolved by name out of nvngx.dll_jasna.dll.
 * All of them return an NVSDK_NGX_Result (0x1 == success) except ffnr_load. */
typedef int      (*PFN_ffnr_load)(const wchar_t *snippet_path);
typedef unsigned (*PFN_ffnr_init)(void *d3d12_device, const wchar_t *data_path);
typedef unsigned (*PFN_ffnr_create)(void *cmdlist, int w, int h, int preset,
                                    int style, void **out_handle);
typedef unsigned (*PFN_ffnr_evaluate)(void *cmdlist, void *handle, void *color,
                                      void *output, int w, int h,
                                      const FFNRTune *tune);
typedef unsigned (*PFN_ffnr_release)(void *handle);
typedef void     (*PFN_ffnr_shutdown)(void);

#ifndef FFNR_SHIM_BUILD

/* ---- the flat C API ctypes calls from Python (jasna/dlss_nr.py) ----
 *
 * Deliberately free of pointers-to-COM and wide strings: every argument is an
 * int, a float or a byte buffer, so the Python side needs no Windows types.
 * All of it is single-threaded - one worker owns the device. */

#ifdef JNR_BUILD
#define JNR_API __declspec(dllexport)
#else
#define JNR_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up D3D12 and the NGX runtime. 0 on success, negative on failure -
 * which is normal and expected on anything but an RTX 50 series GPU with the
 * snippet DLL present. Call jnr_last_error() for the reason. */
JNR_API int jnr_init(void);

/* Size the model. Cheap to call every frame; only a real size change rebuilds
 * the feature. 0 on success. */
JNR_API int jnr_ensure(int w, int h);

/* Enhance one packed RGB24 frame. in_rgb and out_rgb are w*h*3 bytes and may
 * be the same buffer. reset drops the model's temporal history, which a seek
 * or a cut must do or the frame is blended against something unrelated.
 * 0 on success; on failure out_rgb is left untouched and the caller should
 * pass the frame through unchanged. */
JNR_API int jnr_process(const unsigned char *in_rgb, unsigned char *out_rgb,
                        int w, int h, const FFNRTune *tune);

/* Release the feature and the buffers but keep the device and the NGX runtime.
 * A resolution change goes through here: the snippet is not built to be
 * initialised, shut down and initialised again inside one process. */
JNR_API void jnr_free_size(void);

JNR_API void jnr_shutdown(void);

/* 1 once the model is loaded and sized. */
JNR_API int jnr_available(void);

/* Last failure, as ASCII. Never NULL. */
JNR_API const char *jnr_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* !FFNR_SHIM_BUILD */

#endif /* JASNA_NGX_H */
