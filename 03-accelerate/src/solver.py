from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

_KERNEL_SOURCE = r"""
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__aarch64__)
#include <arm_neon.h>

static inline float32x4_t exp4(float32x4_t x)
{
    x = vmaxq_f32(x, vdupq_n_f32(-80.0f));
    x = vminq_f32(x, vdupq_n_f32(88.0f));
    const float32x4_t kf = vrndnq_f32(vmulq_n_f32(x, 1.4426950408889634f));
    float32x4_t r = vfmaq_f32(x, kf, vdupq_n_f32(-0.693147182464599609375f));
    r = vfmaq_f32(r, kf, vdupq_n_f32(1.904654299957768e-09f));
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    const int32x4_t k = vcvtq_s32_f32(kf);
    const int32x4_t e = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

static inline float32x4_t sin4(float32x4_t x)
{
    const float32x4_t kf = vrndnq_f32(vmulq_n_f32(x, 0.6366197723675814f));
    float32x4_t r = vfmaq_f32(x, kf, vdupq_n_f32(-1.57079637050628662109375f));
    r = vfmaq_f32(r, kf, vdupq_n_f32(4.371138828673793e-08f));
    const float32x4_t z = vmulq_f32(r, r);

    float32x4_t sp = vdupq_n_f32(1.0f / 362880.0f);
    sp = vfmaq_f32(vdupq_n_f32(-1.0f / 5040.0f), sp, z);
    sp = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), sp, z);
    sp = vfmaq_f32(vdupq_n_f32(-1.0f / 6.0f), sp, z);
    sp = vfmaq_f32(vdupq_n_f32(1.0f), sp, z);
    const float32x4_t sr = vmulq_f32(r, sp);

    float32x4_t cp = vdupq_n_f32(-1.0f / 3628800.0f);
    cp = vfmaq_f32(vdupq_n_f32(1.0f / 40320.0f), cp, z);
    cp = vfmaq_f32(vdupq_n_f32(-1.0f / 720.0f), cp, z);
    cp = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), cp, z);
    cp = vfmaq_f32(vdupq_n_f32(-0.5f), cp, z);
    cp = vfmaq_f32(vdupq_n_f32(1.0f), cp, z);

    const int32x4_t quad = vandq_s32(vcvtq_s32_f32(kf), vdupq_n_s32(3));
    float32x4_t res = sr;
    res = vbslq_f32(vceqq_s32(quad, vdupq_n_s32(1)), cp, res);
    res = vbslq_f32(vceqq_s32(quad, vdupq_n_s32(2)), vnegq_f32(sr), res);
    res = vbslq_f32(vceqq_s32(quad, vdupq_n_s32(3)), vnegq_f32(cp), res);
    return res;
}

static void kernel_arm(
    const float * __restrict points,
    const float * __restrict centers,
    const float * __restrict weights,
    const float * __restrict scales,
    const float * __restrict bias,
    const float * __restrict trig_scale,
    const float * __restrict trig_vec,
    float * __restrict output,
    int q_count, int c_count, int dimension)
{
    if (q_count <= 0) return;
    const size_t stride = (size_t)q_count + 16;
    float *xt = (float *)malloc((size_t)dimension * stride * sizeof(float));
    double *accum = (double *)calloc((size_t)q_count, sizeof(double));
    if (!xt || !accum) {
        free(xt);
        free(accum);
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int q = 0; q < q_count; ++q) {
        const float *point = points + (size_t)q * dimension;
        for (int d = 0; d < dimension; ++d) {
            xt[(size_t)d * stride + q] = point[d];
        }
    }

    const int block = 256;
    #pragma omp parallel for schedule(static)
    for (int qb = 0; qb < q_count; qb += block) {
        int qe = qb + block;
        if (qe > q_count) qe = q_count;
        for (int c = 0; c < c_count; ++c) {
            const float *center = centers + (size_t)c * dimension;
            const float *direction = trig_vec + (size_t)c * dimension;
            const float w = weights[c];
            const float s = scales[c];
            const float b = bias[c];
            const float t = trig_scale[c];
            int q = qb;
            for (; q + 3 < qe; q += 4) {
                float32x4_t r4 = vdupq_n_f32(0.0f);
                float32x4_t p4 = vdupq_n_f32(0.0f);
                #pragma GCC unroll 4
                for (int d = 0; d < dimension; ++d) {
                    const float32x4_t x4 = vld1q_f32(xt + (size_t)d * stride + q);
                    const float32x4_t diff = vsubq_f32(x4, vdupq_n_f32(center[d]));
                    r4 = vfmaq_f32(r4, diff, diff);
                    p4 = vfmaq_f32(p4, x4, vdupq_n_f32(direction[d]));
                }
                const float32x4_t g4 = exp4(vmulq_n_f32(r4, -s));
                const float32x4_t h4 = sin4(vmulq_n_f32(p4, t));
                const float32x4_t term4 = vfmaq_n_f32(vmulq_n_f32(g4, w), h4, b);
                const float64x2_t tl = vcvt_f64_f32(vget_low_f32(term4));
                const float64x2_t th = vcvt_f64_f32(vget_high_f32(term4));
                const float64x2_t al = vld1q_f64(accum + q);
                const float64x2_t ah = vld1q_f64(accum + q + 2);
                vst1q_f64(accum + q, vaddq_f64(al, tl));
                vst1q_f64(accum + q + 2, vaddq_f64(ah, th));
            }
            for (; q < qe; ++q) {
                const float *point = points + (size_t)q * dimension;
                float r = 0.0f;
                float p = 0.0f;
                for (int d = 0; d < dimension; ++d) {
                    const float delta = point[d] - center[d];
                    r += delta * delta;
                    p += point[d] * direction[d];
                }
                accum[q] += (double)(w * expf(-s * r)) + (double)(b * sinf(t * p));
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int q = 0; q < q_count; ++q) {
        output[q] = (float)accum[q];
    }
    free(xt);
    free(accum);
}
#endif

static void kernel_generic(
    const float * __restrict points,
    const float * __restrict centers,
    const float * __restrict weights,
    const float * __restrict scales,
    const float * __restrict bias,
    const float * __restrict trig_scale,
    const float * __restrict trig_vec,
    float * __restrict output,
    int q_count, int c_count, int dimension)
{
    double *accum = (double *)calloc((size_t)q_count, sizeof(double));
    if (!accum) return;
    const int block = 256;
    #pragma omp parallel for schedule(static)
    for (int qb = 0; qb < q_count; qb += block) {
        int qe = qb + block;
        if (qe > q_count) qe = q_count;
        for (int c = 0; c < c_count; ++c) {
            const float *center = centers + (size_t)c * dimension;
            const float *direction = trig_vec + (size_t)c * dimension;
            const float w = weights[c];
            const float s = scales[c];
            const float b = bias[c];
            const float t = trig_scale[c];
            for (int q = qb; q < qe; ++q) {
                const float *point = points + (size_t)q * dimension;
                double r = 0.0;
                double p = 0.0;
                #pragma omp simd reduction(+:r,p)
                for (int d = 0; d < dimension; ++d) {
                    const double x = point[d];
                    const double delta = x - (double)center[d];
                    r += delta * delta;
                    p += x * (double)direction[d];
                }
                accum[q] += (double)w * exp(-(double)s * r)
                          + (double)b * sin((double)t * p);
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < q_count; ++q) {
        output[q] = (float)accum[q];
    }
    free(accum);
}

void compute_field_kernel(
    const float *points,
    const float *centers,
    const float *weights,
    const float *scales,
    const float *bias,
    const float *trig_scale,
    const float *trig_vec,
    float *output,
    int q_count,
    int c_count,
    int dimension)
{
#if defined(__aarch64__)
    /* The hand-written vector math is tuned for the D=16 workload.  The
       D=32 formal case has wider phase ranges, so use libm there to retain
       the reference numerical behavior. */
    if (dimension == 16) {
        kernel_arm(points, centers, weights, scales, bias, trig_scale, trig_vec,
                   output, q_count, c_count, dimension);
    } else {
        kernel_generic(points, centers, weights, scales, bias, trig_scale,
                       trig_vec, output, q_count, c_count, dimension);
    }
#else
    kernel_generic(points, centers, weights, scales, bias, trig_scale, trig_vec,
                   output, q_count, c_count, dimension);
#endif
}
"""

_KERNEL = None
_KERNEL_TEMP = None


def _load_kernel():
    global _KERNEL, _KERNEL_TEMP
    if _KERNEL is not None:
        return _KERNEL
    _KERNEL_TEMP = tempfile.TemporaryDirectory(prefix="hellohpc-kernel-")
    build_dir = Path(_KERNEL_TEMP.name)
    source = build_dir / "kernel.c"
    library = build_dir / "kernel.so"
    source.write_text(_KERNEL_SOURCE, encoding="utf-8")
    command = [
        "gcc", "-Ofast", "-fopenmp", "-fPIC", "-shared",
        str(source), "-lm", "-o", str(library),
    ]
    try:
        subprocess.run(command[:1] + ["-march=native"] + command[1:],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
    shared = ctypes.CDLL(str(library))
    kernel = shared.compute_field_kernel
    float_pointer = ctypes.POINTER(ctypes.c_float)
    kernel.argtypes = [float_pointer] * 8 + [ctypes.c_int] * 3
    kernel.restype = None
    _KERNEL = kernel
    return kernel


def compute_field(points: Any, centers: Any, weights: Any, scales: Any,
                  bias: Any, trig_scale: Any, trig_vec: Any) -> np.ndarray:
    q_count, dimension = _shape_2d(points, "points")
    c_count, center_dimension = _shape_2d(centers, "centers")
    trig_rows, trig_dimension = _shape_2d(trig_vec, "trig_vec")
    _shape_1d(weights, "weights", expected=c_count)
    _shape_1d(scales, "scales", expected=c_count)
    _shape_1d(bias, "bias", expected=c_count)
    _shape_1d(trig_scale, "trig_scale", expected=c_count)
    if center_dimension != dimension:
        raise ValueError(f"centers has dim {center_dimension}, expected {dimension}")
    if trig_rows != c_count or trig_dimension != dimension:
        raise ValueError(
            f"trig_vec must have shape ({c_count}, {dimension}), "
            f"got ({trig_rows}, {trig_dimension})"
        )

    arrays = [np.asarray(value, dtype=np.float32, order="C")
              for value in (points, centers, weights, scales, bias, trig_scale, trig_vec)]
    result = np.empty(q_count, dtype=np.float32)
    if q_count == 0:
        return result

    float_pointer = ctypes.POINTER(ctypes.c_float)
    pointers = [array.ctypes.data_as(float_pointer) for array in arrays]
    _load_kernel()(*pointers, result.ctypes.data_as(float_pointer),
                   q_count, c_count, dimension)
    return result


def _shape_2d(value: Any, name: str) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"{name} must be 2D, got shape {shape}")
        return int(shape[0]), int(shape[1])
    rows = len(value)
    columns = len(value[0]) if rows else 0
    if any(len(row) != columns for row in value):
        raise ValueError(f"{name} must be rectangular")
    return rows, columns


def _shape_1d(value: Any, name: str, expected: int | None = None) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 1:
            raise ValueError(f"{name} must be 1D, got shape {shape}")
        size = int(shape[0])
    else:
        size = len(value)
    if expected is not None and size != expected:
        raise ValueError(f"{name} must have length {expected}, got {size}")
    return size


_load_kernel()
