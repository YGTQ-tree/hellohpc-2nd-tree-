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
#include <stdlib.h>

void compute_field_kernel(
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
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int q0 = (int)(((long long)q_count * tid) / nth);
        int q1 = (int)(((long long)q_count * (tid + 1)) / nth);
        for (int c = 0; c < c_count; ++c) {
            const float * __restrict center = centers + (size_t)c * dimension;
            const float * __restrict direction = trig_vec + (size_t)c * dimension;
            const float w = weights[c];
            const float s = scales[c];
            const float b = bias[c];
            const float t = trig_scale[c];
            for (int q = q0; q < q1; ++q) {
                const float * __restrict point = points + (size_t)q * dimension;
                float square_distance = 0.0f;
                float dot_product = 0.0f;
                #pragma omp simd reduction(+:square_distance,dot_product)
                for (int d = 0; d < dimension; ++d) {
                    const float x = point[d];
                    const float delta = x - center[d];
                    square_distance += delta * delta;
                    dot_product += x * direction[d];
                }
                accum[q] += (double)(w * expf(-s * square_distance)) + (double)(b * sinf(t * dot_product));
            }
        }
        for (int q = q0; q < q1; ++q) {
            output[q] = (float)accum[q];
        }
    }
    free(accum);
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
    command = ["gcc", "-Ofast", "-fopenmp", "-fPIC", "-shared", str(source), "-lm", "-o", str(library)]
    try:
        subprocess.run(command[:1] + ["-march=native"] + command[1:], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    shared = ctypes.CDLL(str(library))
    kernel = shared.compute_field_kernel
    float_pointer = ctypes.POINTER(ctypes.c_float)
    kernel.argtypes = [float_pointer] * 8 + [ctypes.c_int] * 3
    kernel.restype = None
    _KERNEL = kernel
    return kernel

def compute_field(points: Any, centers: Any, weights: Any, scales: Any, bias: Any, trig_scale: Any, trig_vec: Any) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float32, order="C") for value in (points, centers, weights, scales, bias, trig_scale, trig_vec)]
    q_count, dimension = arrays[0].shape
    c_count = arrays[1].shape[0]
    result = np.zeros(q_count, dtype=np.float32)
    if q_count == 0:
        return result
    float_pointer = ctypes.POINTER(ctypes.c_float)
    pointers = [array.ctypes.data_as(float_pointer) for array in arrays]
    _load_kernel()(*pointers, result.ctypes.data_as(float_pointer), q_count, c_count, dimension)
    return result

_load_kernel()
