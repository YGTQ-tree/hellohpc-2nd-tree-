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
    #pragma omp parallel for schedule(static) if(q_count >= 16)
    for (int q = 0; q < q_count; ++q) {
        const float *point = points + (size_t)q * dimension;
        double total = 0.0;

        for (int c = 0; c < c_count; ++c) {
            const float *center = centers + (size_t)c * dimension;
            const float *direction = trig_vec + (size_t)c * dimension;
            double square_distance = 0.0;
            double dot_product = 0.0;

            for (int d = 0; d < dimension; ++d) {
                const double x = point[d];
                const double delta = x - (double)center[d];
                square_distance += delta * delta;
                dot_product += x * (double)direction[d];
            }

            total += (double)weights[c] * exp(-(double)scales[c] * square_distance);
            total += (double)bias[c] * sin((double)trig_scale[c] * dot_product);
        }
        output[q] = (float)total;
    }
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
    command = ["gcc", "-Ofast", "-fopenmp", "-fPIC", "-shared",
               str(source), "-lm", "-o", str(library)]
    try:
        subprocess.run(
            command[:1] + ["-march=native"] + command[1:],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    shared = ctypes.CDLL(str(library))
    kernel = shared.compute_field_kernel
    float_pointer = ctypes.POINTER(ctypes.c_float)
    kernel.argtypes = [float_pointer] * 8 + [ctypes.c_int] * 3
    kernel.restype = None
    _KERNEL = kernel
    return kernel


def compute_field(
    points: Any,
    centers: Any,
    weights: Any,
    scales: Any,
    bias: Any,
    trig_scale: Any,
    trig_vec: Any,
) -> np.ndarray:
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

    arrays = [
        np.asarray(value, dtype=np.float32, order="C")
        for value in (points, centers, weights, scales, bias, trig_scale, trig_vec)
    ]
    result = np.empty(q_count, dtype=np.float32)
    if q_count == 0:
        return result

    float_pointer = ctypes.POINTER(ctypes.c_float)
    pointers = [array.ctypes.data_as(float_pointer) for array in arrays]
    _load_kernel()(*pointers, result.ctypes.data_as(float_pointer), q_count, c_count, dimension)
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


# Build the user-provided kernel during module setup, before timed calls begin.
_load_kernel()
