#!/usr/bin/env python3
"""CPU model of the Ascend C kernel's arithmetic (development aid only).

This script runs no device code.  It replays, on the host, the float32 operation
sequence the device kernel uses, so numerical-risk cases can be checked before
spending device time.

Numerical design being modelled:

  * Every accumulation is *local first*.  A tile reduces its own rows into a
    tile mean and a tile-centred second moment; only these small per-tile
    quantities are merged (with Kahan compensation) into the segment result.
    Reducing first is what preserves precision: summing thousands of raw
    6e4-scale values into a single float32 accumulator would need a final
    relative accuracy of ~1e-11, which float32 cannot represent at all, whereas
    a local mean is accurate to the float32 epsilon of its own magnitude.

  * Resident path (whole segment fits in UB): one reference row, one pass.
  * Streaming path (longer segments): per-tile local mean, then a weighted merge
    using the parallel-axis identity
        var = sum_t W_t * (within_t + (mean_t - mean)^2) / N.

Usage:
    python3 tools/dev/kernel_precision_model.py --cases tools/data/public_cases.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.correctness.check_correctness import compare_case  # noqa: E402
from tools.correctness.generate_case import generate_case  # noqa: E402
from tools.reference.reference import write_inputs, write_outputs  # noqa: E402

# Must mirror the constants in solution/op_kernel/rsm_solution_core.h.
K_MAX_LOCAL_ROWS = 24
F32 = np.float32


def _pairwise_sum(values: np.ndarray) -> float:
    """Pairwise (halving-tree) float32 reduction, as a vector unit performs it.

    Returns a Python float so callers always divide by a true scalar.
    """
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(F32(values[0]))
    half = values.size // 2
    return float(F32(_pairwise_sum(values[:half]) + _pairwise_sum(values[half:])))


class Kahan:
    """Vector Kahan accumulator (mirrors the device's compensated vectors)."""

    def __init__(self, size: int = 1) -> None:
        self.value = np.zeros(size, dtype=F32)
        self.correction = np.zeros(size, dtype=F32)

    def add(self, term) -> None:
        term = np.asarray(term, dtype=F32)
        term = (term - self.correction).astype(F32)
        nxt = (self.value + term).astype(F32)
        self.correction = ((nxt - self.value).astype(F32) - term).astype(F32)
        self.value = nxt


class DoubleSingle:
    """Two-float (value + error) compensated accumulator.

    Kahan's single correction term is not enough here: when the summands are
    large and one-signed, the correction itself grows until it is no longer
    balanced against the addends (float32 has ~7 digits, the contract needs
    ~11).  Carrying the error term explicitly and folding the *pair* back into
    the accumulator with TwoSum keeps the result accurate to ~2^-46 relative,
    which is what makes an adversarial "distant centre" input tractable.
    """

    def __init__(self, size: int) -> None:
        self.value = np.zeros(size, dtype=F32)
        self.error = np.zeros(size, dtype=F32)

    def add(self, term) -> None:
        term = np.asarray(term, dtype=F32)
        total = (self.value + term).astype(F32)
        # TwoSum: the following two residuals recover the exact rounding error.
        b_virtual = (total - self.value).astype(F32)
        a_virtual = (total - b_virtual).astype(F32)
        b_round = (term - b_virtual).astype(F32)
        a_round = (self.value - a_virtual).astype(F32)
        error = (a_round + b_round).astype(F32)
        self.value = total
        self.error = (self.error + error).astype(F32)


class DoubleSingleVector:
    """A pair of float32 vectors that together carry ~2^-46 relative accuracy."""

    def __init__(self, value, error=None) -> None:
        self.value = np.asarray(value, dtype=F32)
        self.error = (np.zeros_like(self.value) if error is None
                      else np.asarray(error, dtype=F32))

    @property
    def combined(self) -> np.ndarray:
        return (self.value + self.error).astype(F32)

    def add(self, term, term_error=None) -> None:
        """Accumulate a two-float addend with a full TwoSum fold."""
        term = np.asarray(term, dtype=F32)
        term_error = (np.zeros_like(term) if term_error is None
                      else np.asarray(term_error, dtype=F32))
        total = (self.value + term).astype(F32)
        b_virtual = (total - self.value).astype(F32)
        a_virtual = (total - b_virtual).astype(F32)
        error = (((term - b_virtual).astype(F32) + (self.value - a_virtual).astype(F32)
                  ).astype(F32) + term_error).astype(F32)
        self.value = total
        self.error = (self.error + error).astype(F32)

    def normalized(self) -> "DoubleSingleVector":
        """Renormalise so |error| <= ulp(value)/2 (Dekker's quick TwoSum)."""
        total = (self.value + self.error).astype(F32)
        error = ((self.value - total).astype(F32) + self.error).astype(F32)
        return DoubleSingleVector(total, error)

    def subtract(self, other: "DoubleSingleVector") -> "DoubleSingleVector":
        total = (self.value - other.value).astype(F32)
        b_virtual = (total - self.value).astype(F32)
        a_virtual = (total - b_virtual).astype(F32)
        error = ((((-other.value) - b_virtual).astype(F32) +
                  (self.value - a_virtual).astype(F32)).astype(F32) +
                 self.error - other.error).astype(F32)
        return DoubleSingleVector(total, error)

    def scaled_pair(self, factor: "DoubleSingleVector") -> "DoubleSingleVector":
        """Multiply by a two-float scalar, keeping the exact product split."""
        high = self.scaled(factor.value)
        low = (factor.error * self.value).astype(F32)
        return DoubleSingleVector(high.value, (high.error + low).astype(F32))

    def scaled(self, factor: float) -> "DoubleSingleVector":
        """Multiply by a float32 scalar, keeping the exact product split."""
        factor32 = F32(factor)
        product = (self.value * factor32).astype(F32)
        # TwoProduct via FMA-free Dekker split.
        splitter = F32(4097.0)  # 2^12 + 1
        a_big = (splitter * self.value).astype(F32)
        a_hi = (a_big - (a_big - self.value)).astype(F32)
        a_lo = (self.value - a_hi).astype(F32)
        b_big = (splitter * factor32).astype(F32)
        b_hi = (b_big - (b_big - factor32)).astype(F32)
        b_lo = (factor32 - b_hi).astype(F32)
        error = (((a_hi * b_hi - product).astype(F32) +
                  a_hi * b_lo + a_lo * b_hi + a_lo * b_lo).astype(F32))
        error = (error + self.error * factor32).astype(F32)
        return DoubleSingleVector(product, error)


def _tiles(rows: int, tile_rows: int):
    base = 0
    while base < rows:
        stop = min(base + tile_rows, rows)
        yield base, stop
        base = stop


def _tile_statistics(seg_x, weights, base, stop, d):
    """One tile: its total weight plus its weighted mean.

    The mean accumulates the weighted deviation from the tile's first row in a
    two-float (value + error) accumulator, so the running sum stays at the scale
    of the deviations and its rounding error is carried explicitly.
    """
    tile_weight = _pairwise_sum(weights[base:stop])
    if tile_weight <= 0.0:
        return None, 0.0
    reference = seg_x[base]
    residual = DoubleSingle(d)
    for row in range(base + 1, stop):
        diff = (seg_x[row] - reference).astype(F32)
        residual.add((diff * weights[row]).astype(F32))
    offset = (residual.value + residual.error).astype(F32)
    tile_mean = DoubleSingleVector(reference,
                                   (offset / tile_weight).astype(F32))
    return tile_mean, tile_weight


def _segment_mean(seg_x, weights, rows, d):
    """Weighted mean via a two-float online update, rounded once at the end.

    mean += (w_i/W_i) * (x_i - mean) is exact in real arithmetic, and its step is
    a deviation from the running mean, so no intermediate ever has to resolve
    digits finer than the mean itself.  Holding the running mean as a float32
    pair (value + error) keeps the accumulated rounding far below one float32
    ulp; the pair is rounded exactly once, at the end.

    That single rounding matters: in the low-mass-tail fixture the mean moves
    from 65504 to 0.145 on the very last row, so rounding earlier would quantise
    the result onto 65504's 0.0039 grid - the entire error budget.
    """
    mean = DoubleSingleVector(np.zeros(d, dtype=F32))
    total_weight = DoubleSingle(1)
    started = False
    for row in range(rows):
        step_weight = float(weights[row])
        if step_weight <= 0.0:
            continue
        if not started:
            mean = DoubleSingleVector(seg_x[row])
            total_weight = DoubleSingle(1)
            total_weight.value[0] = F32(step_weight)
            started = True
            continue
        running = float(total_weight.value[0] + total_weight.error[0])
        weight_after = float(F32(running + F32(step_weight)))
        quotient = F32(step_weight / weight_after)
        # Split the ratio with an exact product so its residual is captured:
        # rounding the ratio alone would put a 6e-8 relative error onto a step
        # of magnitude ~65504, i.e. 0.023 absolute.
        splitter = F32(4097.0)
        a_big = (splitter * quotient).astype(F32)
        a_hi = (a_big - (a_big - quotient)).astype(F32)
        a_lo = (quotient - a_hi).astype(F32)
        b_big = (splitter * F32(weight_after)).astype(F32)
        b_hi = (b_big - (b_big - F32(weight_after))).astype(F32)
        b_lo = (F32(weight_after) - b_hi).astype(F32)
        product = (quotient * F32(weight_after)).astype(F32)
        ratio_error = ((((a_hi * b_hi).astype(F32) - product).astype(F32) +
                        (a_hi * b_lo).astype(F32) + (a_lo * b_hi).astype(F32) +
                        (a_lo * b_lo).astype(F32)).astype(F32))
        ratio = DoubleSingleVector(quotient,
                                   (ratio_error + (step_weight - product).astype(F32)).astype(F32))
        step = DoubleSingleVector(seg_x[row]).subtract(mean).scaled_pair(ratio)
        mean.add(step.value, step.error)
        mean = mean.normalized()
        total_weight.add(F32(step_weight))
    if not started:
        return mean, 0.0
    return mean, float(total_weight.value[0] + total_weight.error[0])


def _segment_moments(seg_score: np.ndarray, seg_x: np.ndarray, epsilon: F32):
    d = seg_x.shape[1]
    rows = seg_x.shape[0]
    maximum = F32(np.max(seg_score))
    weights = np.exp((seg_score - maximum).astype(F32)).astype(F32)

    mean_pair, weight_sum = _segment_mean(seg_x, weights, rows, d)
    mean_value = mean_pair.combined

    # Centred second moment: centring on the final float32 mean keeps every term
    # at the scale of the variance, so nothing cancels.
    second = DoubleSingleVector(np.zeros(d, dtype=F32))
    for base, stop in _tiles(rows, K_MAX_LOCAL_ROWS):
        chunk = DoubleSingleVector(np.zeros(d, dtype=F32))
        for row in range(base, stop):
            diff = (seg_x[row] - mean_value).astype(F32)
            chunk.add(((diff * diff).astype(F32) * weights[row]).astype(F32))
        second.add(chunk.value, chunk.error)

    var = (second.combined / weight_sum).astype(F32)
    var = np.maximum(var, F32(0.0)).astype(F32)
    rstd = (1.0 / np.sqrt((var + epsilon).astype(F32))).astype(F32)
    lse = F32(float(np.log(np.float64(weight_sum))) + float(maximum))
    return mean_value, rstd, lse


def model_case(score: np.ndarray, x: np.ndarray, offsets: np.ndarray,
               epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments, d = offsets.size - 1, x.shape[1]
    mean_out = np.empty((segments, d), dtype=F32)
    rstd_out = np.empty((segments, d), dtype=F32)
    lse_out = np.empty(segments, dtype=F32)
    eps32 = F32(epsilon)
    for s in range(segments):
        begin, end = int(offsets[s]), int(offsets[s + 1])
        seg_score = score[begin:end].astype(F32)
        seg_x = x[begin:end].astype(F32)
        if end - begin == 1:
            mean_out[s] = seg_x[0]
            rstd_out[s] = F32(1.0 / np.sqrt(np.float64(eps32)))
            lse_out[s] = seg_score[0]
            continue
        mean_out[s], rstd_out[s], lse_out[s] = _segment_moments(seg_score, seg_x, eps32)
    return mean_out, rstd_out, lse_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path,
                        default=Path("tools/data/public_cases.json"))
    parser.add_argument("--work-dir", type=Path,
                        default=Path("artifacts/precision_model"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()

    specs = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.case_id:
        specs = [s for s in specs if s["case_id"] in set(args.case_id)]
    if args.limit:
        specs = specs[: args.limit]

    failures = 0
    worst, worst_id = 0.0, ""
    for spec in specs:
        score, x, offsets, meta = generate_case(spec)
        epsilon = float(spec.get("epsilon", 1e-5))
        outputs = model_case(score, x, offsets, epsilon)
        case_dir = args.work_dir / spec["case_id"]
        write_inputs(case_dir / "input", score, x, offsets, dict(meta, epsilon=epsilon))
        write_outputs(case_dir / "output", *outputs)
        result = compare_case(case_dir / "input", case_dir / "output", epsilon)
        normalized = max(
            result["outputs"][name]["max_normalized_error"]
            for name in ("mean", "rstd", "logsumexp")
        )
        if normalized > worst:
            worst, worst_id = normalized, spec["case_id"]
        if not result["passed"]:
            failures += 1
            print(f"FAIL {spec['case_id']:44s} norm_err={normalized:.4f}")
            for name in ("mean", "rstd", "logsumexp"):
                entry = result["outputs"][name]
                if not entry["passed"]:
                    print(f"     {name}: max_norm={entry['max_normalized_error']:.4f} "
                          f"max_abs={entry['max_abs_error']:.6g} "
                          f"idx={entry['worst_normalized_index']}")
    print(f"\n{len(specs) - failures}/{len(specs)} cases pass the model; "
          f"worst normalized error {worst:.4f} ({worst_id})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
