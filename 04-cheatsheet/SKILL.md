---
name: cpu-hpc-skill
description: Optimize the selected single-threaded ARM CPU kernel within the public rules.
---

# Bitmatrix Plan

Optimize only `bitmatrix`. Read `TASK.md`, `kernel.h`, and the starter. Do not inspect `spec.yaml`, evaluator, runner, tests, or references. Edit only `kernel.cpp` and `compile_options.txt`.

First run `bash tools/test_candidate.sh`; record correctness and per-case times. Preserve the ABI, exact `uint32_t` counts, and overwrite every output. Use one thread; no processes, external compute libraries, input-dependent shortcuts, or runtime output reuse.

Exploit row-major 64-bit words. Maintain 14 zeroed bit planes per matrix word (up to 8192 selected rows). For each selected row, load each word and add its 64 column bits to the planes with XOR and AND carry propagation. Then reconstruct each column count from its plane bits. Check the 8192-row boundary and all-zero/all-one masks. This replaces repeated per-column row tests with one pass over selected words.

Make one change at a time. After each change run `bash tools/test_candidate.sh`; profile only correct candidates with `bash tools/profile.sh`, then compare with `bash tools/bench.sh`. Use `bash tools/vec_report.sh` when it can explain a measured result. Try ARM-native compiler flags only after a correct baseline; reject flags that fail correctness or regress timing.

Keep the fastest fully correct kernel and its matching flags. In the final four rounds, stop exploring: restore that pair and run `bash tools/test_candidate.sh` once more. Leave those verified files in place.
