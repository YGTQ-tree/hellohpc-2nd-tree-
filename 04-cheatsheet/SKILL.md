---
name: cpu-hpc-skill
description: Optimize the selected single-threaded ARM CPU kernel within the public rules.
---

# Bitmatrix plan

Optimize only `bitmatrix`. Read `TASK.md`, `kernel.h`, and starter; edit only `kernel.cpp` and `compile_options.txt`. First run `bash tools/test_candidate.sh` and record per-case times. Preserve ABI, exact `uint32_t` counts, full output writes, and single-thread execution; do not use processes, libraries, input-specific shortcuts, or output reuse.

The row-major matrix stores 64 columns per word. Replace repeated per-column row tests with per-word bit-sliced counters: keep 14 zeroed 64-bit planes (counts up to 8192). Add word `x` with `c=x`; per plane compute `next=p&c; p^=c; c=next`, stopping when `c=0`. Iterate mask set bits with `ctz` and `bits &= bits-1`; reconstruct and overwrite all 64 counts per word. Check zero, all-selected, 8192-row cases and avoid shifting by 64.

Change one thing at a time; test after every edit. Profile correct candidates and identify whether carry updates or count reconstruction dominates before optimizing; then benchmark and use the vectorization report to explain results. Try ARM flags only after the algorithm is correct. Keep the fastest correct source/flags pair. In the final four rounds, stop exploring, restore that pair, and test it again.
