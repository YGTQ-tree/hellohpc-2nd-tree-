---
name: cpu-hpc-skill
description: Optimize the selected single-threaded ARM CPU kernel within the public rules.
---

# Bitmatrix plan

Optimize only `bitmatrix`. Read `TASK.md`, `kernel.h`, and starter; edit only `kernel.cpp` and `compile_options.txt`. First run `bash tools/test_candidate.sh` and record per-case times. Preserve ABI, exact `uint32_t` counts, full output writes, and single-thread execution; do not use processes, libraries, input-specific shortcuts, or output reuse.

The row-major matrix stores 64 columns per word. Replace repeated per-column row tests with per-word bit-sliced counters: keep 14 zeroed 64-bit planes (counts up to 8192); for each selected row word, add all 64 bits using XOR and AND carry propagation, then reconstruct column counts. Iterate set bits of each mask word to skip unselected rows. Check zero, all-selected, and 8192-row cases.

Change one thing at a time; test after every edit. Profile only correct candidates, then benchmark; use the vectorization report to explain results. Try ARM flags only after the algorithm is correct. Keep the fastest correct source/flags pair. In the final four rounds, stop exploring, restore that pair, and test it again.
