#!/bin/bash
# Development benchmark: rebuild and time every official case, then verify.
# Not part of the submission.
set -u
cd "$(dirname "$0")"
W=${1:-$(nproc)}
echo "== rebuild =="
rm -f *.o run
make all 2>&1 | tail -2
echo "== cases =="
declare -A COUNT=( [1]=32 [2]=64 [3]=128 [4]=256 )
declare -A SEED=( [1]=1001 [2]=2002 [3]=3003 [4]=4004 )
declare -A FULL=( [1]=4500 [2]=8000 [3]=15000 [4]=24000 )
declare -A ZERO=( [1]=60000 [2]=120000 [3]=240000 [4]=480000 )
for id in 1 2 3 4; do
  n=${COUNT[$id]}
  dir=cases/c$id
  [ -f "$dir/tasks.txt" ] || python3 ../utils/gencase.py "$n" "$dir" "${SEED[$id]}" >/dev/null
  start=$(date +%s.%N)
  HASHCLASH_STATS=1 HASHCLASH_THREADS=$W ./run "$dir/tasks.txt" 2>&1 | grep -v '^run: [0-9]* task'
  end=$(date +%s.%N)
  ms=$(echo "($end - $start) * 1000" | bc)
  v=$(python3 ../utils/verify.py "$dir/tasks.txt" | tail -1)
  s=$(python3 ../utils/score_curve.py --value "$ms" --full "${FULL[$id]}" --zero "${ZERO[$id]}" --gamma 1.5 | sed 's/.*score=\([0-9.]*\).*/\1/')
  printf "case %s: n=%-4s threads=%-3s wall=%8.1f ms  full=%s  pts=%6.2f/25  verify=%s\n" \
     "$id" "$n" "$W" "$ms" "${FULL[$id]}" "$(echo "$s*25" | bc)" "$v"
done
