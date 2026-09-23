#!/bin/bash
# Local measurement helper (dev only; NOT part of the submission).
set -u
cd "$(dirname "$0")"
N=${1:-32}
SEED=${2:-1001}
W=${3:-$(nproc)}
echo "== build =="
rm -f *.o md5fastcoll run
make all 2>&1 | tail -5
ls -l md5fastcoll
echo "== cases =="
python3 ../utils/gencase.py "$N" "cases/bench$N" "$SEED"
echo "== single collision timing =="
for i in 1 2 3; do
  f=$(head -1 cases/bench$N/tasks.txt | awk '{print $1}')
  /usr/bin/time -f "single wall %e s" ./md5fastcoll -q --seed1 $((i*7919)) --seed2 $((i*104729)) -p "$f" -o /tmp/o1.bin /tmp/o2.bin 2>&1 | tail -1
done
echo "== full run (workers=$W) =="
/usr/bin/time -f "TOTAL wall %e s" ./run cases/bench$N/tasks.txt
echo "== verify =="
python3 ../utils/verify.py cases/bench$N/tasks.txt
