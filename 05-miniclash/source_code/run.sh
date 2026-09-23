#!/usr/bin/env bash

set -u

if [ "$#" -ne 1 ]; then
	echo "usage: $0 <tasks.txt>" >&2
	exit 2
fi

TASKS=$1
BIN=$(dirname "$0")/md5fastcoll

if [ ! -r "$TASKS" ]; then
	echo "$0: cannot read task file: $TASKS" >&2
	exit 1
fi

if [ ! -x "$BIN" ]; then
	echo "$0: $BIN not found or not executable; run 'make all' first" >&2
	exit 1
fi

status=0
n=0
running=0
workers=$(nproc 2>/dev/null || printf '1')
base_seed=$(date +%s)

# Fields are space-separated: <input_file> <output_file1> <output_file2>.
while read -r infile out1 out2 rest; do
	# Skip blank lines.
	[ -n "${infile:-}" ] || continue

	if [ -z "${out2:-}" ] || [ -n "${rest:-}" ]; then
		echo "$0: malformed line: $infile ${out1:-} ${out2:-} ${rest:-}" >&2
		status=1
		continue
	fi

	# Give simultaneous searches distinct, nonzero xorshift states.
	seed1=$(((base_seed + n * 2654435761 + $$) & 4294967295))
	seed2=$(((base_seed * 2246822519 + n * 3266489917 + $$) & 4294967295))
	[ "$seed1" -ne 0 ] || seed1=1
	[ "$seed2" -ne 0 ] || seed2=1
	"$BIN" -q --seed1 "$seed1" --seed2 "$seed2" -p "$infile" -o "$out1" "$out2" >/dev/null &
	n=$((n + 1))
	running=$((running + 1))
	if [ "$running" -ge "$workers" ]; then
		if ! wait -n; then
			echo "$0: a collision task failed" >&2
			status=1
		fi
		running=$((running - 1))
	fi
done < "$TASKS"

while [ "$running" -gt 0 ]; do
	if ! wait -n; then
		echo "$0: a collision task failed" >&2
		status=1
	fi
	running=$((running - 1))
done

echo "$0: generated $n collisions"
exit "$status"
