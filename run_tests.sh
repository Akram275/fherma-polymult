#!/usr/bin/env bash
# Correctness matrix for a solution binary against the Python oracles.
#
#     ./run_tests.sh [solution binary]        default: ./solution_cpu
#
# Points cover every NTT geometry (single pass for N <= 2048, and each column /
# row split for N = 2^12 .. 2^17), every limb edge (L = 1, a q that fills its
# limbs, a q one bit into a new limb, the widest L = 64), the spec's listed points
# (N64-W32, N1024-W512, N8192-W256, N32768-W868, N65536-W1747) and N = 2^17, the
# first N whose primes are not the N <= 2^15 set.
# make_big.py puts the all-(q-1) worst case in case 0 of each large point.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bin="$(realpath "${1:-$here/solution_cpu}")"
work="${TMPDIR:-/tmp}/fherma-polymul-tests"
rm -rf "$work" && mkdir -p "$work"

small=("4 30" "8 60" "16 64" "64 32" "64 100" "256 128" "512 250" "1024 512" "1024 868"
       "2048 868" "256 1747" "128 2048")
big=("2048 129" "4096 868" "8192 256" "8192 868" "16384 868" "4096 1024" "32768 868"
     "65536 1747" "131072 512")

fail=0
run() {  # generator N W
    local gen=$1 N=$2 W=$3 dir="$work/$1_n$2_w$3"
    if [ "$gen" = oracle.py ]; then
        python3 "$here/oracle.py" make "$dir" "$N" "$W" 3 >/dev/null
    else
        python3 "$here/make_big.py" "$dir" "$N" "$W" 3 >/dev/null
    fi
    if "$bin" "$dir" >/dev/null && python3 "$here/oracle.py" verify "$dir" >"$dir/verify.txt"; then
        printf '  PASS  N=%-6s W=%-5s\n' "$N" "$W"
    else
        printf '  FAIL  N=%-6s W=%-5s (see %s)\n' "$N" "$W" "$dir"
        grep -v ': OK' "$dir/verify.txt" | head -5 || true
        fail=1
    fi
}

echo "$bin"
for p in "${small[@]}"; do run oracle.py $p; done
for p in "${big[@]}";   do run make_big.py $p; done
[ $fail = 0 ] && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }
