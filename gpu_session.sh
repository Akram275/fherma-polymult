#!/usr/bin/env bash
# Everything worth doing in one session on a borrowed GPU, logged to
# gpu_session.log. Needs nvcc, cmake, python3 with numpy and gmpy2.
#
#     ./gpu_session.sh            # full run, ~10-15 minutes
#
# 1. the machine: GPU, PCIe link, driver
# 2. build (native arch)
# 3. the transfer floor (bench_copy)
# 4. compute-sanitizer memcheck + racecheck on small points
# 5. the correctness matrix (run_tests.sh) on the GPU build
# 6. timing at the challenge point for each copy mode and pipeline depth,
#    plus one profiled run
set -uo pipefail
# On rst-goose the toolchain lives in the conda env "fherma" (see HANDOVER.md).
if [ "${CONDA_DEFAULT_ENV:-}" != fherma ] && [ -f /opt/anaconda/etc/profile.d/conda.sh ]; then
    set +u   # conda's activate.d hooks read unset variables
    source /opt/anaconda/etc/profile.d/conda.sh && conda activate fherma
    set -u
fi
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
exec > >(tee gpu_session.log) 2>&1
work="${TMPDIR:-/tmp}/fherma-polymul-gpu"
mkdir -p "$work"
section() { printf '\n==== %s ====\n' "$*"; }

section machine
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,name,driver_version,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,memory.total --format=csv
nvcc --version | tail -1
nproc; grep -m1 'model name' /proc/cpuinfo

section build
./build.sh

section transfer floor
build/bench_copy

section sanitizers
python3 oracle.py make "$work/san" 1024 868 1 >/dev/null
python3 make_big.py "$work/san4k" 4096 868 1 >/dev/null
for tool in memcheck racecheck synccheck; do
    for d in "$work/san" "$work/san4k"; do
        echo "-- $tool $(basename "$d")"
        compute-sanitizer --tool $tool --error-exitcode 9 build/solution "$d" 2>&1 | tail -2
    done
done

section correctness
./run_tests.sh build/solution

section timing
bench="$work/bench"
[ -f "$bench/manifest.json" ] || python3 make_big.py "$bench" 32768 868 21 >/dev/null
median() {
    python3 - "$1" <<'EOF'
import json, sys, statistics
r = json.load(open(sys.argv[1] + "/out/results.json"))
t = [c["seconds"] for c in r["cases"] if c["status"] == "ok"]
print("init %6.1f ms   median %7.1f us   min %7.1f us   (%d cases)" % (
    r["init_s"] * 1e3, statistics.median(t) * 1e6, min(t) * 1e6, len(t)))
EOF
}
for in_mode in staged register; do
    for out_mode in staged register; do
        for chunks in 1 4 8 16; do
            printf '%-8s %-8s chunks=%-3s ' "$in_mode" "$out_mode" "$chunks"
            if FHERMA_IN=$in_mode FHERMA_OUT=$out_mode FHERMA_CHUNKS=$chunks build/solution "$bench" >/dev/null 2>&1; then
                median "$bench"
            else
                echo "FAILED"
            fi
        done
    done
done
python3 oracle.py verify "$bench" | tail -1

section profile
FHERMA_PROFILE=1 build/solution "$bench" 2>&1 >/dev/null | tail -5
if command -v nsys >/dev/null; then
    nsys profile -o "$work/polymul" --force-overwrite true build/solution "$bench" >/dev/null 2>&1 \
        && nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$work/polymul.nsys-rep" 2>/dev/null | head -40
fi

section done
echo "log: $here/gpu_session.log"
