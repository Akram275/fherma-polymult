#!/usr/bin/env bash
# Builds and runs the submission inside FHERMA's challenge image, the way the
# platform does (fherma.toml: build command, then ./build/solution <point>),
# from a clean copy of only the files a submission carries.
#
#     ./docker_check.sh [gpu index]          # default GPU 0
#
# Needs the image (docker pull fherma/cupqc:0.6.0) and the conda env "fherma"
# on the host for the test-point generators and the verifier.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
image=fherma/cupqc:0.6.0
gpu="${1:-0}"
if [ "${CONDA_DEFAULT_ENV:-}" != fherma ]; then
    set +u   # conda's activate.d hooks read unset variables
    source /opt/anaconda/etc/profile.d/conda.sh && conda activate fherma
    set -u
fi
work="$(mktemp -d "${TMPDIR:-/tmp}/fherma-docker.XXXXXX")"
echo "work dir: $work"

# The submission: what the platform builds, nothing else.
mkdir -p "$work/src"
cp fherma.toml CMakeLists.txt main.cpp fherma.h solve.cu rns.cuh rns_kernels.cuh rns_plan.h "$work/src/"

# Points: a small exact one, the worst-case limb edges, and both benchmark
# points (Point #1, N = 65536 / W = 1747, decides the rank).
python3 oracle.py make "$work/p_n1024" 1024 868 3 >/dev/null
python3 make_big.py "$work/p_n4096_w1024" 4096 1024 3 >/dev/null
python3 make_big.py "$work/p_n32768" 32768 868 21 >/dev/null
python3 make_big.py "$work/p_n65536" 65536 1747 11 >/dev/null

run() {   # command inside the image, as the host user, with the GPU
    docker run --rm --gpus "device=$gpu" --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -v "$work:/work" -w /work/src "$image" bash -c "$1"
}

echo "== image"
run 'nvcc --version | tail -1; cmake --version | head -1; g++ --version | head -1; echo "CUPQC_SDK_DIR=${CUPQC_SDK_DIR:-<unset>}"; ls "${CUPQC_SDK_DIR:-/usr/local/cupqc-sdk}/lib" 2>&1 | head; nproc'

echo "== build (fherma.toml's command)"
run 'cmake -B build -DCMAKE_BUILD_TYPE=Release > /work/cmake.log 2>&1 && cmake --build build -j >> /work/cmake.log 2>&1; rc=$?; grep -iE "warning|error" /work/cmake.log | head -20; exit $rc'
echo "build ok"

echo "== run + verify"
for p in p_n1024 p_n4096_w1024 p_n32768 p_n65536; do
    run "./build/solution /work/$p > /dev/null"
    printf '%-16s ' "$p"
    python3 oracle.py verify "$work/$p" | tail -1
done
for p in p_n65536 p_n32768; do
    python3 - "$work/$p" <<'EOF'
import json, statistics, sys
r = json.load(open(sys.argv[1] + "/out/results.json"))
t = [c["seconds"] for c in r["cases"] if c["status"] == "ok"]
print("%s in the image: init %.1f ms, median %.1f us, min %.1f us (%d cases)" % (
    sys.argv[1].rsplit("/", 1)[1], r["init_s"] * 1e3, statistics.median(t) * 1e6, min(t) * 1e6, len(t)))
EOF
done
