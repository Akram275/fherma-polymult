#!/usr/bin/env bash
# Builds build/solution, build/bench_copy and solution_cpu on rst-goose.
#
#     ./build.sh            # incremental
#     ./build.sh clean      # wipe build/ first (after a toolchain change)
#
# Toolchain: the conda env "fherma" (nvcc 12.8 + gcc 13.4). The system gcc
# (15 since the Ubuntu 26.04 upgrade) is too new for nvcc 12.8. The env's
# sysroot crt1.o carries a GNU property note claiming x86-64-v3 (AVX2) without
# containing any such instruction, and rst-goose's CPU is x86-64-v2, so glibc
# refuses to start anything it links ("CPU ISA level is lower than required").
# The note is removed from each binary after the link. Local only: FHERMA
# builds CMakeLists.txt in its own image.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
if [ "${CONDA_DEFAULT_ENV:-}" != fherma ]; then
    set +u   # conda's activate.d hooks read unset variables
    source /opt/anaconda/etc/profile.d/conda.sh && conda activate fherma
    set -u
fi
[ "${1:-}" = clean ] && rm -rf build

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j 2>&1 | grep -E 'error|warning|Built target'
nvcc -O2 -std=c++17 -arch=native -o build/bench_copy bench_copy.cu
"${CXX:-g++}" -O2 -std=c++17 -fopenmp -o solution_cpu main.cpp solve_cpu.cpp

for b in build/solution build/bench_copy solution_cpu; do
    objcopy --remove-section=.note.gnu.property "$b"
done
