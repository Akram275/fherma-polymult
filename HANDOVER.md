# Handover: fherma-polymul on rst-goose

For the Claude Code session starting on the GPU server. Everything up to here
was done on a laptop VM with **no GPU**: the code is written and its math and
schedule are verified on the CPU, but **no kernel in this repo has ever run on a
GPU**. Your first job is to make it run, then make it fast.

## This machine (rst-goose.telecom-sudparis.eu)

Checked 2026-10-01:

- **2x NVIDIA L40S** (sm_89, 46 GB, PCIe Gen4 x16), driver 580.159.03, which is the
  challenge's reference GPU. Ubuntu 24.04, 96 cores. Home is on NFS
  (`/netfs/tsp/student/2022/abendoukha`).
- **Shared server.** GPU 0 was busy with other users' Python jobs; GPU 1 was idle.
  Use `export CUDA_VISIBLE_DEVICES=1` (check `nvidia-smi` first) and expect timing
  noise if anyone else is on the card.
- **No system CUDA toolkit.** The toolchain is the conda env `fherma` (created for
  this project: CUDA 12.8 nvcc + compute-sanitizer, cmake 4.4, gcc 13, numpy, gmpy2):
  ```bash
  source /opt/anaconda/etc/profile.d/conda.sh && conda activate fherma
  ```
  `gpu_session.sh` activates it by itself. No `nsys`, `ncu` or `cuobjdump` is
  installed; add `nsight-systems`/`nsight-compute` to the env if needed. Docker is available too.
- **Build with `./build.sh` (`./build.sh clean` after a toolchain change).** Since the
  Ubuntu 26.04 upgrade (seen 2026-10-13) the system gcc is 15, too new for nvcc 12.8,
  so everything builds with the env's gcc 13.4. That compiler's sysroot `crt1.o`
  carries a false x86-64-v3 (AVX2) property note; this CPU is x86-64-v2 (QEMU, no
  AVX), so glibc refuses to start its binaries ("CPU ISA level is lower than
  required"). `build.sh` strips the note with `objcopy` after linking. Local only:
  `CMakeLists.txt` is untouched for FHERMA's image.

## The challenge

FHERMA "Polynomial Multiplication (GPU-Accelerated)", spec
`polynomial-multiplication/negacyclic@1.0.0`
(https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025).

- Compute `c = a*b mod (X^N + 1, q)`, exactly, coefficients canonical in [0, q).
- Benchmark point: **N = 32768, W = 868, L = 28** u32 limbs per coefficient,
  q = largest prime < 2^W with q = 1 (mod 2N). Coefficients are L little-endian u32.
- Score: **median wall-clock time of `fherma_run`**, lowest wins. Wrong answers are
  not ranked. Copies to and from the GPU are inside the timed call (see `main.cpp`).
- Prize $5,000. **Deadline 29 Nov 2026 00:00 UTC.**
- Leader on 2026-09-24: **260 us** (36 submissions).
- Reference hardware per `fherma.toml`: L40S (sm_89, PCIe Gen4). The CMake comment
  also mentions an RTX PRO 6000. Not confirmed.

## Leaderboard and rules as of 2026-10-14 (read this first)

- **A second benchmark point was added on 2026-09-28: N = 65536, W = 1747, L = 55.
  Rank is decided at that point (Point #1)**; N = 32768, W = 868 is now Point #2, shown
  beside it. An entry with no Point #1 result stands behind every entry that has one.
  "N, W, L and q are parameters, not constants"; the solution must handle "the full
  valid input domain" of the spec. **Our build throws in fherma_init at Point #1**
  ("N must be a power of two in [2, 2^15]"): primes are 1 mod 2^16, kMaxL = 32,
  kMaxK = 80, all too small for it.
- Runner on every ranked row: **fherma-gpu-rtx6000** (RTX PRO 6000, not the L40S this
  doc assumed; sm_120, presumably PCIe Gen5), image digest sha256:17b7ae... = our
  pulled fherma/cupqc:0.6.0. 20 random seeds per point, exact results only.
- Provisional leaderboard (61 submitted, 12 ranked): 1 @artemiy-altamare 870 us /
  294 us; 2 @haozhe-sun 885 us / -; 3 @asimandia 987 us / 250 us; 4 @tekkac 1.13 ms /
  293 us; 5 @megan-park 1.99 ms / 945 us; 6 @ben-shumeyko 3.56 ms / 1.24 ms; then
  Point #2 only: @xander-pottier 304 us, @ilya-usmanov (cuPQC) 378 us, @damir-salakhov
  435 us, @engelailabs 1.27 ms, @kenny-leung (cuPQC 0.6.0) 1.8 ms, @g-arakelov 816 ms.
- Deadline: **closes 29 Nov 2026 00:00** (timezone not stated; 46 days left on
  2026-10-14). $5,000 to the fastest correct submission at close. Submissions may be
  re-run on more seeds and points; multiple submissions allowed.
- Rules say nothing on cuPQC being required (title: "...with NVIDIA cuPQC"; two
  ranked entries use it, most do not), nor on threads, busy-waiting or mallopt.
  Timing: from a, b available to c available; setup must not depend on a, b.

## Point #1 support (2026-10-14, later)

Done: primes step by max(2N, 2^16) (same set as before for N <= 2^15), N up to
2^22 (`kMaxLogN`, the NTT row pass needs rows <= 2048), `kMaxL` 64 (W <= 2048),
`kMaxK` 160, CRT shared memory dynamic (`rns::crt_smem_bytes`, opt-in past 48 KB,
clear error if the device has less). `run_tests.sh` now has 21 points incl. the
spec's five, W = 1747 and W = 2048 against the exact oracle, N = 65536/W = 1747 and
N = 2^17. **All pass on CPU and GPU; sanitizers clean incl. L = 55 and 64.**
ptxas: no spills at L = 55; L >= 62 gets a 264-byte stack frame in k_crt (no spills).
Backup: `backup/2026-10-14-pre-point1/`.

Point #1 on the L40S (GPU 0): **3.09 ms median, GPU span 2.6 ms**; Point #2 unchanged
(704 us). Timeline: 4 x 7.2 MB H2D (0-1196 us), then **707 us of kernels on the
critical path** (decompose 75, cols 65, rows_mul 335, cols inv 77, first CRT chunk
155), then D2H overlapped with the other CRT chunks (237 + 130). Host overhead is
~480 us (wall 3.09 ms vs span 2.6 ms): not yet examined.

Next, in order of critical-path time at Point #1:
1. `k_ntt_rows_mul` 335 us (31 at Point #2): the twiddle tables are now
   K x N x 16 B ~ 120 MB per direction pair, beyond L2, so they stream from DRAM.
   Plan: twiddles from small per-prime tables (psi^e = psi_hi^(e >> s) * psi_lo^(e & mask),
   ~2 sqrt(N) entries per prime, L2/L1-resident) in Montgomery form, at the price of
   one extra modular multiply per butterfly.
2. CRT ~520 us over 3 chunks: 52 KB of shared memory per block -> one 4-warp block
   per SM. Plan: split each coefficient's work over limb ranges instead of prime
   groups (accumulators by column, y read straight from global, coalesced), ~15 KB.
3. Host side at Point #1 (~480 us over the GPU span).

## Point #1 fixes (2026-10-14, later still)

1. **Twiddles generated from small tables** (`rns::GenTwiddles`): psi^e = hi[e >> s] * lo[e & mask]
   in Montgomery form, s = ceil(logN/2), ~0.5 MB for all primes. The row passes use it when
   the full tables (16 K N bytes) exceed `kGenTwiddleBytes` = 64 MB (auto: Point #1 yes,
   Point #2 no; `FHERMA_TWIDDLES=table|gen|auto`); the column pass always reads a small head
   table (`col_psi`, indices < N >> s). Point #1: rows_mul 336 -> 132 us, rows_fwd 120 -> 64,
   cols 65 -> 55; Point #2 unchanged (gen would cost 31 -> 38 us there). Full tables are only
   built for the table path (saves init time and 121 MB of device memory at Point #1).
2. **CRT split over limb ranges** (`crt_columns` / `crt_combine` in rns_kernels.cuh): warp g
   owns limbs [g span, g span + span), one coefficient per thread, prime loop unrolled by 4,
   y read straight from global. Shared memory ~10 KB at L = 55 (was 52 KB, independent of K
   now). Point #1: ~170 -> 50 us per chunk; Point #2: 11.9 -> 13.1 us (first chunk only on
   the critical path). Tried: 2 coefficients per thread (31.5 / 53.6 us), unroll 8 (worse).
3. **Host overhead at Point #1**: input copies saturate at ~29 GB/s (~1 ms for 28.8 MB) from
   16 threads on; more threads, uneven chunks and a later zero-fill did not help (the later
   zero-fill made it worse: 2688 -> 3082 us). Left as is.

Result at Point #1 (native, GPU 0): **wall 2.55-2.67 ms (was 3.09), GPU span 2.18 ms (was
2.60)**; kernels between the last H2D and the first D2H: 370 us (was 707). Point #2 unchanged
(~700 us). All 21 test points pass on CPU and GPU with both twiddle paths; sanitizers clean.
Backups: `backup/2026-10-14-pre-twgen/`, `backup/2026-10-14-pre-crt-limbs/`.

**Container finding**: in `fherma/cupqc:0.6.0` Point #1 runs ~0.3-1 ms slower than natively
(3.0-3.9 ms vs 2.6-3.1, noisy). The helper's 14.4 MB zero-fill of the output takes ~3 ms
there instead of ~1.3 ms: the image's glibc 2.39 zero-fills with `rep stosb` above 2 KB, and
this VM's CPU has no ERMS/FSRM, so that is slow; the host's glibc 2.43 uses non-temporal
stores instead. Page faults are identical. Probably a VM artifact (a modern host CPU has
ERMS), not verified for FHERMA's box. Possible fix (needs the user's OK, it moves work out
of the timed window): zero-fill the next call's output vector in the background between calls.

## Getting closer to the transfer floor (2026-10-14, evening)

**New default copy mode: `hmm`** (FHERMA_IN/OUT, falls back to staged without pageable
memory access). The decompose kernel reads the caller's input vectors directly and the
CRT writes the answer's vector directly, through HMM (this box: "Addressing Mode: HMM",
open kernel module), with cudaMemAdvise preferred=host + accessedBy=GPU (cached per
address). Benchmark, 14.4 MB: pinned DMA 593 us (24.3 GB/s); HMM read with that advice
684 us (21.1); without advice 32 ms (pages migrate); HMM write 732 us (19.7). With it:
no host staging copies at all, host overhead ~60 us (was ~400-500 at Point #1).
Also: a's forward NTT on its own low-priority stream (`ntt_a`), comp at high priority
(b's decompose *is* b's transfer in hmm mode and must not wait); decompose loads 16 bytes
at a time for full blocks; in hmm mode one input and one output chunk (each chunk
boundary leaves a kernel tail idling the link).

Native, interleaved (L40S, GPU 0): **Point #1 2616 -> ~2300 us (GPU span 2242), Point #2
684 -> ~605-624 us.** Call at Point #1: input (decompose over PCIe) ~1440 us, kernels 244 us,
output (CRT over PCIe) ~656 us.

Tried, not adopted:
- Tail chunk shapes in hmm mode: more chunks are worse (8 input chunks: span 2705).
- Pinned arena behind a replaced global operator new (EXPERIMENT ONLY, scratch build,
  never in the repo): deadlocks inside the CUDA runtime (main thread on a futex at the
  first case), even with a re-entrancy guard and enabling it after init. Abandoned;
  it also needs FHERMA's OK. Expected gain over hmm only the DMA-vs-HMM rate (~250-300 us
  at Point #1).
- Cluster-based single-pass NTT: needs sm_90+, cannot run on the L40S. Not attempted.
- Preview run on FHERMA: needs pushing to a repo and registering the commit from the
  user's account; no CLI here. Not done.

**Container caveat (bigger now):** in fherma/cupqc:0.6.0 on this VM, every mode is
dominated by waiting for the output vector's zero-fill (1.5-2.9 ms, load-dependent;
glibc 2.39 memset), and hmm out mode must wait for it before the CRT starts. Measured in
the image (3 rounds): staged 3.1-3.7 ms, hmm/hmm 2.5-3.6, hmm/staged 2.9-3.4. Natively the
fill takes ~1.3 ms and is hidden. Likely specific to this VM (old CPU model) but unverified
for FHERMA's box. The only full fix found is preparing the next output vector between calls
(operand-independent, but moves work out of the timed window): user's decision.

## Improvement round (2026-10-14, late)

1. **Huge pages for HMM buffers: no gain.** With the advice calls out of the timing (as the
   real code caches them), a plain streaming kernel already reads pageable memory at 23.9 GB/s
   and writes at 25 GB/s over HMM with 4 KB pages; MADV_COLLAPSE (12 of 14 MB became 2 MB
   pages) changed nothing. The HMM-vs-DMA gap is in our kernels (decompose ~20 GB/s, CRT ~22).
2. **`stream` copy mode (opt-in, not default)**: a lean k_stream kernel pulls inputs / pushes
   outputs between the caller's memory and device memory at ~24-26 GB/s, decompose and CRT
   on device memory. Slower overall (best GPU span 2452 vs 2245 us at Point #1): the pulls
   starve the decompose of SMs (902 us instead of 72 for the first chunk). hmm-in +
   stream-out: 2364-2385 vs 2245. Kept as `FHERMA_IN/OUT=stream` for other hardware.
3. **Decompose chunks alternate between two equal-priority streams** (`comp2`) in hmm mode:
   2 chunks now cost the same as 1 (2235 vs 2245 span; was 2323), 8 chunks 2372 (was 2705).
4. **Column-group streaming (decompose + column pass of b by column group, inverse column
   pass + CRT by group): not built.** It needs ~8 groups per operand, and 8 decompose chunks
   still cost ~120 us more than 1 (small kernels keep too few PCIe reads in flight), against
   ~50 us of column pass it would hide.
5. **Init-time autotuning of the copy mode** (`configure()`, unless FHERMA_IN/OUT are set or
   FHERMA_AUTOTUNE=0): times hmm/hmm, staged/staged and hmm/staged on dummy operands, 7 runs
   each, evicting the caches between runs (else it underrated the zero fill), and leaves the
   default (hmm/hmm) unless another is >= 5 % faster. FHERMA_AUTOTUNE_LOG=1 prints the timings.
   Adds ~50-100 ms to init (untimed). Natively it picks hmm/hmm at both points.

Results now: **native Point #1 ~2.29-2.31 ms, Point #2 ~605 us**. In fherma/cupqc:0.6.0 the
numbers swing with the machine's load (another user's GPU job and a database on the host):
Point #1 2.7-4.0 ms, Point #2 0.61-0.89 ms; the zero fill of the output is still the
dominant cost there (see the container caveat above).

## Safety checks and the last speed attempt (2026-10-14, night)

- **Compile check for the reference GPU (sm_120)**: builds and links with no errors or
  warnings. k_crt<55> (Point #1, 158 registers) and k_crt<28> don't spill; k_crt for
  L = 33 and 43-48 spill 16-64 bytes on sm_120 (speed only, those L).
- **Correctness fuzz (`fuzz.py`, new)**: `python3 fuzz.py <work dir> [random points] [seed]`.
  61 points (edge widths at every 32-bit boundary up to 2048, N = 2 ... 2^22, the spec's five,
  30 random), six adversarial input patterns each (all q-1, random, zero, x^(N-1)*x^(N-1)
  wrap, identity, alternating), 8 GPU configs (default/autotune, staged, hmm, stream,
  mapped, register, twiddles gen/table) + the CPU build for N <= 2^16; GMP reference
  cross-checked against the O(N^2) product for N <= 256. **0 failures (508 s).** Out of range
  (N = 2^23, W = 2049) refused cleanly with a message. Rerun it after any change.
- **Software-pipelined remote decompose** (persistent blocks, double-buffered tiles,
  cp.async): correct (matrix + race/sync/memcheck with a 2-block grid to force the buffer
  swap) but **slower**: 895 us per Point #1 operand against 667 us (56 KB of shared memory ->
  one block per SM; 4-byte cp.async per word). Not adopted; code kept in
  `backup/2026-10-14-experiments/solve_dec_pipe.cu` (the pinned-arena experiment is there too).

## Checked in the challenge image (2026-10-14)

`./docker_check.sh [gpu]` copies only the submission files (fherma.toml,
CMakeLists.txt, main.cpp, fherma.h, solve.cu, rns*.cuh, rns_plan.h) to a clean
dir, builds with fherma.toml's exact command inside `fherma/cupqc:0.6.0` (the only
tag on Docker Hub, pulled here, 9.8 GB), runs `./build/solution` on three points
and verifies them on the host. Result: **0 build warnings; N=1024, N=4096/W=1024
and the challenge point (21 cases) all VERIFY: PASS; median 702.9 us**, same as
the native build. Image: Ubuntu 24.04.1, CUDA 12.8.93, gcc 13.3, cmake 3.28;
CUPQC_SDK_DIR=/usr/local/cupqc-sdk with libcupqc-bigint/-ntt/-hash/-pk.a.
libcupqc-bigint.a gets linked (CMakeLists does that when it exists) but is unused.
Note the image ships **cuPQC NTT** (`libcupqc-ntt.a`): if cuPQC turns out to be
mandatory, its NTT is the natural piece to adopt.

## Open questions for the user (do not decide these alone)

1. **Is cuPQC mandatory?** The challenge page says "using NVIDIA cuPQC" but also
   calls it "recommended". `solve.cu` uses none of it. Must be settled before submitting.
2. **`mallopt` in `fherma_init`** makes large freed vectors stay on the heap
   (avoids page faults on the per-call output vector). It is process-wide; the user
   has not yet said whether they are comfortable with it.
3. 11 MB cross PCIe per call, which on Gen4 is ~450 us at best, yet the leader is at
   260 us. Either the reference box is faster (Gen5) or the timing differs. The
   numbers from `bench_copy` here will inform this.

## Algorithm (RNS + negacyclic NTT + CRT)

`rns_ref.py` is the NumPy blueprint; the C++/CUDA follows it.

1. **Primes:** K word-size primes p_j < 2^31, p_j = 1 (mod 2^16) (one set serves all
   N <= 2^15), product P > 2^20 * 2 * N * q^2. At the challenge point **K = 58**.
2. **Decompose:** each coefficient's L limbs -> residue mod each p_j (Shoup multiplies).
3. **NTT:** per prime, forward negacyclic NTT of a and b (Cooley-Tukey, twiddles
   psi^bitrev(i), so no pre/post twist), pointwise Montgomery product times a constant
   `N^-1 (P/p_j)^-1 2^32`, inverse NTT (Gentleman-Sande). The output is y_j directly.
4. **CRT:** v = round(sum y_j / p_j) in double (exact thanks to the 2^-20 margin), then
   `c = sum_j y_j * (P/p_j mod q) + (-v P mod q)`, reduced mod q with a double-estimated
   quotient plus at most 2 subtractions.

## Files

| File | What |
|---|---|
| `fherma.h`, `main.cpp`, `fherma.toml` | **Generated by the platform, do not edit.** Harness: reads a point dir, times `fherma_run` per case, writes `out/`. |
| `rns.cuh` | Per-element arithmetic, `__host__ __device__`: mod-p ops, decompose, butterflies, multi-limb reduce, `crt_one<L>`, `with_L` dispatch. |
| `rns_plan.h` | Host setup run in `fherma_init`: primes, twiddle tables, CRT constants. |
| `rns_kernels.cuh` | **The GPU schedule**: tile geometry, per-phase steps and index maps for every kernel, shared by GPU and CPU. |
| `solve.cu` | **The answer.** Kernels (thin wrappers over `rns_kernels.cuh`) + host pipeline (streams, chunked copies, pinned staging or `cudaHostRegister`). |
| `solve_cpu.cpp` | Runs the *same schedule* on CPU, block by block and phase by phase, and checks that no NTT stage touches an element twice. Built by `build.sh`. |
| `run_tests.sh [binary]` | Correctness matrix: 14 points (every NTT geometry, limb edge cases L = 1..32, the challenge point with an all-(q-1) worst case). Default binary is `./solution_cpu`. ~1 min. |
| `gpu_session.sh` | **Run this first.** Machine info, build, `bench_copy`, compute-sanitizer (memcheck/racecheck/synccheck), `run_tests.sh build/solution`, timing sweep over copy modes and chunk counts, profile. Logs to `gpu_session.log`. |
| `bench_copy.cu` | Transfer floor: 11 MB in and out, comparing pageable, pinned, staged and registered copies. No cuPQC. |
| `oracle.py` | Test-point generator (`make <dir> N W [cases]`) + verifier (`verify <dir>`). O(N^2) expected answers, so small N only. |
| `make_big.py` | Same layout, expected answer via GMP Kronecker substitution. Fast at N = 32768 (~3 s). Case 0 is the all-(q-1) worst case. |
| `rns_ref.py` | NumPy reference model of the algorithm (`params N W`, `solve <dir>`). |
| `baseline_schoolbook.cu` | Original O(N^2) cuPQC BigInt baseline. Not built. Needs cuPQC. |
| `CMakeLists.txt` | Builds `main.cpp + solve.cu`; `-arch=native` by default (`-DFHERMA_CUDA_ARCH=sm_89` to override); links cuPQC only if `$CUPQC_SDK_DIR/lib/libcupqc-bigint.a` exists. |

## GPU pipeline (`solve.cu`)

Kernels per multiplication, all on stream `comp`:

| Kernel | Grid | Does |
|---|---|---|
| `k_decompose` x (2 * chunks) | N/32 blocks per operand | limbs -> K x N residues, tile padded to L+1 |
| `k_ntt_cols<fwd>` on A, `k_ntt_rows_fwd` on A | (S/C, K), (N/T, K) | a's whole forward transform, enqueued right after a's decompose so it runs under b's H2D |
| `k_ntt_cols<fwd>` on B | (S/C, K) | NTT stages with t >= S on B |
| `k_ntt_rows_mul<true>` | (N/T, K) | stages t < S forward on B, pointwise, inverse on A |
| `k_ntt_cols<inv>` | (S/C, K) | inverse stages t >= S on A |
| `k_crt<L>` x chunks | 32 coefficients x 4 prime groups = 128 threads per block | Y tile to smem, partial sums per prime group, combine + reduce per coefficient, coalesced store |

NTT split at N = 32768: S = 256 (row length), R = 128, tile T = 2048 elements,
C = 16 columns per column-pass block. H2D copies (stream `h2d`) overlap decompose;
D2H copies (stream `d2h`) overlap CRT, chunk by chunk, chained by events.
`fherma_init` uploads the tables and does one untimed warm-up run.

Environment knobs (read in `fherma_init`):
`FHERMA_IN=staged|register`, `FHERMA_OUT=staged|register` (default staged),
`FHERMA_IN|OUT=mapped` (decompose reads / CRT writes the pinned host buffers
directly over PCIe, no copies or events per chunk), `FHERMA_CHUNKS=n` (input side,
default 2), `FHERMA_OUT_CHUNKS=n` (output side, default = FHERMA_CHUNKS),
`FHERMA_THREADS=n` (host copy threads, default min(16, cores/2)),
`FHERMA_SPIN_MS=n` (copy threads keep spinning n ms after a call, default 20),
`FHERMA_PROFILE=1` (per-phase GPU times and the host timeline on stderr).

Host side (staged, the default): a `CopyPool` of std::threads copies each input
chunk into pinned memory (64 KB pieces claimed from a shared counter, the caller
included), then the chunk is sent. Once all GPU work is enqueued, the calling
thread zero-fills the output vector (`resize`) while the GPU runs, and copies
each output chunk out of pinned memory as it lands. Workers spin during a call
and sleep between calls. `CMakeLists.txt` links `Threads::Threads` for this.

ptxas (sm_89, from nvcc 12.2): every kernel has 0 stack and 0 spills; `k_crt<28>`
uses 80 registers, the NTT kernels 22-30 registers with 16 KB smem.

## Status

**2026-10-13: correct on the GPU.** `run_tests.sh` passes all 14 points with both
`solution_cpu` and `build/solution`; memcheck, racecheck and synccheck are clean
(gpu_session.log, 2026-10-01). The verifier was confirmed to catch a single flipped bit.

Measured on GPU 1 (idle), N = 32768, W = 868, 21 cases:

| | |
|---|---|
| PCIe floor (`bench_copy`, pinned DMA, 11 MB) | ~450 us (24.5 GB/s, Gen4 x16) |
| Pageable / staged / registered copies alone | ~2.0-2.3 / ~2.5-2.8 / ~2.6-3.2 ms |
| End to end before the host rework, best (register/staged/4) | 2104 us median |
| **End to end now, defaults (staged, 16 threads, 2 chunks), GPU 0** | **~860-900 us median** |
| GPU phases (register/staged/8) | in+decompose ~350, NTT ~78, CRT ~92 (38 at 1 chunk), out-tail ~620 us |

The wall clock is dominated by **host-side work**, not kernels: the GPU-measured
window is ~1.1 ms of the ~2.1 ms call (the rest is cudaHostRegister/unregister and
output-vector setup), and the out-tail is mostly the CPU copy out of the pinned
buffer. This VM's host memory bandwidth is poor (~4 GB/s memcpy), so the reference
box may differ a lot.

CRT rework (2026-10-13): was one thread per coefficient, 64-thread blocks, Y read
from global inside the loops; each chunk launch cost ~38 us however small (latency
bound), 308 us at 8 chunks. Now Y is staged in smem and the K primes are split
over 4 threads per coefficient: 92 us at 8 chunks, 38 us at 1. End to end that was
worth only ~1-2 % here, because CRT is not on the critical path on this machine.
Pre-change files are in `backup/2026-10-13-pre-crt/`.

Host rework (2026-10-13, pre-change files in `backup/2026-10-13-pre-host/`):
- On this VM, one operand (3.67 MB): cudaHostRegister+unregister ~400 us, one-thread
  memcpy ~300 us, 16-thread memcpy ~13-60 us, zero-filling a new vector ~240 us,
  pinned DMA ~150 us, pageable cudaMemcpy ~357 us.
- Waking sleeping workers costs ~150-200 us at the start of each call (VM wake-up
  latency). With workers kept spinning, `send` drops from ~400 to ~180 us. main.cpp
  leaves ~13 ms between calls, so hiding this would mean spinning 15 threads for
  ~13 ms per call. **Not done: user's call.**
- More chunks are slower here (8 is worse than 2). It is NOT the CUDA API calls,
  as first thought: with mapped output a chunk is one launch and one event and it
  still hurts. The cost is per pool copy (fixed overhead of each parallel memcpy).

2026-10-14, interleaved A/B on GPU 0 (5 rounds x 21 cases, median of medians):

| config | median |
|---|---|
| staged/staged, chunks 2/2 (default) | 838-860 us |
| mapped in, staged out | 818-840 us |
| mapped/mapped, chunks 2/4 | 823 us |
| staged + `FHERMA_SPIN_MS=20` | **760 us** |
| mapped in + `FHERMA_SPIN_MS=20` | 763 us |

- Zero-copy (`mapped`) is worth ~2 % here, inside the run-to-run noise
  (790-940 us). Kept as an opt-in; default stays staged.
- Spinning between calls saves ~80-100 us (~11 %), consistently. Default 20 ms
  since 2026-10-14 (user's go-ahead); 15 threads busy-wait through main.cpp's
  ~13 ms gap. `FHERMA_SPIN_MS=0` turns it off.
- Tried and rejected: spinning without `pause` (in case KVM pause-loop exiting
  descheduled the workers): no difference.

2026-10-14, continued. `nsys` is now in the env (`nsight-systems-cli`). The
reliable measure here is the **GPU span per call from nsys** (wall-clock A/B is
swamped by other users' jobs: ±50 us, outliers to 1.3 ms); see "Measuring" below.

| change | effect |
|---|---|
| `FHERMA_SPIN_MS` default 20 (user's go-ahead) | wall ~860 -> ~760 us |
| `Helper` thread zero-fills the output from call start | wall 736 -> 708 us (zero fill was on the critical path) |
| output chunks 1,2,1 (`FHERMA_C_CHUNKS`, new default) | wall 715 -> 700 us (10 interleaved rounds) |
| a's forward NTT under b's H2D (`k_ntt_rows_fwd`) | GPU span 622 -> 591 us |
| uneven input chunks (`FHERMA_A/B_CHUNKS`) | no gain; B=3,1 / 7,1 worse (625 / 647 us span). Not default. |

GPU timeline of one call now (us): 4 x H2D 1.84 MB back to back (0-326), a's
decompose + forward NTT hidden under them; after the last H2D: decompose 20, cols(B)
16, rows_mul 32, cols inv 15, first CRT chunk 12; then D2H 0.92 + 1.84 + 0.92 MB
back to back (434-582). Transfers alone are ~470 us of the ~585; what is left on
the critical path is ~95 us of kernels plus the host edges (~25-100 us).

Kernel work (2026-10-14, kernel times from nsys, which ignore the machine's noise;
`cuobjdump`/`nvdisasm` are now in the env for SASS):

| kernel | before | now | what changed |
|---|---|---|---|
| `k_decompose` (half an operand) | 20.6 us | **12.2 us** | `L` templated; lazy reduction (terms < 2p summed in 64 bits, one reduction); 4 coefficients per thread so each table entry serves 4 terms; 512 threads per block. SASS showed 2 global table loads per term were the bottleneck, not the arithmetic. |
| `k_ntt_rows_mul` | 31.8 us | 30.8 us | twiddle + Shoup companion interleaved (`rns::Twiddle`, one 8-byte load) |
| `k_ntt_cols` | 15.9 us | 15.3 us | same |

NTT anatomy (rows_mul / cols, by elimination): data movement ~14 / ~10 us,
butterfly arithmetic ~8 / ~2.5 us, twiddles ~9 / ~3 us. The twiddle cost is bytes,
not instructions: a row-pass block needs ~2048 distinct twiddles per transform, so
rows_mul reads ~30 MB of twiddles from L2. Montgomery-form twiddles would halve the
bytes but add ~1-2 IMAD per butterfly (estimated net gain 2-3 us); not done.
Critical path after the last H2D is now ~85 us of kernels (decompose 12, cols 15,
rows_mul 31, cols inv 15, first CRT chunk 12). Backups: `backup/2026-10-14-pre-decompose/`,
`backup/2026-10-14-pre-twiddle/`.

Measuring: `nsys profile -t cuda --sample=none --cpuctxsw=none -o X build/solution
<bench>`, `nsys export --type sqlite -o X.sqlite X.nsys-rep`, then split kernels +
memcpys into calls on >2 ms gaps (the scripts used are short; rewrite as needed).

## First steps on this machine

```bash
cd ~/fherma-polymul
source /opt/anaconda/etc/profile.d/conda.sh && conda activate fherma
nvidia-smi                                     # pick the idle GPU
export CUDA_VISIBLE_DEVICES=1
./build.sh                                     # all three binaries
./run_tests.sh && ./run_tests.sh build/solution
./gpu_session.sh                               # ~10-15 min, writes gpu_session.log
```

Then read `gpu_session.log` top to bottom:
- **sanitizers or correctness failing:** fix first. If the CPU build still passes
  `run_tests.sh` but the GPU fails, the bug is in `solve.cu` (launch/streams), not
  in the shared headers.
- **all passing:** compare the timing sweep with `bench_copy`'s floor and the
  `FHERMA_PROFILE` split to see where the time goes. Report the numbers to the user.

## Optimization backlog (once it runs, measure first)

- **Transfers** are the expected floor. Pick the best `FHERMA_IN`/`FHERMA_OUT`/chunk
  setting; consider multithreaded memcpy into pinned buffers; check whether output
  vector allocation (zero-fill) is significant.
- **Launch overhead:** ~30 launches/copies per call. Try CUDA Graphs (capture the
  device part; the host memcpys stay outside).
- **Twiddle traffic:** psi/ipsi plus Shoup companions are ~30 MB of reads per
  multiply. Options: Montgomery-form twiddles without companions, or on-the-fly
  twiddles from small tables.
- **CRT:** 58 x 28 multiply-accumulates per coefficient with 64-thread blocks; try
  several threads per coefficient, or 32-bit `madc` chains.
- **Decompose + NTT fusion**, and lazy reduction in the butterflies.

## Working rules

- After any change to `rns*.cuh`/`rns_plan.h`: rebuild **both** `solution_cpu` and
  `build/solution`, run `./run_tests.sh` and `./run_tests.sh build/solution`.
- Keep the schedule in `rns_kernels.cuh` shared, so the CPU run keeps testing what
  the GPU does. Kernels in `solve.cu` should stay thin.
- Timing: `python3 make_big.py /tmp/bench 32768 868 21`, run `build/solution /tmp/bench`,
  take the median of `out/results.json`. `gpu_session.sh` has a helper.
- Don't run `pkill -f <name>` from a shell whose command line contains `<name>`: it
  kills itself.
- Ask the user before submitting anything to FHERMA.
