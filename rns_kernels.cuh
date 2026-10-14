// The GPU schedule for rns.cuh: how the work is cut into blocks, and what one
// thread does in each phase of each kernel.
//
// Every kernel in solve.cu is a sequence of phases separated by __syncthreads,
// and every phase is "for each item i in [0, count): step(i)" with the items
// dealt out to the block's threads. The step functions and the index maps they
// use live here, as __host__ __device__ code, so solve_cpu.cpp can run the same
// schedule block by block and phase by phase, and oracle.py then checks the
// tiling and indexing, not just the arithmetic.
//
// Kernels, per multiplication:
//
//   decompose   x2    limbs -> K x N residues        block: kDecTile coefficients
//   ntt_cols    fwd   A and B, stages t >= S         block: T elements, C columns
//   ntt_rows          A and B fwd stages t < S, pointwise, A inv stages t < S
//   ntt_cols    inv   A, stages t >= S
//   crt               K x N residues -> limbs         block: kCrtTile coefficients,
//                                                           kCrtGroups threads each
//
// The NTT is split at S = 2^ceil(logN / 2) (S = N for N <= 2048): the stages
// whose butterflies span t >= S pair elements in the same residue class mod S
// (a "column" of R = N / S elements, stride S), the stages with t < S pair
// elements inside one contiguous run of S (a "row"). Each pass keeps a tile of
// T = min(N, kNttTile) elements in shared memory for all of its stages.
#pragma once

#include "rns.cuh"

namespace rns {

constexpr uint32_t kNttTile = 2048;   // elements per NTT block (8 KB per operand)
constexpr uint32_t kDecPer = 4;                // coefficients per decompose thread
constexpr uint32_t kDecTile = 32 * kDecPer;    // coefficients per decompose block
constexpr uint32_t kCrtTile = 32;     // coefficients per CRT block (one warp)
constexpr uint32_t kCrtGroups = 4;    // prime groups: CRT block = kCrtTile * kCrtGroups threads

// Device-side views of the Plan tables (pointers into device memory on the GPU,
// into the Plan's vectors on the CPU).
struct Tables {
    int K, L;
    uint32_t N;
    const uint32_t* p;
    const uint32_t* pneg;
    const uint32_t* pow2;      // K x L
    const uint32_t* pow2_s;
    const Twiddle* psi;        // K x N
    const Twiddle* ipsi;
    const uint32_t* scale;     // K
    const uint32_t* scale_s;
};

struct NttGeom {
    uint32_t N, T;
    uint32_t logN, logS, logR, logC;   // S = row length, R = N / S, C = T / R
};

inline NttGeom make_geom(uint32_t logN) {
    NttGeom g{};
    g.logN = logN;
    g.N = 1u << logN;
    g.T = g.N < kNttTile ? g.N : kNttTile;
    g.logS = logN <= 11 ? logN : (logN + 1) / 2;
    g.logR = logN - g.logS;
    uint32_t logT = 0;
    while ((1u << logT) < g.T) ++logT;
    g.logC = logT - g.logR;            // columns per column-pass block (<= S)
    return g;
}

// One butterfly: tile[lo] and tile[hi], with twiddle index tw.
struct Pair {
    uint32_t lo, hi, tw;
};

// --- column pass -----------------------------------------------------------
// A block holds C adjacent columns r0 .. r0 + C - 1, all R rows each:
// tile[u * C + c] = x[r0 + c + u * S]. Grid: (S / C) x K.

RNS_HD uint32_t col_global(uint32_t i, uint32_t r0, const NttGeom& g) {
    return r0 + (i & ((1u << g.logC) - 1)) + ((i >> g.logC) << g.logS);
}

// Butterfly b in [0, T/2) of the stage whose global distance is t = 2^lt * S.
// Its twiddle index m + i, with m = N / (2t) and i = k / (2t), is the same for
// all C columns because r < S.
RNS_HD Pair col_pair(uint32_t b, uint32_t lt, const NttGeom& g) {
    const uint32_t c = b & ((1u << g.logC) - 1);
    const uint32_t bb = b >> g.logC;
    const uint32_t blk = bb >> lt;
    const uint32_t u = (blk << (lt + 1)) + (bb & ((1u << lt) - 1));
    const uint32_t lo = (u << g.logC) + c;
    return Pair{lo, lo + (1u << (lt + g.logC)), (1u << (g.logR - lt - 1)) + blk};
}

// --- row pass --------------------------------------------------------------
// A block holds T contiguous elements from base = blockIdx * T. Grid: (N / T) x K.

// Butterfly b in [0, T/2) of the stage with distance t = 2^lt < S.
RNS_HD Pair row_pair(uint32_t b, uint32_t lt, uint32_t base, const NttGeom& g) {
    const uint32_t blk = b >> lt;
    const uint32_t lo = (blk << (lt + 1)) + (b & ((1u << lt) - 1));
    return Pair{lo, lo + (1u << lt),
                (1u << (g.logN - lt - 1)) + (base >> (lt + 1)) + blk};
}

// Butterflies on a tile, in either pass.
RNS_HD void fwd_butterfly(uint32_t* tile, const Pair& pr, const Twiddle* tw, uint32_t p) {
    const Twiddle t = tw[pr.tw];
    ct_butterfly(tile[pr.lo], tile[pr.hi], t.w, t.ws, p);
}

RNS_HD void inv_butterfly(uint32_t* tile, const Pair& pr, const Twiddle* tw, uint32_t p) {
    const Twiddle t = tw[pr.tw];
    gs_butterfly(tile[pr.lo], tile[pr.hi], t.w, t.ws, p);
}

// Pointwise product for prime j, times N^-1 (P/p_j)^-1 so the inverse NTT
// ends in y_j directly.
RNS_HD uint32_t pointwise(uint32_t a, uint32_t b, const Tables& t, int j) {
    const uint32_t p = t.p[j];
    return mul_shoup(mont_mul(a, b, p, t.pneg[j]), t.scale[j], t.scale_s[j], p);
}

// --- decompose -------------------------------------------------------------
// A block handles coefficients kb .. kb + nk - 1 (nk <= kDecTile), all primes.
// Phase 1 (items: nk * L)   copies the limbs to a padded tile, row stride L + 1
//                           (odd, so the phase 2 reads are bank-conflict free).
// Phase 2 (items: 32 * K)   item (j, lane) writes residues (j, lane + 32 c) for
//                           c < kDecPer: one prime's table entries serve kDecPer
//                           coefficients, and each of a warp's kDecPer stores is
//                           32 consecutive coefficients of one prime, so they coalesce.

// L is a template parameter (dispatched with with_L, like the CRT): the limb
// loop unrolls and i / L, i % L are divisions by a constant.
template <int L>
RNS_HD void dec_load(uint32_t i, uint32_t* tile, const uint32_t* limbs, uint32_t kb) {
    tile[(i / L) * (L + 1) + i % L] = limbs[static_cast<size_t>(kb) * L + i];
}

template <int L>
RNS_HD void dec_compute(uint32_t b, const uint32_t* tile, uint32_t* res,
                        uint32_t kb, uint32_t nk, const Tables& t) {
    const uint32_t lane = b % 32, j = b / 32;
    if (lane >= nk) return;
    // Rows past nk are unset: computed from whatever is there, never stored.
    uint32_t r[kDecPer];
    decompose_many<L, kDecPer>([&](int c) { return tile + (lane + 32 * c) * (L + 1); },
                               t.pow2 + j * L, t.pow2_s + j * L, t.p[j], r);
    uint32_t* out = res + static_cast<size_t>(j) * t.N + kb + lane;
RNS_UNROLL
    for (uint32_t c = 0; c < kDecPer; ++c) {
        if (lane + 32 * c < nk) out[32 * c] = r[c];
    }
}

// --- crt -------------------------------------------------------------------
// A block handles coefficients kb .. kb + nk - 1 (nk <= kCrtTile), with the K
// primes split into kCrtGroups groups, so each coefficient has kCrtGroups
// threads for the products and a chunk of the output keeps the GPU busy.
// Phase 1 (items: kCrtTile * K)          residue (j, kb + b) to ytile[j * kCrtTile + b]:
//                                        consecutive items are consecutive b, so
//                                        reads coalesce, all loads in flight at once.
// Phase 2 (items: kCrtTile * kCrtGroups) item (g, b): the partial of coefficient b
//                                        over prime group g. A warp is one g and all
//                                        b, so its M loads are uniform and its smem
//                                        accesses are consecutive words.
// Phase 3 (items: nk)                    coefficient b from its partials, into a
//                                        padded tile.
// Phase 4 (items: nk * L)                the tile out to c, contiguous, so writes coalesce.
//
// part[(g * (L + 2) + w) * kCrtTile + b] is word w of group g's partial of b.

// The CRT block's shared memory, sized by the point (K up to kMaxK, L up to
// kMaxL is too much for a static 48 KB): fracs (doubles, first for alignment),
// ytile, part, then the output tile.
struct CrtSmem {
    double* fracs;
    uint32_t *ytile, *part, *tile;
};

RNS_HD size_t crt_smem_bytes(int L, int K) {
    return sizeof(double) * kCrtGroups * kCrtTile +
           sizeof(uint32_t) * (static_cast<size_t>(kCrtTile) * K + kCrtGroups * (L + 2) * kCrtTile +
                               kCrtTile * (L + 1));
}

RNS_HD CrtSmem crt_smem(void* base, int L, int K) {
    CrtSmem m;
    m.fracs = static_cast<double*>(base);
    m.ytile = reinterpret_cast<uint32_t*>(m.fracs + kCrtGroups * kCrtTile);
    m.part = m.ytile + static_cast<size_t>(kCrtTile) * K;
    m.tile = m.part + kCrtGroups * (L + 2) * kCrtTile;
    return m;
}

RNS_HD void crt_load(uint32_t i, uint32_t* ytile, const uint32_t* Y, uint32_t kb,
                     uint32_t nk, uint32_t N) {
    const uint32_t b = i % kCrtTile, j = i / kCrtTile;
    if (b < nk) ytile[i] = Y[static_cast<size_t>(j) * N + kb + b];
}

template <int L>
RNS_HD void crt_products(uint32_t i, const uint32_t* ytile, uint32_t* part, double* fracs,
                         uint32_t nk, const CrtView& v) {
    const uint32_t b = i % kCrtTile, g = i / kCrtTile;
    if (b >= nk) return;
    const int j0 = static_cast<int>(g * v.K / kCrtGroups);
    const int j1 = static_cast<int>((g + 1) * v.K / kCrtGroups);
    crt_partial<L>(ytile + b, kCrtTile, j0, j1, v,
                   part + g * (L + 2) * kCrtTile + b, kCrtTile, fracs + g * kCrtTile + b);
}

template <int L>
RNS_HD void crt_combine(uint32_t b, const uint32_t* part, const double* fracs, uint32_t* tile,
                        const CrtView& v) {
    crt_finish<L>(part + b, kCrtTile, (L + 2) * kCrtTile, kCrtGroups, fracs + b, kCrtTile, v,
                  tile + b * (L + 1));
}

RNS_HD void crt_store(uint32_t i, const uint32_t* tile, uint32_t* c, uint32_t kb, int L) {
    c[static_cast<size_t>(kb) * L + i] = tile[(i / L) * (L + 1) + i % L];
}

}  // namespace rns
