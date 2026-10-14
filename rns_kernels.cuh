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
//                                                           kCrtGroups warps (limb ranges)
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
// Coefficients per CRT thread. Measured (kernel time per chunk at N = 32768, W = 868 /
// N = 65536, W = 1747): 2 per thread 31.5 / 53.6 us, 1 per thread 15.9 / 50.0 us,
// 1 per thread with the prime loop unrolled by 4: 13.1 / 50.1 us (the old split by
// prime groups: 11.9 / ~170 us). More blocks matter more than reusing M entries.
constexpr uint32_t kCrtPer = 1;
constexpr uint32_t kCrtTile = 32 * kCrtPer;    // coefficients per CRT block
constexpr uint32_t kCrtGroups = 4;             // limb groups = warps per CRT block

// Device-side views of the Plan tables (pointers into device memory on the GPU,
// into the Plan's vectors on the CPU).
struct Tables {
    int K, L;
    uint32_t N;
    const uint32_t* p;
    const uint32_t* pneg;
    const uint32_t* pow2;      // K x L
    const uint32_t* pow2_s;
    const Twiddle* psi;        // K x N, table path only (else null)
    const Twiddle* ipsi;
    const uint32_t* scale;     // K
    const uint32_t* scale_s;
    // Generated path: Plan::tw_hi etc., and s.
    const uint32_t* tw_hi;     // K x (N >> tw_s)
    const uint32_t* tw_lo;     // K x 2^tw_s
    const uint32_t* itw_hi;
    const uint32_t* itw_lo;
    uint32_t tw_s, logN;
    const Twiddle* col_psi;    // K x (N >> tw_s): the column pass's twiddles
    const Twiddle* col_ipsi;
};

// Twiddle sources for the butterflies: mul(y, i) = y * w_i mod p, where w_i is
// psi^bitrev(i) (forward) or psi^-bitrev(i) (inverse) for one prime.
struct TableTwiddles {     // read from the full table, Shoup form
    const Twiddle* w;
    uint32_t p;
    RNS_HD uint32_t mul(uint32_t y, uint32_t i) const {
        const Twiddle t = w[i];
        return mul_shoup(y, t.w, t.ws, p);
    }
};

struct GenTwiddles {       // built from two small Montgomery-form tables
    const uint32_t* hi;
    const uint32_t* lo;
    uint32_t s, logN, p, pneg;
    RNS_HD uint32_t mul(uint32_t y, uint32_t i) const {
        const uint32_t e = bitrev_bits(i, logN);
        const uint32_t w = mont_mul(hi[e >> s], lo[e & ((1u << s) - 1)], p, pneg);   // psi^e 2^32
        return mont_mul(y, w, p, pneg);
    }
};

// The column pass's twiddle indices, (1 << (logR - lt - 1)) + blk, are all below
// 2^logR <= N >> tw_s: it reads the head table on both paths.
RNS_HD TableTwiddles col_twiddles(const Tables& t, int j, bool forward) {
    const size_t hn = static_cast<size_t>(t.N) >> t.tw_s;
    return TableTwiddles{(forward ? t.col_psi : t.col_ipsi) + j * hn, t.p[j]};
}

template <bool kGen>
RNS_HD auto twiddles(const Tables& t, int j, bool forward) {
    if constexpr (kGen) {
        const size_t hn = static_cast<size_t>(t.N) >> t.tw_s, ln = static_cast<size_t>(1) << t.tw_s;
        return GenTwiddles{(forward ? t.tw_hi : t.itw_hi) + j * hn, (forward ? t.tw_lo : t.itw_lo) + j * ln,
                           t.tw_s, t.logN, t.p[j], t.pneg[j]};
    } else {
        return TableTwiddles{(forward ? t.psi : t.ipsi) + static_cast<size_t>(j) * t.N, t.p[j]};
    }
}

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
// Forward (Cooley-Tukey): (x, y) -> (x + w y, x - w y).
template <class Tw>
RNS_HD void fwd_butterfly(uint32_t* tile, const Pair& pr, const Tw& tw, uint32_t p) {
    const uint32_t u = tile[pr.lo], v = tw.mul(tile[pr.hi], pr.tw);
    tile[pr.lo] = add_mod(u, v, p);
    tile[pr.hi] = sub_mod(u, v, p);
}

// Inverse (Gentleman-Sande): (x, y) -> (x + y, (x - y) w).
template <class Tw>
RNS_HD void inv_butterfly(uint32_t* tile, const Pair& pr, const Tw& tw, uint32_t p) {
    const uint32_t u = tile[pr.lo], v = tile[pr.hi];
    tile[pr.lo] = add_mod(u, v, p);
    tile[pr.hi] = tw.mul(sub_mod(u, v, p), pr.tw);
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
// A block handles coefficients kb .. kb + nk - 1 (nk <= kCrtTile) with
// kCrtGroups warps. Warp g owns the output limbs [l0, l0 + span) of
//     sum_j y_j M_j        (span = ceil(L / kCrtGroups), l0 = g span)
// for all of the block's coefficients, and the rounding sum sum_j y_j / p_j over
// the primes j = g (mod kCrtGroups). Lane i takes coefficients i, i + 32, ...:
// each M entry it loads (uniform across the warp) serves kCrtPer products, and
// its y loads are 32 consecutive coefficients of one prime, so they coalesce.
// Splitting by limbs rather than by primes keeps each partial to span + 2 words,
// so the block's shared memory no longer grows with K (~10 KB at L = 55).
//
// Phase 1 (items: 32 kCrtGroups)  item (g, lane): its column sums, carried to
//                                 span + 2 words, into row k at offset g (span + 2),
//                                 and its partial rounding sums into fracs.
// Phase 2 (items: nk)             coefficient k: the partials plus -vP mod q,
//                                 reduced mod q, written over row k.
// Phase 3 (items: nk * L)         the rows out to c, contiguous, so writes coalesce.
// Rows have an odd stride, so the lanes of a warp hit distinct banks.

RNS_HD constexpr int crt_span(int L) { return (L + static_cast<int>(kCrtGroups) - 1) / static_cast<int>(kCrtGroups); }

RNS_HD constexpr int crt_stride(int L) {
    return (static_cast<int>(kCrtGroups) * (crt_span(L) + 2) > L + 1 ? static_cast<int>(kCrtGroups) * (crt_span(L) + 2)
                                                                      : L + 1) | 1;
}

// fracs (kCrtGroups x kCrtTile doubles, first for alignment), then kCrtTile rows.
struct CrtSmem {
    double* fracs;
    uint32_t* rows;
};

RNS_HD size_t crt_smem_bytes(int L) {
    return sizeof(double) * kCrtGroups * kCrtTile + sizeof(uint32_t) * static_cast<size_t>(kCrtTile) * crt_stride(L);
}

RNS_HD CrtSmem crt_smem(void* base) {
    CrtSmem m;
    m.fracs = static_cast<double*>(base);
    m.rows = reinterpret_cast<uint32_t*>(m.fracs + kCrtGroups * kCrtTile);
    return m;
}

template <int L>
RNS_HD void crt_columns(uint32_t i, const uint32_t* Y, uint32_t kb, uint32_t nk, uint32_t N,
                        const CrtView& v, const CrtSmem& m) {
    constexpr int span = crt_span(L), stride = crt_stride(L);
    const uint32_t lane = i % 32, g = i / 32;
    const int l0 = static_cast<int>(g) * span;
    const int n = L - l0 < span ? L - l0 : span;    // this group's limbs (<= 0: none)

    // Each product adds < 2^32 to two columns: K <= kMaxK keeps a column < 2^41.
    uint64_t col[kCrtPer][span + 1];
    double f[kCrtPer];
RNS_UNROLL
    for (uint32_t c = 0; c < kCrtPer; ++c) {
        f[c] = 0;
RNS_UNROLL
        for (int w = 0; w <= span; ++w) col[c][w] = 0;
    }
    // Unrolled so several primes' y loads are in flight: each iteration starts
    // with a global load and has little arithmetic to hide it behind.
#ifdef __CUDA_ARCH__
#pragma unroll 4
#endif
    for (int j = 0; j < v.K; ++j) {
        uint32_t y[kCrtPer];
RNS_UNROLL
        for (uint32_t c = 0; c < kCrtPer; ++c) {
            const uint32_t k = lane + 32 * c;
            y[c] = k < nk ? Y[static_cast<size_t>(j) * N + kb + k] : 0;
        }
        if (j % static_cast<int>(kCrtGroups) == static_cast<int>(g)) {
            const double r = v.p_recip[j];
RNS_UNROLL
            for (uint32_t c = 0; c < kCrtPer; ++c) f[c] += y[c] * r;
        }
RNS_UNROLL
        for (int w = 0; w < span; ++w) {
            if (w < n) {
                const uint64_t mw = v.M[static_cast<size_t>(j) * L + l0 + w];
RNS_UNROLL
                for (uint32_t c = 0; c < kCrtPer; ++c) {
                    const uint64_t prod = y[c] * mw;
                    col[c][w] += static_cast<uint32_t>(prod);
                    col[c][w + 1] += prod >> 32;
                }
            }
        }
    }
RNS_UNROLL
    for (uint32_t c = 0; c < kCrtPer; ++c) {
        const uint32_t k = lane + 32 * c;
        if (k >= nk) continue;
        uint32_t* row = m.rows + k * stride + g * (span + 2);
        uint64_t carry = 0;
RNS_UNROLL
        for (int w = 0; w < span + 2; ++w) {
            const uint64_t s = (w <= span ? col[c][w] : 0) + carry;
            row[w] = static_cast<uint32_t>(s);
            carry = s >> 32;
        }
        m.fracs[g * kCrtTile + k] = f[c];
    }
}

// The exact centred coefficient is c = sum_j y_j (P/p_j) - v P with
// v = round(sum_j y_j / p_j) (see CrtView); mod q that is sum_j y_j M_j + negvP_v.
template <int L>
RNS_HD void crt_combine(uint32_t k, const CrtView& v, const CrtSmem& m) {
    constexpr int span = crt_span(L), stride = crt_stride(L);
    double frac = 0;
RNS_UNROLL
    for (uint32_t g = 0; g < kCrtGroups; ++g) frac += m.fracs[g * kCrtTile + k];
    int vi = static_cast<int>(floor(frac + 0.5));
    vi = vi < 0 ? 0 : (vi > v.K ? v.K : vi);

    uint32_t* row = m.rows + k * stride;
    const uint32_t* t = v.negvP + vi * L;
    uint64_t col[L + 2];
RNS_UNROLL
    for (int l = 0; l < L + 2; ++l) col[l] = l < L ? t[l] : 0;
    // Group g's words sit at limbs g span .. g span + span + 1; any past L + 1 are 0.
RNS_UNROLL
    for (int g = 0; g < static_cast<int>(kCrtGroups); ++g) {
RNS_UNROLL
        for (int w = 0; w < span + 2; ++w) {
            if (g * span + w < L + 2) col[g * span + w] += row[g * (span + 2) + w];
        }
    }
    // < K * 2^31 * q + q < 2^(32 (L + 2)): L + 2 limbs.
    uint32_t x[L + 2];
    uint64_t carry = 0;
RNS_UNROLL
    for (int l = 0; l < L + 2; ++l) {
        const uint64_t s = col[l] + carry;
        x[l] = static_cast<uint32_t>(s);
        carry = s >> 32;
    }
    reduce_wide(x, L + 2, v.mod.q, L, v.mod.q_top);
RNS_UNROLL
    for (int l = 0; l < L; ++l) row[l] = x[l];   // all of row k was read above
}

template <int L>
RNS_HD void crt_store(uint32_t i, const uint32_t* rows, uint32_t* c, uint32_t kb) {
    c[static_cast<size_t>(kb) * L + i] = rows[(i / L) * crt_stride(L) + i % L];
}

}  // namespace rns
