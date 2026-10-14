// RNS + negacyclic NTT + CRT for polymul/negacyclic, shared by host and device.
//
// This is rns_ref.py in the form the kernels use: every per-element operation
// is a __host__ __device__ function on 32-bit words, so the CPU build
// (solve_cpu.cpp) runs exactly the arithmetic the GPU will, and can be checked
// against oracle.py on a machine without a GPU. The kernels only decide which
// thread computes which element.
//
//     c = a*b mod (X^N + 1, q)
//
// 1. decompose  a coefficient (L limbs) -> its residue mod each prime p_j
// 2. forward    negacyclic NTT per prime (Cooley-Tukey, bit-reversed out)
// 3. pointwise  C = A*B, times the constant that makes step 4's output y_j
// 4. inverse    NTT per prime (Gentleman-Sande, natural order out)
// 5. crt        y_j (j < K) -> sum_j y_j*(P/p_j mod q) + (-v*P mod q), mod q
//
// The primes are p_j < 2^31 with p_j = 1 (mod 2^16), so one set serves every
// N <= 2^15, residue products fit in 62 bits, and a lazy value below 2p still
// fits in a u32. Their product P exceeds 2^20 * 2 * N * q^2, which makes the
// rounding in step 5 exact.
#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#define RNS_HD __host__ __device__ __forceinline__
#define RNS_UNROLL _Pragma("unroll")
#else
#define RNS_HD inline
#define RNS_UNROLL
#endif

namespace rns {

constexpr int kMaxL = 64;          // limbs per coefficient (W <= 2048)
constexpr int kMaxK = 160;         // primes (~115 at W = 1747, N = 2^16; ~134 at W = 2048)
constexpr int kMaxLogN = 22;       // the NTT's row pass needs rows of at most kNttTile
constexpr double kGenTwiddleBytes = 64e6;   // full tables above this: generate twiddles
constexpr int kMarginBits = 20;    // |c| / P < 2^-20: the CRT rounding is exact

// ---------------------------------------------------------------------------
// Word arithmetic mod p < 2^31.

RNS_HD uint32_t mulhi(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a, b);
#else
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
#endif
}

RNS_HD uint32_t add_mod(uint32_t a, uint32_t b, uint32_t p) {
    const uint32_t s = a + b;          // < 2p < 2^32
    return s >= p ? s - p : s;
}

RNS_HD uint32_t sub_mod(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? a - b : a + p - b;
}

// a * w mod p for a constant w < p with its Shoup companion ws = floor(w*2^32/p).
// Any a < 2^32 is fine: the quotient estimate is off by at most one, so the
// remainder before the final correction is below 2p.
RNS_HD uint32_t mul_shoup(uint32_t a, uint32_t w, uint32_t ws, uint32_t p) {
    const uint32_t r = a * w - mulhi(a, ws) * p;   // mod 2^32, true value < 2p
    return r >= p ? r - p : r;
}

// The low `bits` bits of x in reverse order (bits in [1, 32]).
RNS_HD uint32_t bitrev_bits(uint32_t x, uint32_t bits) {
#ifdef __CUDA_ARCH__
    return __brev(x) >> (32 - bits);
#else
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) r |= ((x >> i) & 1u) << (bits - 1 - i);
    return r;
#endif
}

// mul_shoup without its final correction: a * w mod p plus 0 or p, below 2p.
RNS_HD uint32_t mul_shoup_lazy(uint32_t a, uint32_t w, uint32_t ws, uint32_t p) {
    return a * w - mulhi(a, ws) * p;
}

// Montgomery product a*b*2^-32 mod p, for a, b < p. pneg = -p^-1 mod 2^32.
RNS_HD uint32_t mont_mul(uint32_t a, uint32_t b, uint32_t p, uint32_t pneg) {
    const uint64_t t = static_cast<uint64_t>(a) * b;          // < 2^62
    const uint32_t m = static_cast<uint32_t>(t) * pneg;
    const uint32_t u = static_cast<uint32_t>((t + static_cast<uint64_t>(m) * p) >> 32);
    return u >= p ? u - p : u;                                 // u < 2p
}

// ---------------------------------------------------------------------------
// Step 1: a coefficient's residue mod p, as sum_l limb_l * (2^(32 l) mod p).
//
// C coefficients per call, all for the same prime, so each table entry loaded
// serves C terms: the kernel was bound by its two table loads per term (58 global
// loads per residue against 101 multiplies), not by its arithmetic.
// limbs(c) gives coefficient c's limbs; out[c] gets its residue.
//
// Lazily: each term is reduced only to below 2p, the terms are summed in 64
// bits (< 2^37 for L <= 32), and the sum hi * 2^32 + lo is reduced once, as
// lo * 1 + hi * (2^32 mod p), with pow2[0] = 1 and pow2[1] = 2^32 mod p and
// their Shoup companions.
//
// L is a template parameter so the loops unroll and the accumulators stay in
// registers.
template <int L, int C, class Limbs>
RNS_HD void decompose_many(Limbs limbs, const uint32_t* pow2, const uint32_t* pow2_s, uint32_t p,
                           uint32_t (&out)[C]) {
    uint64_t acc[C];
RNS_UNROLL
    for (int c = 0; c < C; ++c) acc[c] = 0;
RNS_UNROLL
    for (int l = 0; l < L; ++l) {
        const uint32_t w = pow2[l], ws = pow2_s[l];
RNS_UNROLL
        for (int c = 0; c < C; ++c) acc[c] += mul_shoup_lazy(limbs(c)[l], w, ws, p);
    }
RNS_UNROLL
    for (int c = 0; c < C; ++c) {
        const uint32_t lo = mul_shoup(static_cast<uint32_t>(acc[c]), pow2[0], pow2_s[0], p);
        out[c] = L == 1 ? lo   // acc < 2p: no high word
                        : add_mod(lo, mul_shoup(static_cast<uint32_t>(acc[c] >> 32), pow2[L > 1], pow2_s[L > 1], p), p);
    }
}

// ---------------------------------------------------------------------------
// Steps 2 and 4: the butterflies. Twiddles are psi^bitrev(i) (forward) and
// psi^-bitrev(i) (inverse), with psi a primitive 2N-th root of unity, so the
// negacyclic twist is folded into the transform and no pre/post pass is needed.

// A twiddle and its Shoup companion side by side, so a butterfly fetches both
// with one 8-byte load (the NTT kernels spent ~16 us of a call on twiddle loads
// when they were two separate tables).
struct alignas(8) Twiddle {
    uint32_t w, ws;
};

// Forward (Cooley-Tukey): (x, y) -> (x + w y, x - w y).
RNS_HD void ct_butterfly(uint32_t& x, uint32_t& y, uint32_t w, uint32_t ws, uint32_t p) {
    const uint32_t v = mul_shoup(y, w, ws, p);
    const uint32_t u = x;
    x = add_mod(u, v, p);
    y = sub_mod(u, v, p);
}

// Inverse (Gentleman-Sande): (x, y) -> (x + y, (x - y) w).
RNS_HD void gs_butterfly(uint32_t& x, uint32_t& y, uint32_t w, uint32_t ws, uint32_t p) {
    const uint32_t u = x, v = y;
    x = add_mod(u, v, p);
    y = mul_shoup(sub_mod(u, v, p), w, ws, p);
}

// ---------------------------------------------------------------------------
// Multi-limb helpers. A number is n little-endian u32 limbs; q has L limbs and
// is read as zero above them.

RNS_HD bool geq(const uint32_t* x, int n, const uint32_t* q, int L) {
RNS_UNROLL
    for (int i = n - 1; i >= 0; --i) {
        const uint32_t qi = i < L ? q[i] : 0;
        if (x[i] != qi) return x[i] > qi;
    }
    return true;
}

RNS_HD void sub_in_place(uint32_t* x, int n, const uint32_t* q, int L) {
    uint32_t borrow = 0;
RNS_UNROLL
    for (int i = 0; i < n; ++i) {
        const uint64_t d = static_cast<uint64_t>(x[i]) - (i < L ? q[i] : 0) - borrow;
        x[i] = static_cast<uint32_t>(d);
        borrow = static_cast<uint32_t>(d >> 63);
    }
}

// x -= (e * q) << (32 * shift), for a result known to be non-negative.
RNS_HD void submul(uint32_t* x, int n, const uint32_t* q, int L, uint32_t e, int shift) {
    uint64_t carry = 0;
    uint32_t borrow = 0;
RNS_UNROLL
    for (int i = shift; i < n; ++i) {
        const int k = i - shift;
        const uint64_t prod = (k < L ? static_cast<uint64_t>(e) * q[k] : 0) + carry;
        carry = prod >> 32;
        const uint64_t d = static_cast<uint64_t>(x[i]) - static_cast<uint32_t>(prod) - borrow;
        x[i] = static_cast<uint32_t>(d);
        borrow = static_cast<uint32_t>(d >> 63);
    }
}

// q, and q / 2^(32 s) as a double, where s = max(0, L - 3) limbs are dropped
// from both sides of reduce_wide's quotient estimate.
struct Modulus {
    const uint32_t* q;
    int L;
    double q_top;
};

RNS_HD int dropped_limbs(int L) { return L > 3 ? L - 3 : 0; }

// x mod q in place, for x < 2^64 * q (n limbs). The quotient is estimated in
// double from the top limbs; the estimate is within 1/4 of the true quotient,
// so one less than its floor never overshoots and leaves x below 3q, which at
// most two subtractions finish.
//
// n and L are plain arguments so that a caller with compile-time sizes
// (crt_finish<L>) gets every loop unrolled and x kept in registers.
RNS_HD void reduce_wide(uint32_t* x, int n, const uint32_t* q, int L, double q_top) {
    double xd = 0;
RNS_UNROLL
    for (int i = n - 1; i >= dropped_limbs(L); --i) xd = xd * 4294967296.0 + x[i];
    const double f = floor(xd / q_top);
    const uint64_t e = f >= 1 ? static_cast<uint64_t>(f) - 1 : 0;
    if (e >> 32) submul(x, n, q, L, static_cast<uint32_t>(e >> 32), 1);
    if (e) submul(x, n, q, L, static_cast<uint32_t>(e), 0);
    while (geq(x, n, q, L)) sub_in_place(x, n, q, L);
}

// ---------------------------------------------------------------------------
// Step 5: one output coefficient from its K residues y_j (already multiplied by
// (P/p_j)^-1 in step 3, so y_j = c * (P/p_j)^-1 mod p_j).
//
// The exact centred coefficient is c = sum_j y_j (P/p_j) - v P with
// v = round(sum_j y_j / p_j); the margin puts sum_j y_j / p_j within 2^-21 of
// an integer, far beyond double's error. Mod q that is
//     sum_j y_j * M_j + negvP_v,   M_j = P/p_j mod q,  negvP_v = -v P mod q.

struct CrtView {
    int K;
    Modulus mod;
    const double* p_recip;     // K:  1 / p_j
    const uint32_t* M;         // K x L
    const uint32_t* negvP;     // (K + 1) x L
};

// The sum is split over groups of primes so that several threads can share one
// coefficient: crt_partial covers primes [j0, j1), crt_finish adds the groups.
// L is a template parameter so the accumulators live in registers on the GPU.

// Primes [j0, j1): sum_j y_j * M_j as L + 2 words (< K * 2^31 * q, which fits)
// into part[w * pstride], and sum_j y_j / p_j into *frac.
template <int L>
RNS_HD void crt_partial(const uint32_t* y, uint32_t stride, int j0, int j1, const CrtView& v,
                        uint32_t* part, uint32_t pstride, double* frac) {
    double f = 0;
    for (int j = j0; j < j1; ++j) f += y[j * stride] * v.p_recip[j];
    *frac = f;

    // Column sums: each product adds < 2^32 to two columns.
    uint64_t col[L + 2];
RNS_UNROLL
    for (int l = 0; l < L + 2; ++l) col[l] = 0;
    for (int j = j0; j < j1; ++j) {
        const uint64_t yj = y[j * stride];
        const uint32_t* Mj = v.M + j * L;
RNS_UNROLL
        for (int l = 0; l < L; ++l) {
            const uint64_t prod = yj * Mj[l];
            col[l] += static_cast<uint32_t>(prod);
            col[l + 1] += prod >> 32;
        }
    }
    uint64_t carry = 0;
RNS_UNROLL
    for (int l = 0; l < L + 2; ++l) {
        const uint64_t s = col[l] + carry;
        part[l * pstride] = static_cast<uint32_t>(s);
        carry = s >> 32;
    }
}

// One output coefficient from G partials (group g at part + g * gstride, word w
// at + w * pstride; its frac at fracs[g * fstride]).
template <int L>
RNS_HD void crt_finish(const uint32_t* part, uint32_t pstride, uint32_t gstride, int G,
                       const double* fracs, uint32_t fstride, const CrtView& v, uint32_t* out) {
    const int K = v.K;
    double frac = 0;
    for (int g = 0; g < G; ++g) frac += fracs[g * fstride];
    int vi = static_cast<int>(floor(frac + 0.5));
    vi = vi < 0 ? 0 : (vi > K ? K : vi);

    uint64_t col[L + 2];
    const uint32_t* t = v.negvP + vi * L;
RNS_UNROLL
    for (int l = 0; l < L + 2; ++l) col[l] = l < L ? t[l] : 0;
    for (int g = 0; g < G; ++g) {
RNS_UNROLL
        for (int l = 0; l < L + 2; ++l) col[l] += part[g * gstride + l * pstride];
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
    for (int l = 0; l < L; ++l) out[l] = x[l];
}

// Calls f(std::integral_constant<int, L>{}) for a run-time L in [1, kMaxL]:
// the bridge from the point's L to the CRT templates. Host only.
template <int Lc = 1, class F>
void with_L(int L, F&& f) {
    if constexpr (Lc <= kMaxL) {
        if (L == Lc) {
            f(std::integral_constant<int, Lc>{});
        } else {
            with_L<Lc + 1>(L, static_cast<F&&>(f));
        }
    }
}

}  // namespace rns
