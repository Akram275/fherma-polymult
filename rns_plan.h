// Host-side setup for rns.cuh: the prime set, the twiddle tables and the CRT
// constants for one point (N, q). Built once in fherma_init, outside the timed
// window; everything here is plain host code and may be slow.
#pragma once

#include "rns.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace rns {

struct Plan {
    uint32_t N = 0, logN = 0;
    int L = 0, K = 0;
    double q_top = 0;
    std::vector<uint32_t> q;                  // L

    std::vector<uint32_t> p, pneg;            // K; pneg = -p^-1 mod 2^32
    std::vector<uint32_t> pow2, pow2_s;       // K x L: 2^(32 l) mod p_j
    std::vector<uint32_t> psi, psi_s;         // K x N: psi^bitrev(i)
    std::vector<uint32_t> ipsi, ipsi_s;       // K x N: psi^-bitrev(i)
    std::vector<uint32_t> scale, scale_s;     // K: N^-1 (P/p_j)^-1 2^32 mod p_j

    std::vector<double> p_recip;              // K
    std::vector<uint32_t> M;                  // K x L: P/p_j mod q
    std::vector<uint32_t> negvP;              // (K + 1) x L: -v P mod q

    Modulus modulus() const { return Modulus{q.data(), L, q_top}; }
    CrtView crt() const {
        return CrtView{K, modulus(), p_recip.data(), M.data(), negvP.data()};
    }
};

namespace detail {

inline uint32_t pow_mod(uint64_t b, uint64_t e, uint32_t p) {
    uint64_t r = 1;
    b %= p;
    for (; e; e >>= 1, b = b * b % p) {
        if (e & 1) r = r * b % p;
    }
    return static_cast<uint32_t>(r);
}

// Deterministic Miller-Rabin: bases 2, 3, 5, 7 decide every n < 3.2e9.
inline bool is_prime(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t d : {2u, 3u, 5u, 7u}) {
        if (n % d == 0) return n == d;
    }
    uint32_t d = n - 1;
    int r = 0;
    while (!(d & 1)) d >>= 1, ++r;
    for (uint32_t a : {2u, 3u, 5u, 7u}) {
        uint64_t x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int i = 1; i < r && composite; ++i) {
            x = x * x % n;
            if (x == n - 1) composite = false;
        }
        if (composite) return false;
    }
    return true;
}

inline uint32_t shoup(uint32_t w, uint32_t p) {
    return static_cast<uint32_t>((static_cast<uint64_t>(w) << 32) / p);
}

inline uint32_t bitrev(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) r |= ((x >> i) & 1) << (bits - 1 - i);
    return r;
}

// x = x * m mod q, for x < q.
inline void mul_small_mod(std::vector<uint32_t>& x, uint32_t m, const Modulus& mod) {
    std::vector<uint32_t> t(mod.L + 1);
    uint64_t carry = 0;
    for (int l = 0; l < mod.L; ++l) {
        const uint64_t v = static_cast<uint64_t>(x[l]) * m + carry;
        t[l] = static_cast<uint32_t>(v);
        carry = v >> 32;
    }
    t[mod.L] = static_cast<uint32_t>(carry);
    reduce_wide(t.data(), mod.L + 1, mod.q, mod.L, mod.q_top);
    for (int l = 0; l < mod.L; ++l) x[l] = t[l];
}

inline int bit_length(const std::vector<uint32_t>& x) {
    for (int i = static_cast<int>(x.size()) - 1; i >= 0; --i) {
        if (x[i]) return 32 * i + 32 - __builtin_clz(x[i]);
    }
    return 0;
}

}  // namespace detail

inline Plan make_plan(uint32_t N, const uint32_t* q, int L) {
    using namespace detail;
    if (N < 2 || (N & (N - 1)) || N > (1u << 15)) {
        throw std::runtime_error("N must be a power of two in [2, 2^15]; got " + std::to_string(N));
    }
    if (L < 1 || L > kMaxL) {
        throw std::runtime_error("L must be in [1, " + std::to_string(kMaxL) + "]; got " + std::to_string(L));
    }

    Plan pl;
    pl.N = N;
    while ((1u << pl.logN) < N) ++pl.logN;
    pl.L = L;
    pl.q.assign(q, q + L);
    const int qbits = bit_length(pl.q);
    if (qbits <= 32 * (L - 1)) throw std::runtime_error("q has fewer limbs than L");
    for (int i = L - 1; i >= dropped_limbs(L); --i) pl.q_top = pl.q_top * 4294967296.0 + pl.q[i];
    const Modulus mod = pl.modulus();

    // |c| < N q^2, and the rounding needs |c| / P < 2^-(margin + 1).
    const double need_bits = 2.0 * qbits + pl.logN + 1 + kMarginBits;
    double have_bits = 0;
    const uint32_t step = 1u << 16;
    for (uint32_t c = ((1u << 31) - 1) / step * step + 1; have_bits <= need_bits; c -= step) {
        if (c < step) throw std::runtime_error("ran out of primes");
        if (!is_prime(c)) continue;
        pl.p.push_back(c);
        have_bits += std::log2(static_cast<double>(c));
    }
    pl.K = static_cast<int>(pl.p.size());
    if (pl.K > kMaxK) throw std::runtime_error("K = " + std::to_string(pl.K) + " exceeds kMaxK");
    const int K = pl.K;

    pl.pneg.resize(K);
    pl.pow2.resize(K * L);
    pl.pow2_s.resize(K * L);
    pl.psi.resize(static_cast<size_t>(K) * N);
    pl.psi_s.resize(static_cast<size_t>(K) * N);
    pl.ipsi.resize(static_cast<size_t>(K) * N);
    pl.ipsi_s.resize(static_cast<size_t>(K) * N);
    pl.scale.resize(K);
    pl.scale_s.resize(K);
    pl.p_recip.resize(K);

    for (int j = 0; j < K; ++j) {
        const uint32_t p = pl.p[j];

        uint32_t inv = 1;                      // p^-1 mod 2^32 by Newton
        for (int i = 0; i < 5; ++i) inv *= 2 - p * inv;
        pl.pneg[j] = 0u - inv;

        for (int l = 0; l < L; ++l) {
            pl.pow2[j * L + l] = pow_mod(2, 32ull * l, p);
            pl.pow2_s[j * L + l] = shoup(pl.pow2[j * L + l], p);
        }

        uint32_t psi = 0;
        for (uint32_t g = 2; !psi; ++g) {
            const uint32_t cand = pow_mod(g, (p - 1) / (2ull * N), p);
            if (pow_mod(cand, N, p) == p - 1) psi = cand;
        }
        const uint32_t ipsi = pow_mod(psi, p - 2, p);
        for (uint32_t i = 0; i < N; ++i) {
            const uint32_t e = bitrev(i, pl.logN);
            const size_t at = static_cast<size_t>(j) * N + i;
            pl.psi[at] = pow_mod(psi, e, p);
            pl.psi_s[at] = shoup(pl.psi[at], p);
            pl.ipsi[at] = pow_mod(ipsi, e, p);
            pl.ipsi_s[at] = shoup(pl.ipsi[at], p);
        }

        uint64_t Pj = 1;                       // P/p_j mod p_j
        for (int i = 0; i < K; ++i) {
            if (i != j) Pj = Pj * (pl.p[i] % p) % p;
        }
        const uint64_t n_inv = pow_mod(N, p - 2, p);
        const uint64_t Pj_inv = pow_mod(Pj, p - 2, p);
        const uint64_t r = (1ull << 32) % p;   // undoes mont_mul's 2^-32
        pl.scale[j] = static_cast<uint32_t>(n_inv * Pj_inv % p * r % p);
        pl.scale_s[j] = shoup(pl.scale[j], p);

        pl.p_recip[j] = 1.0 / p;
    }

    // M_j = prod_{i != j} p_i mod q, and P mod q.
    pl.M.assign(static_cast<size_t>(K) * L, 0);
    std::vector<uint32_t> Pq(L, 0);
    Pq[0] = 1;
    for (int j = 0; j < K; ++j) {
        std::vector<uint32_t> m(L, 0);
        m[0] = 1;
        for (int i = 0; i < K; ++i) {
            if (i != j) mul_small_mod(m, pl.p[i], mod);
        }
        std::copy(m.begin(), m.end(), pl.M.begin() + j * L);
        mul_small_mod(Pq, pl.p[j], mod);
    }

    // negvP_v = (q - v P mod q) mod q, for v = 0..K.
    pl.negvP.assign(static_cast<size_t>(K + 1) * L, 0);
    for (int v = 1; v <= K; ++v) {
        std::vector<uint32_t> t = Pq;
        mul_small_mod(t, static_cast<uint32_t>(v), mod);
        if (bit_length(t) == 0) continue;
        std::vector<uint32_t> neg = pl.q;
        sub_in_place(neg.data(), L, t.data(), L);
        std::copy(neg.begin(), neg.end(), pl.negvP.begin() + v * L);
    }
    return pl;
}

}  // namespace rns
