// CPU run of the GPU schedule, for checking it without a GPU.
//
//     g++ -O2 -std=c++17 -fopenmp -o solution_cpu main.cpp solve_cpu.cpp
//     ./solution_cpu <point dir> && python3 oracle.py verify <point dir>
//
// It links against the same generated main.cpp as the CUDA answer and runs
// solve.cu's kernels as written in rns_kernels.cuh: the same blocks, the same
// shared-memory tiles, the same phases, with each phase's items done in order
// instead of by a block's threads. On top of that it checks what a GPU run
// would not: that every NTT stage touches each tile element exactly once and
// only reads twiddles in range, so no two threads of a stage can collide.
// Not meant to be fast.
#include "fherma.h"
#include "rns_kernels.cuh"
#include "rns_plan.h"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct State {
    rns::Plan plan;
    rns::Tables t;
    rns::NttGeom g;
    std::vector<uint32_t> A, B;     // K x N residues
    bool gen_tw = false;            // the GPU's twiddle path, rns::use_gen_twiddles
};

rns::Tables tables(const rns::Plan& pl) {
    return rns::Tables{pl.K, pl.L, pl.N,
                       pl.p.data(), pl.pneg.data(), pl.pow2.data(), pl.pow2_s.data(),
                       pl.psi.empty() ? nullptr : pl.psi.data(), pl.ipsi.empty() ? nullptr : pl.ipsi.data(),
                       pl.scale.data(), pl.scale_s.data(),
                       pl.tw_hi.data(), pl.tw_lo.data(), pl.itw_hi.data(), pl.itw_lo.data(),
                       pl.tw_s, pl.logN, pl.col_psi.data(), pl.col_ipsi.data()};
}

// One phase of one block: items 0 .. count-1, which a block's threads share.
template <class F>
void phase(uint32_t count, F&& step) {
    for (uint32_t i = 0; i < count; ++i) step(i);
}

// One NTT stage on a tile, with the collision checks.
template <class PairOf, class Butterfly>
void stage(const rns::NttGeom& g, PairOf pair_of, Butterfly butterfly) {
    std::vector<uint8_t> seen(g.T, 0);
    phase(g.T / 2, [&](uint32_t b) {
        const rns::Pair pr = pair_of(b);
        if (pr.lo >= g.T || pr.hi >= g.T || pr.tw == 0 || pr.tw >= g.N ||
            seen[pr.lo]++ || seen[pr.hi]++) {
            throw std::logic_error("NTT stage schedule is broken at butterfly " + std::to_string(b));
        }
        butterfly(pr);
    });
}

template <int L>
void decompose(const State& s, const std::vector<uint32_t>& limbs, std::vector<uint32_t>& res) {
    const rns::Tables& t = s.t;
    const uint32_t blocks = (t.N + rns::kDecTile - 1) / rns::kDecTile;
#pragma omp parallel for
    for (uint32_t blk = 0; blk < blocks; ++blk) {
        uint32_t tile[rns::kDecTile * (L + 1)] = {};   // rows past nk are read, never stored
        const uint32_t kb = blk * rns::kDecTile;
        const uint32_t nk = std::min(rns::kDecTile, t.N - kb);
        phase(nk * L, [&](uint32_t i) { rns::dec_load<L>(i, tile, limbs.data(), kb); });
        phase(32 * t.K, [&](uint32_t b) { rns::dec_compute<L>(b, tile, res.data(), kb, nk, t); });
    }
}

// Column pass over A (and B when given), forward or inverse.
// Column pass on X, and on Y too when given (forward only).
void ntt_cols(State& s, bool forward, std::vector<uint32_t>& X, std::vector<uint32_t>* Y) {
    const bool both = Y != nullptr;
    const rns::Tables& t = s.t;
    const rns::NttGeom g = s.g;
    if (g.logR == 0) return;
    const uint32_t blocks = 1u << (g.logS - g.logC);
#pragma omp parallel for collapse(2)
    for (int j = 0; j < t.K; ++j) {
        for (uint32_t blk = 0; blk < blocks; ++blk) {
            std::vector<uint32_t> ta(g.T), tb(g.T);
            uint32_t* a = &X[static_cast<size_t>(j) * t.N];
            uint32_t* b = both ? &(*Y)[static_cast<size_t>(j) * t.N] : nullptr;
            const uint32_t r0 = blk << g.logC;
            const rns::TableTwiddles w = rns::col_twiddles(t, j, forward);
            const uint32_t p = t.p[j];

            phase(g.T, [&](uint32_t i) {
                ta[i] = a[rns::col_global(i, r0, g)];
                if (both) tb[i] = b[rns::col_global(i, r0, g)];
            });
            for (uint32_t k = 0; k < g.logR; ++k) {
                const uint32_t lt = forward ? g.logR - 1 - k : k;
                stage(g, [&](uint32_t bf) { return rns::col_pair(bf, lt, g); },
                      [&](const rns::Pair& pr) {
                          if (forward) {
                              rns::fwd_butterfly(ta.data(), pr, w, p);
                              if (both) rns::fwd_butterfly(tb.data(), pr, w, p);
                          } else {
                              rns::inv_butterfly(ta.data(), pr, w, p);
                          }
                      });
            }
            phase(g.T, [&](uint32_t i) {
                a[rns::col_global(i, r0, g)] = ta[i];
                if (both) b[rns::col_global(i, r0, g)] = tb[i];
            });
        }
    }
}

// Row pass: forward on A and B, pointwise into A, inverse on A.
// Forward row stages on A alone, back to A (k_ntt_rows_fwd): lets a's whole
// forward transform run while b is still on its way in.
template <bool kGen>
void ntt_rows_fwd(State& s) {
    const rns::Tables& t = s.t;
    const rns::NttGeom g = s.g;
    const uint32_t blocks = t.N / g.T;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < t.K; ++j) {
        for (uint32_t blk = 0; blk < blocks; ++blk) {
            std::vector<uint32_t> ta(g.T);
            const uint32_t base = blk * g.T;
            uint32_t* a = &s.A[static_cast<size_t>(j) * t.N + base];
            const size_t row = static_cast<size_t>(j) * t.N;
            const uint32_t p = t.p[j];

            phase(g.T, [&](uint32_t i) { ta[i] = a[i]; });
            for (uint32_t k = 0; k < g.logS; ++k) {
                const uint32_t lt = g.logS - 1 - k;
                stage(g, [&](uint32_t bf) { return rns::row_pair(bf, lt, base, g); },
                      [&](const rns::Pair& pr) {
                          rns::fwd_butterfly(ta.data(), pr, rns::twiddles<kGen>(t, j, true), p);
                      });
            }
            phase(g.T, [&](uint32_t i) { a[i] = ta[i]; });
        }
    }
}

// a_ready: A already went through ntt_rows_fwd, so only B's forward stages run.
template <bool kGen>
void ntt_rows_mul(State& s, bool a_ready) {
    const rns::Tables& t = s.t;
    const rns::NttGeom g = s.g;
    const uint32_t blocks = t.N / g.T;
#pragma omp parallel for collapse(2)
    for (int j = 0; j < t.K; ++j) {
        for (uint32_t blk = 0; blk < blocks; ++blk) {
            std::vector<uint32_t> ta(g.T), tb(g.T);
            const uint32_t base = blk * g.T;
            uint32_t* a = &s.A[static_cast<size_t>(j) * t.N + base];
            const uint32_t* b = &s.B[static_cast<size_t>(j) * t.N + base];
            const size_t row = static_cast<size_t>(j) * t.N;
            const uint32_t p = t.p[j];

            phase(g.T, [&](uint32_t i) { ta[i] = a[i]; tb[i] = b[i]; });
            for (uint32_t k = 0; k < g.logS; ++k) {
                const uint32_t lt = g.logS - 1 - k;
                stage(g, [&](uint32_t bf) { return rns::row_pair(bf, lt, base, g); },
                      [&](const rns::Pair& pr) {
                          if (!a_ready) rns::fwd_butterfly(ta.data(), pr, rns::twiddles<kGen>(t, j, true), p);
                          rns::fwd_butterfly(tb.data(), pr, rns::twiddles<kGen>(t, j, true), p);
                      });
            }
            phase(g.T, [&](uint32_t i) { ta[i] = rns::pointwise(ta[i], tb[i], t, j); });
            for (uint32_t lt = 0; lt < g.logS; ++lt) {
                stage(g, [&](uint32_t bf) { return rns::row_pair(bf, lt, base, g); },
                      [&](const rns::Pair& pr) {
                          rns::inv_butterfly(ta.data(), pr, rns::twiddles<kGen>(t, j, false), p);
                      });
            }
            phase(g.T, [&](uint32_t i) { a[i] = ta[i]; });
        }
    }
}

template <int L>
void crt(const State& s, std::vector<uint32_t>& c) {
    const rns::CrtView v = s.plan.crt();
    const uint32_t N = s.t.N;
    const uint32_t blocks = (N + rns::kCrtTile - 1) / rns::kCrtTile;
#pragma omp parallel for
    for (uint32_t blk = 0; blk < blocks; ++blk) {
        // The GPU's shared-memory layout, so the CPU run checks it too.
        std::vector<double> smem((rns::crt_smem_bytes(L) + sizeof(double) - 1) / sizeof(double));
        const rns::CrtSmem m = rns::crt_smem(smem.data());
        const uint32_t kb = blk * rns::kCrtTile;
        const uint32_t nk = std::min(rns::kCrtTile, N - kb);
        phase(32 * rns::kCrtGroups, [&](uint32_t i) { rns::crt_columns<L>(i, s.A.data(), kb, nk, N, v, m); });
        phase(nk, [&](uint32_t k) { rns::crt_combine<L>(k, v, m); });
        phase(nk * L, [&](uint32_t i) { rns::crt_store<L>(i, m.rows, c.data(), kb); });
    }
}

}  // namespace

void* fherma_init(const fherma::Point& p) {
    auto* s = new State{rns::make_plan(p.N, p.q.data.data(), static_cast<int>(p.L)), {}, {}, {}, {}};
    s->gen_tw = rns::use_gen_twiddles(s->plan);
    if (!s->gen_tw) rns::add_full_twiddles(s->plan);
    s->t = tables(s->plan);
    s->g = rns::make_geom(s->plan.logN);
    const size_t residues = static_cast<size_t>(s->plan.K) * s->plan.N;
    s->A.resize(residues);
    s->B.resize(residues);
    return s;
}

fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto* s = static_cast<State*>(state);
    const uint32_t N = s->t.N;
    const int L = s->t.L;

    rns::with_L(L, [&](auto Lc) {
        decompose<decltype(Lc)::value>(*s, in.a.data, s->A);
        decompose<decltype(Lc)::value>(*s, in.b.data, s->B);
    });
    // The GPU order: a's forward transform while b is still arriving, then b's.
    auto ntt = [&](auto gen) {
        constexpr bool kGen = decltype(gen)::value;
        ntt_cols(*s, /*forward=*/true, s->A, nullptr);
        ntt_rows_fwd<kGen>(*s);
        ntt_cols(*s, /*forward=*/true, s->B, nullptr);
        ntt_rows_mul<kGen>(*s, /*a_ready=*/true);
        ntt_cols(*s, /*forward=*/false, s->A, nullptr);
    };
    if (s->gen_tw) {
        ntt(std::true_type{});
    } else {
        ntt(std::false_type{});
    }

    fherma::Outputs out;
    out.c.shape = {static_cast<int64_t>(N), static_cast<int64_t>(L)};
    out.c.data.resize(static_cast<size_t>(N) * L);
    rns::with_L(L, [&](auto Lc) { crt<decltype(Lc)::value>(*s, out.c.data); });
    return out;
}

void fherma_free(void* state) {
    delete static_cast<State*>(state);
}
