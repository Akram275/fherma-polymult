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
#include <vector>

namespace {

struct State {
    rns::Plan plan;
    rns::Tables t;
    rns::NttGeom g;
    std::vector<uint32_t> A, B;     // K x N residues
};

rns::Tables tables(const rns::Plan& pl) {
    return rns::Tables{pl.K, pl.L, pl.N,
                       pl.p.data(), pl.pneg.data(), pl.pow2.data(), pl.pow2_s.data(),
                       pl.psi.data(), pl.ipsi.data(),
                       pl.scale.data(), pl.scale_s.data()};
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
            const rns::Twiddle* w = (forward ? t.psi : t.ipsi) + static_cast<size_t>(j) * t.N;
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
                          rns::fwd_butterfly(ta.data(), pr, t.psi + row, p);
                      });
            }
            phase(g.T, [&](uint32_t i) { a[i] = ta[i]; });
        }
    }
}

// a_ready: A already went through ntt_rows_fwd, so only B's forward stages run.
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
                          if (!a_ready) rns::fwd_butterfly(ta.data(), pr, t.psi + row, p);
                          rns::fwd_butterfly(tb.data(), pr, t.psi + row, p);
                      });
            }
            phase(g.T, [&](uint32_t i) { ta[i] = rns::pointwise(ta[i], tb[i], t, j); });
            for (uint32_t lt = 0; lt < g.logS; ++lt) {
                stage(g, [&](uint32_t bf) { return rns::row_pair(bf, lt, base, g); },
                      [&](const rns::Pair& pr) {
                          rns::inv_butterfly(ta.data(), pr, t.ipsi + row, p);
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
        std::vector<double> smem((rns::crt_smem_bytes(L, v.K) + sizeof(double) - 1) / sizeof(double));
        const rns::CrtSmem m = rns::crt_smem(smem.data(), L, v.K);
        uint32_t* const ytile = m.ytile;
        uint32_t* const part = m.part;
        double* const fracs = m.fracs;
        uint32_t* const tile = m.tile;
        const uint32_t kb = blk * rns::kCrtTile;
        const uint32_t nk = std::min(rns::kCrtTile, N - kb);
        phase(rns::kCrtTile * v.K, [&](uint32_t i) { rns::crt_load(i, ytile, s.A.data(), kb, nk, N); });
        phase(rns::kCrtTile * rns::kCrtGroups,
              [&](uint32_t i) { rns::crt_products<L>(i, ytile, part, fracs, nk, v); });
        phase(nk, [&](uint32_t b) { rns::crt_combine<L>(b, part, fracs, tile, v); });
        phase(nk * L, [&](uint32_t i) { rns::crt_store(i, tile, c.data(), kb, L); });
    }
}

}  // namespace

void* fherma_init(const fherma::Point& p) {
    auto* s = new State{rns::make_plan(p.N, p.q.data.data(), static_cast<int>(p.L)), {}, {}, {}, {}};
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
    ntt_cols(*s, /*forward=*/true, s->A, nullptr);
    ntt_rows_fwd(*s);
    ntt_cols(*s, /*forward=*/true, s->B, nullptr);
    ntt_rows_mul(*s, /*a_ready=*/true);
    ntt_cols(*s, /*forward=*/false, s->A, nullptr);

    fherma::Outputs out;
    out.c.shape = {static_cast<int64_t>(N), static_cast<int64_t>(L)};
    out.c.data.resize(static_cast<size_t>(N) * L);
    rns::with_L(L, [&](auto Lc) { crt<decltype(Lc)::value>(*s, out.c.data); });
    return out;
}

void fherma_free(void* state) {
    delete static_cast<State*>(state);
}
