// A CUDA answer to polymul/negacyclic@1.0.0: RNS + negacyclic NTT + CRT.
//
// The arithmetic is rns.cuh, the schedule is rns_kernels.cuh, and both are
// checked on the CPU by solve_cpu.cpp (see run_tests.sh). This file only puts
// them on the GPU: kernels that deal each phase's items out to threads, and a
// host side that streams the operands in and the answer out in chunks, so the
// copies overlap the decompose at the start and the CRT at the end.
//
//   h2d   a0 a1 .. b0 b1 ..                                      (copy engine)
//   comp     d(a0) d(a1) .. d(b0) .. | NTT fwd, x, inv | crt0 crt1 ..
//   d2h                                                     c0   c1  ..
//
// Tuning knobs, read once in fherma_init (defaults in brackets):
//   FHERMA_IN=staged|register   how a and b reach the GPU            [staged]
//   FHERMA_OUT=staged|register  how c comes back                     [staged]
//     staged:   memcpy through pinned buffers allocated at init, chunk by chunk
//     register: cudaHostRegister the caller's vectors for the call, DMA directly
//     mapped:   memcpy through the same pinned buffers, mapped into the device:
//               decompose reads its limbs from host memory over PCIe and CRT
//               writes its limbs there, with no copy, event or stream hop per
//               chunk (each costs ~25 us of API calls on rst-goose)
//   FHERMA_CHUNKS=n             pipeline depth, input side: n equal chunks [2]
//   FHERMA_OUT_CHUNKS=n         pipeline depth, output side          [FHERMA_CHUNKS]
//   FHERMA_A_CHUNKS=w,w,..      a's chunks as relative weights (overrides FHERMA_CHUNKS)
//   FHERMA_B_CHUNKS=w,w,..      b's chunks, likewise
//   FHERMA_C_CHUNKS=w,w,..      c's chunks (overrides FHERMA_OUT_CHUNKS) [1,2,1 if neither is set]
//     A small first chunk of a starts the first DMA sooner, a small last chunk
//     of b leaves less decompose after the last DMA; on the output side, a
//     small first chunk starts the first D2H sooner and a small last one leaves
//     less to copy out after the last.
//   FHERMA_THREADS=n            host threads for the staged memcpys   [min(16, cores / 2)]
//   FHERMA_SPIN_MS=n            copy threads keep spinning n ms after a call [20]
//     Waking sleeping threads costs ~150-200 us per call on rst-goose, and
//     main.cpp leaves ~13 ms between calls; 20 hides the wake-up (~11 % of a
//     call) at the price of the workers busy-waiting through that gap.
//   FHERMA_PROFILE=1            per-phase GPU times on stderr, every call
//
// Host side of a staged call: each input chunk is copied into pinned memory by
// all the copy threads, then sent; once everything is enqueued, the calling
// thread sizes the output vector (a zero fill) while the GPU works, and copies
// each output chunk out of pinned memory as soon as it has landed. Measured on
// rst-goose, one thread copies ~12 GB/s and 16 threads ~250 GB/s (cache-hot),
// against ~400 us for a cudaHostRegister of one operand. Each chunk costs
// ~25 us of CUDA API calls there, so the pipeline is shallow by default.
#include "fherma.h"
#include "rns_kernels.cuh"
#include "rns_plan.h"

#include <cuda_runtime.h>
#include <malloc.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kThreads = 256;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

// ---------------------------------------------------------------------------
// Kernels. Each loop is one phase of rns_kernels.cuh, dealt out to the block.

__global__ void __launch_bounds__(kThreads)
k_decompose(const uint32_t* __restrict__ limbs, uint32_t* __restrict__ res,
            uint32_t k0, uint32_t count, rns::Tables t) {
    __shared__ uint32_t tile[rns::kDecTile * (rns::kMaxL + 1)];
    const uint32_t kb = k0 + blockIdx.x * rns::kDecTile;
    const uint32_t nk = min(rns::kDecTile, k0 + count - kb);
    for (uint32_t i = threadIdx.x; i < nk * t.L; i += blockDim.x) {
        rns::dec_load(i, tile, limbs, kb, t.L);
    }
    __syncthreads();
    for (uint32_t b = threadIdx.x; b < rns::kDecTile * t.K; b += blockDim.x) {
        rns::dec_compute(b, tile, res, kb, nk, t);
    }
}

// Column pass: forward on A and B together (sharing the twiddle loads), or
// inverse on A alone (B == nullptr).
template <bool kForward>
__global__ void __launch_bounds__(kThreads)
k_ntt_cols(uint32_t* __restrict__ A, uint32_t* __restrict__ B, rns::Tables t, rns::NttGeom g) {
    __shared__ uint32_t ta[rns::kNttTile], tb[rns::kNttTile];
    const int j = blockIdx.y;
    const size_t row = static_cast<size_t>(j) * t.N;
    const uint32_t r0 = blockIdx.x << g.logC;
    uint32_t* a = A + row;
    uint32_t* b = B ? B + row : nullptr;
    const uint32_t* w = (kForward ? t.psi : t.ipsi) + row;
    const uint32_t* ws = (kForward ? t.psi_s : t.ipsi_s) + row;
    const uint32_t p = t.p[j];

    for (uint32_t i = threadIdx.x; i < g.T; i += blockDim.x) {
        const uint32_t k = rns::col_global(i, r0, g);
        ta[i] = a[k];
        if (b) tb[i] = b[k];
    }
    __syncthreads();
    for (uint32_t s = 0; s < g.logR; ++s) {
        const uint32_t lt = kForward ? g.logR - 1 - s : s;
        for (uint32_t bf = threadIdx.x; bf < g.T / 2; bf += blockDim.x) {
            const rns::Pair pr = rns::col_pair(bf, lt, g);
            if (kForward) {
                rns::fwd_butterfly(ta, pr, w, ws, p);
                if (b) rns::fwd_butterfly(tb, pr, w, ws, p);
            } else {
                rns::inv_butterfly(ta, pr, w, ws, p);
            }
        }
        __syncthreads();
    }
    for (uint32_t i = threadIdx.x; i < g.T; i += blockDim.x) {
        const uint32_t k = rns::col_global(i, r0, g);
        a[k] = ta[i];
        if (b) b[k] = tb[i];
    }
}

// Row pass: forward on A and B, pointwise into A, inverse on A, one tile.
__global__ void __launch_bounds__(kThreads)
k_ntt_rows_mul(uint32_t* __restrict__ A, const uint32_t* __restrict__ B, rns::Tables t, rns::NttGeom g) {
    __shared__ uint32_t ta[rns::kNttTile], tb[rns::kNttTile];
    const int j = blockIdx.y;
    const size_t row = static_cast<size_t>(j) * t.N;
    const uint32_t base = blockIdx.x * g.T;
    uint32_t* a = A + row + base;
    const uint32_t* b = B + row + base;
    const uint32_t p = t.p[j];

    for (uint32_t i = threadIdx.x; i < g.T; i += blockDim.x) {
        ta[i] = a[i];
        tb[i] = b[i];
    }
    __syncthreads();
    for (uint32_t s = 0; s < g.logS; ++s) {
        const uint32_t lt = g.logS - 1 - s;
        for (uint32_t bf = threadIdx.x; bf < g.T / 2; bf += blockDim.x) {
            const rns::Pair pr = rns::row_pair(bf, lt, base, g);
            rns::fwd_butterfly(ta, pr, t.psi + row, t.psi_s + row, p);
            rns::fwd_butterfly(tb, pr, t.psi + row, t.psi_s + row, p);
        }
        __syncthreads();
    }
    for (uint32_t i = threadIdx.x; i < g.T; i += blockDim.x) {
        ta[i] = rns::pointwise(ta[i], tb[i], t, j);
    }
    __syncthreads();
    for (uint32_t lt = 0; lt < g.logS; ++lt) {
        for (uint32_t bf = threadIdx.x; bf < g.T / 2; bf += blockDim.x) {
            rns::inv_butterfly(ta, rns::row_pair(bf, lt, base, g), t.ipsi + row, t.ipsi_s + row, p);
        }
        __syncthreads();
    }
    for (uint32_t i = threadIdx.x; i < g.T; i += blockDim.x) a[i] = ta[i];
}

constexpr uint32_t kCrtThreads = rns::kCrtTile * rns::kCrtGroups;

template <int L>
__global__ void __launch_bounds__(kCrtThreads)
k_crt(const uint32_t* __restrict__ Y, uint32_t* __restrict__ c,
      uint32_t k0, uint32_t count, uint32_t N, rns::CrtView v) {
    __shared__ uint32_t ytile[rns::kCrtTile * rns::kMaxK];
    __shared__ uint32_t part[rns::kCrtGroups * (L + 2) * rns::kCrtTile];
    __shared__ double fracs[rns::kCrtGroups * rns::kCrtTile];
    __shared__ uint32_t tile[rns::kCrtTile * (L + 1)];
    const uint32_t kb = k0 + blockIdx.x * rns::kCrtTile;
    const uint32_t nk = min(rns::kCrtTile, k0 + count - kb);
    for (uint32_t i = threadIdx.x; i < rns::kCrtTile * v.K; i += blockDim.x) {
        rns::crt_load(i, ytile, Y, kb, nk, N);
    }
    __syncthreads();
    rns::crt_products<L>(threadIdx.x, ytile, part, fracs, nk, v);
    __syncthreads();
    if (threadIdx.x < nk) rns::crt_combine<L>(threadIdx.x, part, fracs, tile, v);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < nk * L; i += blockDim.x) {
        rns::crt_store(i, tile, c, kb, L);
    }
}

// ---------------------------------------------------------------------------
// Host side.

enum class Copy { staged, registered, mapped };

Copy copy_mode(const char* var) {
    const char* v = std::getenv(var);
    if (!v || !std::strcmp(v, "staged")) return Copy::staged;
    if (!std::strcmp(v, "register")) return Copy::registered;
    if (!std::strcmp(v, "mapped")) return Copy::mapped;
    throw std::runtime_error(std::string(var) + " must be staged, register or mapped");
}

// memcpy split into 64 KB pieces that the calling thread and a few workers
// claim from a shared counter. The workers spin while a call is running (copies
// come every few microseconds) and sleep on a condition variable between calls
// (main.cpp spends ~13 ms on files between calls). Waking them takes ~150 us
// here, so the caller never waits for them: it claims pieces itself from the
// start, and workers join whenever they are up.
//
// next_ is (job << 32) | next piece. A job is opened by writing its
// parameters while next_ is closed (piece index kClosed) and then storing
// (job << 32) | 0; a piece is claimed by a compare-and-swap on next_, which
// fails if the job has changed since it was read, so a late worker can never
// claim a piece of a job with the parameters of another. The caller returns
// once finished_ has counted every piece, and only then can the parameters
// change, so no worker is ever inside a piece when they do.
class CopyPool {
public:
    CopyPool(int workers, int spin_ms) : spin_(std::chrono::milliseconds(spin_ms)) {
        for (int w = 0; w < workers; ++w) threads_.emplace_back([this] { work(); });
    }
    ~CopyPool() {
        {
            std::lock_guard<std::mutex> g(m_);
            quit_.store(true);
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }

    void wake() {
        {
            std::lock_guard<std::mutex> g(m_);
            awake_.store(true);
        }
        cv_.notify_all();
    }
    void rest() {
        rested_.store(Clock::now().time_since_epoch().count());
        awake_.store(false);
    }

    void copy(void* dst, const void* src, size_t n) {
        if (threads_.empty() || n <= kPiece) {
            std::memcpy(dst, src, n);
            return;
        }
        const uint64_t job = (next_.load(std::memory_order_relaxed) >> 32) + 1;
        next_.store(job << 32 | kClosed, std::memory_order_relaxed);
        dst_ = static_cast<char*>(dst);
        src_ = static_cast<const char*>(src);
        n_ = n;
        // Release: a worker that reads this value then cannot win a CAS on the old job.
        pieces_.store(static_cast<uint32_t>((n + kPiece - 1) / kPiece), std::memory_order_release);
        finished_.store(0, std::memory_order_relaxed);
        {
            // Under the lock, so a worker deciding to sleep cannot miss it.
            std::lock_guard<std::mutex> g(m_);
            next_.store(job << 32, std::memory_order_release);
        }
        if (!awake_.load()) cv_.notify_all();
        while (claim()) {
        }
        const uint32_t pieces = pieces_.load(std::memory_order_relaxed);
        while (finished_.load(std::memory_order_acquire) < pieces) pause();
    }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr size_t kPiece = 64 << 10;
    static constexpr uint64_t kClosed = 0xffffffffu;

    static void pause() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }

    // Claims and copies one piece of the open job; false if there is none left.
    bool claim() {
        uint64_t v = next_.load(std::memory_order_acquire);
        for (;;) {
            const uint32_t i = static_cast<uint32_t>(v);
            if (i >= pieces_.load(std::memory_order_acquire)) return false;   // kClosed too
            if (next_.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) break;
        }
        const size_t lo = static_cast<size_t>(static_cast<uint32_t>(v)) * kPiece;
        std::memcpy(dst_ + lo, src_ + lo, std::min(kPiece, n_ - lo));
        finished_.fetch_add(1, std::memory_order_release);
        return true;
    }

    void work() {
        for (;;) {
            if (claim()) continue;
            if (quit_.load()) return;
            if (awake_.load() || Clock::now() < Clock::time_point(Clock::duration(rested_.load())) + spin_) {
                pause();
                continue;
            }
            std::unique_lock<std::mutex> g(m_);
            const uint64_t seen = next_.load();
            cv_.wait(g, [&] { return quit_.load() || awake_.load() || next_.load() != seen; });
        }
    }

    const Clock::duration spin_;
    std::atomic<Clock::rep> rested_{0};
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<bool> quit_{false}, awake_{false};
    std::atomic<uint64_t> next_{kClosed};
    std::atomic<uint32_t> pieces_{0}, finished_{0};
    char* dst_ = nullptr;
    const char* src_ = nullptr;
    size_t n_ = 0;
};

// One thread for one background job per call (sizing the output vector while
// the calling thread copies the inputs), spinning and sleeping like CopyPool.
class Helper {
public:
    explicit Helper(int spin_ms) : spin_(std::chrono::milliseconds(spin_ms)), thread_([this] { work(); }) {}
    ~Helper() {
        {
            std::lock_guard<std::mutex> g(m_);
            quit_.store(true);
        }
        cv_.notify_all();
        thread_.join();
    }

    // Starts fn on the helper. The caller must wait() before starting another.
    void run(std::function<void()> fn) {
        job_ = std::move(fn);
        {
            std::lock_guard<std::mutex> g(m_);
            pending_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
    }
    void wait() {
        while (pending_.load(std::memory_order_acquire)) pause();
    }

private:
    using Clock = std::chrono::steady_clock;

    static void pause() {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }

    void work() {
        auto idle_since = Clock::now();
        for (;;) {
            if (pending_.load(std::memory_order_acquire)) {
                job_();
                job_ = nullptr;
                idle_since = Clock::now();
                pending_.store(false, std::memory_order_release);
                continue;
            }
            if (quit_.load()) return;
            if (Clock::now() < idle_since + spin_) {
                pause();
                continue;
            }
            std::unique_lock<std::mutex> g(m_);
            cv_.wait(g, [&] { return quit_.load() || pending_.load(); });
        }
    }

    const Clock::duration spin_;
    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<bool> quit_{false}, pending_{false};
    std::function<void()> job_;
    std::thread thread_;   // last: starts after everything it reads
};

int copy_threads() {
    if (const char* v = std::getenv("FHERMA_THREADS")) return std::max(1, std::atoi(v));
    const int cores = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, std::min(16, cores / 2));
}

struct State {
    rns::Plan plan;
    rns::Tables t{};
    rns::NttGeom g{};
    rns::CrtView crt{};

    std::vector<void*> device;              // everything cudaMalloc'd, for free
    uint32_t *d_la = nullptr, *d_lb = nullptr, *d_c = nullptr;   // N x L limbs
    uint32_t *d_A = nullptr, *d_B = nullptr;                     // K x N residues
    uint32_t *h_a = nullptr, *h_b = nullptr, *h_c = nullptr;     // pinned staging
    uint32_t *m_a = nullptr, *m_b = nullptr, *m_c = nullptr;     // the same, as the device sees them

    cudaStream_t comp = nullptr, h2d = nullptr, d2h = nullptr;
    struct Span {
        uint32_t k0, count;
    };
    std::vector<Span> a_chunks, b_chunks, c_chunks;   // coefficient ranges
    std::vector<cudaEvent_t> in_a, in_b, crt_done, out_done;
    Copy in_mode = Copy::staged, out_mode = Copy::staged;
    std::unique_ptr<CopyPool> pool;
    std::unique_ptr<Helper> helper;

    bool profile = false;
    cudaEvent_t prof[5] = {};
};

// Relative chunk weights from var ("1,3"), or n equal ones.
std::vector<double> weights(const char* var, int n) {
    std::vector<double> w;
    if (const char* v = std::getenv(var)) {
        for (const char* p = v; *p;) {
            char* end = nullptr;
            const double x = std::strtod(p, &end);
            if (end == p || x <= 0) throw std::runtime_error(std::string(var) + ": positive weights, comma separated");
            w.push_back(x);
            p = *end == ',' ? end + 1 : end;
        }
    }
    if (w.empty()) w.assign(static_cast<size_t>(n), 1.0);
    return w;
}

// Coefficient ranges in proportion to w. Every chunk but the last starts and
// ends on a multiple of both block tiles; empty chunks are dropped.
std::vector<State::Span> spans(uint32_t N, const std::vector<double>& w) {
    const uint32_t align = std::max(rns::kDecTile, rns::kCrtTile);
    double total = 0;
    for (double x : w) total += x;
    std::vector<State::Span> out;
    double acc = 0;
    uint32_t k0 = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        acc += w[i];
        uint32_t k1 = i + 1 == w.size() ? N
                                         : static_cast<uint32_t>(std::lround(acc / total * N / align)) * align;
        k1 = std::min(std::max(k1, k0), N);
        if (k1 > k0) out.push_back(State::Span{k0, k1 - k0});
        k0 = k1;
    }
    return out;
}

template <class T>
T* device_alloc(State& s, size_t count) {
    void* p = nullptr;
    check(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
    s.device.push_back(p);
    return static_cast<T*>(p);
}

template <class T>
const T* upload(State& s, const std::vector<T>& v) {
    T* d = device_alloc<T>(s, v.size());
    check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

// Pins a caller's buffer for one call, and unpins it however the call ends.
struct Registration {
    void* at = nullptr;
    Registration(const void* p, size_t bytes) {
        check(cudaHostRegister(const_cast<void*>(p), bytes, cudaHostRegisterDefault), "cudaHostRegister");
        at = const_cast<void*>(p);
    }
    ~Registration() {
        if (at) cudaHostUnregister(at);
    }
    Registration(const Registration&) = delete;
    Registration& operator=(const Registration&) = delete;
};

void launch_decompose(State& s, const uint32_t* limbs, uint32_t* res, uint32_t k0, uint32_t count) {
    const uint32_t blocks = (count + rns::kDecTile - 1) / rns::kDecTile;
    k_decompose<<<blocks, kThreads, 0, s.comp>>>(limbs, res, k0, count, s.t);
}

void launch_ntt(State& s) {
    const rns::NttGeom& g = s.g;
    const dim3 cols(1u << (g.logS - g.logC), s.t.K);
    const dim3 rows(s.t.N / g.T, s.t.K);
    if (g.logR) k_ntt_cols<true><<<cols, kThreads, 0, s.comp>>>(s.d_A, s.d_B, s.t, g);
    k_ntt_rows_mul<<<rows, kThreads, 0, s.comp>>>(s.d_A, s.d_B, s.t, g);
    if (g.logR) k_ntt_cols<false><<<cols, kThreads, 0, s.comp>>>(s.d_A, nullptr, s.t, g);
}

void launch_crt(State& s, uint32_t* c, uint32_t k0, uint32_t count) {
    const uint32_t blocks = (count + rns::kCrtTile - 1) / rns::kCrtTile;
    rns::with_L(s.t.L, [&](auto Lc) {
        constexpr int L = decltype(Lc)::value;
        k_crt<L><<<blocks, kCrtThreads, 0, s.comp>>>(s.d_A, c, k0, count, s.t.N, s.crt);
    });
}

using HostClock = std::chrono::steady_clock;

// GPU phases from the events, then the host's own timeline: t[0] call start,
// t[1] inputs sent, t[2] all enqueued, t[3] output sized, t[4] output copied.
void report_profile(State& s, const HostClock::time_point (&t)[5]) {
    float ms[4];
    for (int i = 0; i < 4; ++i) cudaEventElapsedTime(&ms[i], s.prof[i], s.prof[i + 1]);
    auto us = [&](int i) { return std::chrono::duration<double, std::micro>(t[i + 1] - t[i]).count(); };
    std::fprintf(stderr, "gpu us: in+decompose %.1f  ntt %.1f  crt %.1f  out-tail %.1f | "
                 "host us: send %.1f  enqueue %.1f  size-out %.1f  drain %.1f\n",
                 ms[0] * 1e3, ms[1] * 1e3, ms[2] * 1e3, ms[3] * 1e3, us(0), us(1), us(2), us(3));
}

}  // namespace

void* fherma_init(const fherma::Point& p) {
    // Load every kernel now rather than on first launch, inside the timed call.
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    // Keep large blocks on the heap once freed, so the vectors main.cpp and
    // fherma_run allocate per case reuse resident pages instead of fresh mmaps
    // that fault on first touch.
    mallopt(M_MMAP_THRESHOLD, 32 << 20);    // glibc's maximum
    mallopt(M_TRIM_THRESHOLD, 512 << 20);

    auto* s = new State{};
    try {
        s->plan = rns::make_plan(p.N, p.q.data.data(), static_cast<int>(p.L));
        const rns::Plan& pl = s->plan;
        s->g = rns::make_geom(pl.logN);
        s->in_mode = copy_mode("FHERMA_IN");
        s->out_mode = copy_mode("FHERMA_OUT");
        s->profile = std::getenv("FHERMA_PROFILE") != nullptr;
        const char* spin = std::getenv("FHERMA_SPIN_MS");
        const int spin_ms = spin ? std::max(0, std::atoi(spin)) : 20;
        s->pool.reset(new CopyPool(copy_threads() - 1, spin_ms));
        s->helper.reset(new Helper(spin_ms));

        s->t = rns::Tables{pl.K, pl.L, pl.N,
                           upload(*s, pl.p), upload(*s, pl.pneg),
                           upload(*s, pl.pow2), upload(*s, pl.pow2_s),
                           upload(*s, pl.psi), upload(*s, pl.psi_s),
                           upload(*s, pl.ipsi), upload(*s, pl.ipsi_s),
                           upload(*s, pl.scale), upload(*s, pl.scale_s)};
        s->crt = rns::CrtView{pl.K, rns::Modulus{upload(*s, pl.q), pl.L, pl.q_top},
                              upload(*s, pl.p_recip), upload(*s, pl.M), upload(*s, pl.negvP)};

        const size_t words = static_cast<size_t>(pl.N) * pl.L;
        const size_t residues = static_cast<size_t>(pl.K) * pl.N;
        s->d_la = device_alloc<uint32_t>(*s, words);
        s->d_lb = device_alloc<uint32_t>(*s, words);
        s->d_c = device_alloc<uint32_t>(*s, words);
        s->d_A = device_alloc<uint32_t>(*s, residues);
        s->d_B = device_alloc<uint32_t>(*s, residues);
        for (auto hm : {std::make_pair(&s->h_a, &s->m_a), std::make_pair(&s->h_b, &s->m_b),
                        std::make_pair(&s->h_c, &s->m_c)}) {
            check(cudaHostAlloc(reinterpret_cast<void**>(hm.first), words * sizeof(uint32_t),
                                cudaHostAllocMapped), "pinned");
            check(cudaHostGetDevicePointer(reinterpret_cast<void**>(hm.second), *hm.first, 0), "mapped");
        }

        check(cudaStreamCreateWithFlags(&s->comp, cudaStreamNonBlocking), "stream");
        check(cudaStreamCreateWithFlags(&s->h2d, cudaStreamNonBlocking), "stream");
        check(cudaStreamCreateWithFlags(&s->d2h, cudaStreamNonBlocking), "stream");

        const char* env_in = std::getenv("FHERMA_CHUNKS");
        const int in_n = std::max(1, env_in ? std::atoi(env_in) : 2);
        const char* env_out = std::getenv("FHERMA_OUT_CHUNKS");
        const int out_n = env_out ? std::max(1, std::atoi(env_out)) : in_n;
        s->a_chunks = spans(pl.N, weights("FHERMA_A_CHUNKS", in_n));
        s->b_chunks = spans(pl.N, weights("FHERMA_B_CHUNKS", in_n));
        s->c_chunks = spans(pl.N, env_out || std::getenv("FHERMA_C_CHUNKS")
                                      ? weights("FHERMA_C_CHUNKS", out_n)
                                      : std::vector<double>{1, 2, 1});   // 2026-10-14: ~15 us better than 1,1
        s->in_a.resize(s->a_chunks.size());
        s->in_b.resize(s->b_chunks.size());
        for (auto* evs : {&s->crt_done, &s->out_done}) evs->resize(s->c_chunks.size());
        for (auto* evs : {&s->in_a, &s->in_b, &s->crt_done, &s->out_done}) {
            for (auto& e : *evs) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
        }
        for (auto& e : s->prof) check(cudaEventCreate(&e), "event");

        // One untimed run, so nothing the first case pays for is a first time.
        fherma::Inputs warm;
        warm.a.shape = warm.b.shape = {static_cast<int64_t>(pl.N), static_cast<int64_t>(pl.L)};
        warm.a.data.assign(words, 0);
        warm.b.data.assign(words, 0);
        const bool profile = s->profile;
        s->profile = false;
        fherma_run(s, warm);
        s->profile = profile;
    } catch (...) {
        fherma_free(s);
        throw;
    }
    return s;
}

fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto* s = static_cast<State*>(state);
    const uint32_t N = s->t.N;
    const uint32_t L = static_cast<uint32_t>(s->t.L);
    const size_t words = static_cast<size_t>(N) * L;
    const size_t bytes = words * sizeof(uint32_t);
    if (in.a.data.size() != words || in.b.data.size() != words) {
        throw std::runtime_error("operand size does not match the point");
    }

    fherma::Outputs out;
    out.c.shape = {static_cast<int64_t>(N), static_cast<int64_t>(L)};

    const bool reg_in = s->in_mode == Copy::registered;
    const bool reg_out = s->out_mode == Copy::registered;
    const bool map_in = s->in_mode == Copy::mapped;
    const bool map_out = s->out_mode == Copy::mapped;
    std::vector<std::unique_ptr<Registration>> hold;
    if (reg_in) {
        hold.emplace_back(new Registration(in.a.data.data(), bytes));
        hold.emplace_back(new Registration(in.b.data.data(), bytes));
    }
    HostClock::time_point t[5];
    t[0] = HostClock::now();
    if (reg_out) {
        out.c.data.resize(words);
        hold.emplace_back(new Registration(out.c.data.data(), bytes));
    }
    s->pool->wake();
    // Size the answer on the helper while this thread copies the inputs, so the
    // zero fill (~350 us here) is done before the first output chunk lands.
    // The guard waits for it however the call ends: it writes into `out`.
    struct Sized {
        Helper& h;
        bool started;
        ~Sized() {
            if (started) h.wait();
        }
    } sized{*s->helper, !reg_out};
    if (sized.started) s->helper->run([&out, words] { out.c.data.resize(words); });
    struct Rest {
        CopyPool& p;
        ~Rest() { p.rest(); }
    } rest{*s->pool};

    if (s->profile) cudaEventRecord(s->prof[0], s->comp);

    // In: each chunk's copy, then its decompose once the copy has landed.
    // Mapped: the decompose reads the pinned buffer itself; the launch comes
    // after the host's writes, which is all the ordering it needs.
    auto send = [&](const uint32_t* src, uint32_t* pin, uint32_t* mapped, uint32_t* d_limbs,
                    uint32_t* d_res, const std::vector<State::Span>& chunks,
                    std::vector<cudaEvent_t>& landed) {
        for (uint32_t ch = 0; ch < chunks.size(); ++ch) {
            const uint32_t k0 = chunks[ch].k0;
            const uint32_t count = chunks[ch].count;
            const size_t off = static_cast<size_t>(k0) * L;
            const size_t n = static_cast<size_t>(count) * L * sizeof(uint32_t);
            if (map_in) {
                s->pool->copy(pin + off, src + off, n);
                launch_decompose(*s, mapped, d_res, k0, count);
                continue;
            }
            const uint32_t* from = src + off;
            if (!reg_in) {
                s->pool->copy(pin + off, src + off, n);
                from = pin + off;
            }
            check(cudaMemcpyAsync(d_limbs + off, from, n, cudaMemcpyHostToDevice, s->h2d), "copy in");
            check(cudaEventRecord(landed[ch], s->h2d), "event");
            check(cudaStreamWaitEvent(s->comp, landed[ch], 0), "wait");
            launch_decompose(*s, d_limbs, d_res, k0, count);
        }
    };
    send(in.a.data.data(), s->h_a, s->m_a, s->d_la, s->d_A, s->a_chunks, s->in_a);
    send(in.b.data.data(), s->h_b, s->m_b, s->d_lb, s->d_B, s->b_chunks, s->in_b);
    t[1] = HostClock::now();
    if (s->profile) cudaEventRecord(s->prof[1], s->comp);

    launch_ntt(*s);
    if (s->profile) cudaEventRecord(s->prof[2], s->comp);

    // Out: each chunk's CRT, then its copy back once the CRT is done.
    // Mapped: the CRT writes the pinned buffer itself, and its end is the copy's.
    uint32_t* dst = reg_out ? out.c.data.data() : s->h_c;
    for (uint32_t ch = 0; ch < s->c_chunks.size(); ++ch) {
        const uint32_t k0 = s->c_chunks[ch].k0;
        const uint32_t count = s->c_chunks[ch].count;
        const size_t off = static_cast<size_t>(k0) * L;
        if (map_out) {
            launch_crt(*s, s->m_c, k0, count);
            check(cudaEventRecord(s->out_done[ch], s->comp), "event");
            continue;
        }
        launch_crt(*s, s->d_c, k0, count);
        check(cudaEventRecord(s->crt_done[ch], s->comp), "event");
        check(cudaStreamWaitEvent(s->d2h, s->crt_done[ch], 0), "wait");
        check(cudaMemcpyAsync(dst + off, s->d_c + off, static_cast<size_t>(count) * L * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, s->d2h), "copy out");
        check(cudaEventRecord(s->out_done[ch], s->d2h), "event");
    }
    if (s->profile) cudaEventRecord(s->prof[3], s->comp);
    check(cudaGetLastError(), "launch");
    t[2] = HostClock::now();

    if (reg_out) {
        check(cudaStreamSynchronize(s->d2h), "copy out");
    } else {
        s->helper->wait();
        t[3] = HostClock::now();
        for (uint32_t ch = 0; ch < s->c_chunks.size(); ++ch) {
            const size_t off = static_cast<size_t>(s->c_chunks[ch].k0) * L;
            const size_t n = static_cast<size_t>(s->c_chunks[ch].count) * L;
            check(cudaEventSynchronize(s->out_done[ch]), "copy out");
            s->pool->copy(out.c.data.data() + off, s->h_c + off, n * sizeof(uint32_t));
        }
    }

    t[4] = HostClock::now();
    if (reg_out) t[3] = t[2];
    if (s->profile) {
        check(cudaStreamWaitEvent(s->comp, s->out_done.back(), 0), "wait");
        cudaEventRecord(s->prof[4], s->comp);
        cudaEventSynchronize(s->prof[4]);
        report_profile(*s, t);
    }
    return out;
}

void fherma_free(void* state) {
    auto* s = static_cast<State*>(state);
    if (!s) return;
    cudaDeviceSynchronize();
    for (auto* evs : {&s->in_a, &s->in_b, &s->crt_done, &s->out_done}) {
        for (auto e : *evs) cudaEventDestroy(e);
    }
    for (auto e : s->prof) {
        if (e) cudaEventDestroy(e);
    }
    for (auto st : {s->comp, s->h2d, s->d2h}) {
        if (st) cudaStreamDestroy(st);
    }
    for (void* p : s->device) cudaFree(p);
    cudaFreeHost(s->h_a);
    cudaFreeHost(s->h_b);
    cudaFreeHost(s->h_c);
    delete s;
}
