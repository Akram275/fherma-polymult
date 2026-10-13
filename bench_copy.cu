// The transfer floor for polymul/negacyclic at the challenge point.
//
//     nvcc -O2 -std=c++17 -o bench_copy bench_copy.cu && ./bench_copy [N] [L]
//
// fherma_run starts with a and b in host std::vectors and ends with c in one,
// so every answer pays for 2*N*L u32 host-to-device and N*L device-to-host,
// plus allocating the output vector. No kernel runs here: whatever this
// measures is time no amount of arithmetic can win back.
//
// Each mode is timed like main.cpp times fherma_run (steady_clock around the
// whole thing, synchronised at the end), and the median of many runs is shown.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

double median_us(const std::function<void()>& body, int runs = 50) {
    for (int i = 0; i < 5; ++i) body();  // warm up
    std::vector<double> t;
    for (int i = 0; i < runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        body();
        const std::chrono::duration<double, std::micro> took =
            std::chrono::steady_clock::now() - start;
        t.push_back(took.count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const size_t N = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 32768;
    const size_t L = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 28;
    const size_t words = N * L;
    const size_t bytes = words * sizeof(uint32_t);

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "device");
    std::printf("%s, %d async copy engine(s)\n", prop.name, prop.asyncEngineCount);
    std::printf("N=%zu L=%zu: %.2f MB per operand, %.2f MB per call\n\n",
                N, L, bytes / 1e6, 3 * bytes / 1e6);

    // Inputs as the harness hands them over: ordinary pageable vectors.
    std::vector<uint32_t> a(words, 1), b(words, 2);

    uint32_t *d_a, *d_b, *d_c;
    check(cudaMalloc(&d_a, bytes), "malloc a");
    check(cudaMalloc(&d_b, bytes), "malloc b");
    check(cudaMalloc(&d_c, bytes), "malloc c");
    check(cudaMemset(d_c, 0, bytes), "memset c");

    uint32_t *h_a, *h_b, *h_c;  // pinned staging, as fherma_init could set up
    check(cudaMallocHost(&h_a, bytes), "host a");
    check(cudaMallocHost(&h_b, bytes), "host b");
    check(cudaMallocHost(&h_c, bytes), "host c");

    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");

    // 1. What the baseline does: plain cudaMemcpy from and to pageable memory.
    const double pageable = median_us([&] {
        cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
        std::vector<uint32_t> c(words);
        cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    });

    // 2. DMA alone, pinned on both sides: the floor, not reachable as such,
    //    because the data starts and ends in pageable vectors.
    const double pinned = median_us([&] {
        cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
    });

    // 3. Staged: the CPU copies chunks into pinned buffers while the DMA engine
    //    moves the chunks before them; the same in reverse for c.
    const size_t chunk = 256 * 1024;
    std::vector<cudaEvent_t> done;
    const size_t chunks = (bytes + chunk - 1) / chunk;
    done.resize(chunks);
    for (auto& e : done) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
    auto staged_in = [&](const uint32_t* src, uint32_t* pin, uint32_t* dev) {
        for (size_t k = 0; k < chunks; ++k) {
            const size_t off = k * chunk, n = std::min(chunk, bytes - off);
            std::memcpy(reinterpret_cast<char*>(pin) + off,
                        reinterpret_cast<const char*>(src) + off, n);
            cudaMemcpyAsync(reinterpret_cast<char*>(dev) + off,
                            reinterpret_cast<char*>(pin) + off, n,
                            cudaMemcpyHostToDevice, s);
        }
    };
    const double staged = median_us([&] {
        staged_in(a.data(), h_a, d_a);
        staged_in(b.data(), h_b, d_b);
        for (size_t k = 0; k < chunks; ++k) {
            const size_t off = k * chunk, n = std::min(chunk, bytes - off);
            cudaMemcpyAsync(reinterpret_cast<char*>(h_c) + off,
                            reinterpret_cast<char*>(d_c) + off, n,
                            cudaMemcpyDeviceToHost, s);
            cudaEventRecord(done[k], s);
        }
        std::vector<uint32_t> c(words);
        for (size_t k = 0; k < chunks; ++k) {
            const size_t off = k * chunk, n = std::min(chunk, bytes - off);
            cudaEventSynchronize(done[k]);
            std::memcpy(reinterpret_cast<char*>(c.data()) + off,
                        reinterpret_cast<char*>(h_c) + off, n);
        }
    });

    // 4. Pin the caller's vectors in place for the duration of the call.
    const double registered = median_us([&] {
        cudaHostRegister(a.data(), bytes, cudaHostRegisterDefault);
        cudaHostRegister(b.data(), bytes, cudaHostRegisterDefault);
        std::vector<uint32_t> c(words);
        cudaHostRegister(c.data(), bytes, cudaHostRegisterDefault);
        cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        cudaHostUnregister(c.data());
        cudaHostUnregister(a.data());
        cudaHostUnregister(b.data());
    });

    // And the part that is pure host work: the output vector itself.
    const double alloc = median_us([&] {
        std::vector<uint32_t> c(words);
        asm volatile("" : : "r"(c.data()) : "memory");
    });

    check(cudaGetLastError(), "copies");
    const double mb = 3 * bytes / 1e6;
    std::printf("%-34s %9s %9s\n", "mode", "median", "GB/s");
    auto row = [&](const char* name, double us) {
        std::printf("%-34s %7.1f us %8.1f\n", name, us, mb / us * 1e3);
    };
    row("1 pageable cudaMemcpy (baseline)", pageable);
    row("2 pinned DMA only (floor)", pinned);
    row("3 staged through pinned chunks", staged);
    row("4 cudaHostRegister per call", registered);
    std::printf("%-34s %7.1f us\n", "  output vector allocation alone", alloc);
    return 0;
}
