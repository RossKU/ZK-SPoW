// ZK-SPoW GPU Benchmark — Poseidon2 Width-24 / M31
//
// Measures complementary bottleneck thesis:
//   Phase 1: Pure PoW throughput        (compute-bound, register-only)
//   Phase 2: Merkle/LDE throughput      (memory-bound, global memory R/W)
//   Phase 3: Symbiotic = both on same GPU simultaneously
//
// If Phase3_total ≈ Phase1_rate, the paper's claim holds:
//   memory-bound STARK work doesn't steal compute from PoW.
//
// Build:  nvcc -O3 -arch=sm_80 bench.cu -o bench
// Run:    ./bench [difficulty] [log_array_size] [seconds]
// Example: ./bench 24 22 10

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>

// ── M31 field: p = 2^31 - 1 ────────────────────────────────

#define P       0x7FFFFFFFu
#define W       24
#define RF      8
#define RP      22
#define HRF     4

__device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    uint32_t r = (s & P) + (s >> 31);
    return r >= P ? r - P : r;
}

__device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : P - b + a;
}

__device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * b;
    uint32_t lo = (uint32_t)(prod & P);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t s = lo + hi;
    uint32_t r = (s & P) + (s >> 31);
    return r >= P ? r - P : r;
}

__device__ __forceinline__ uint32_t m31_pow5(uint32_t x) {
    uint32_t x2 = m31_mul(x, x);
    uint32_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

// ── Poseidon2 constants (device constant memory) ────────────

__constant__ uint32_t d_rc_ext[RF * W];   // external round constants
__constant__ uint32_t d_rc_int[RP];        // internal round constants
__constant__ uint32_t d_diag[W];           // internal MDS diagonal
__constant__ uint32_t d_h_h[8];            // header hash
__constant__ uint32_t d_target[8];         // PoW target

// ── Poseidon2 Width-24 (device) ─────────────────────────────

__device__ void m4(uint32_t x[4]) {
    uint32_t a = x[0], b = x[1], c = x[2], d = x[3];
    uint32_t ab = m31_add(a, b);
    uint32_t cd = m31_add(c, d);
    uint32_t t2 = m31_add(m31_add(b, b), cd);
    uint32_t t3 = m31_add(m31_add(d, d), ab);
    uint32_t ab2 = m31_add(ab, ab);
    uint32_t cd2 = m31_add(cd, cd);
    uint32_t ab4 = m31_add(ab2, ab2);
    uint32_t cd4 = m31_add(cd2, cd2);
    uint32_t t5 = m31_add(ab4, t2);
    uint32_t t4 = m31_add(cd4, t3);
    x[0] = m31_add(t3, t5);
    x[1] = t5;
    x[2] = m31_add(t2, t4);
    x[3] = t4;
}

__device__ void ext_mds(uint32_t s[W]) {
    uint32_t sum[4] = {0, 0, 0, 0};
    for (int g = 0; g < 6; g++)
        for (int i = 0; i < 4; i++)
            sum[i] = m31_add(sum[i], s[g * 4 + i]);
    for (int g = 0; g < 6; g++) {
        uint32_t tmp[4];
        for (int i = 0; i < 4; i++) tmp[i] = m31_add(s[g * 4 + i], sum[i]);
        m4(tmp);
        for (int i = 0; i < 4; i++) s[g * 4 + i] = tmp[i];
    }
}

__device__ void int_mds(uint32_t s[W]) {
    uint32_t sum = 0;
    for (int i = 0; i < W; i++) sum = m31_add(sum, s[i]);
    uint32_t s0 = s[0];
    s[0] = m31_sub(sum, m31_add(s0, s0));
    for (int i = 1; i < W; i++) {
        uint32_t orig = s[i];
        s[i] = m31_add(sum, m31_mul(d_diag[i], orig));
    }
}

__device__ void poseidon2(uint32_t s[W]) {
    ext_mds(s);
    for (int r = 0; r < HRF; r++) {
        for (int i = 0; i < W; i++) s[i] = m31_add(s[i], d_rc_ext[r * W + i]);
        for (int i = 0; i < W; i++) s[i] = m31_pow5(s[i]);
        ext_mds(s);
    }
    for (int r = 0; r < RP; r++) {
        s[0] = m31_add(s[0], d_rc_int[r]);
        s[0] = m31_pow5(s[0]);
        int_mds(s);
    }
    for (int r = HRF; r < RF; r++) {
        for (int i = 0; i < W; i++) s[i] = m31_add(s[i], d_rc_ext[r * W + i]);
        for (int i = 0; i < W; i++) s[i] = m31_pow5(s[i]);
        ext_mds(s);
    }
}

// ── Ticket check ────────────────────────────────────────────

__device__ bool ticket_lt(const uint32_t* ticket) {
    for (int i = 0; i < 8; i++) {
        if (ticket[i] < d_target[i]) return true;
        if (ticket[i] > d_target[i]) return false;
    }
    return false;
}

__device__ int check_tickets(const uint32_t s[W]) {
    for (int slot = 0; slot < 3; slot++)
        if (ticket_lt(&s[slot * 8])) return slot;
    return -1;
}

// ── Kernel 1: Pure PoW (compute-bound, zero memory) ────────
//
// Each thread loops doing Poseidon2 on register-only data.
// State lives entirely in registers (24 × 4B = 96B).
// This is the theoretical maximum Poseidon2 throughput.

__global__ void pow_kernel(
    uint64_t nonce_base,
    volatile int* stop,
    unsigned long long* perm_count,
    unsigned long long* found_nonce,
    int* found_slot
) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * (uint64_t)blockDim.x;
    uint64_t local = 0;
    uint64_t nonce = nonce_base + tid;

    while (!*stop) {
        uint32_t s[W];
        s[0] = (uint32_t)(nonce) & P;
        s[1] = (uint32_t)(nonce >> 31) & P;
        s[2] = (uint32_t)(nonce >> 62) & 0x3;
        for (int i = 3; i < 8; i++) s[i] = 0;
        s[8] = (uint32_t)(tid) & P;
        for (int i = 9; i < 16; i++) s[i] = 0;
        for (int i = 0; i < 8; i++) s[16 + i] = d_h_h[i];

        poseidon2(s);
        local++;

        int slot = check_tickets(s);
        if (slot >= 0) {
            if (atomicCAS((int*)stop, 0, 1) == 0) {
                *found_nonce = nonce;
                *found_slot = slot;
            }
        }
        nonce += stride;
    }

    atomicAdd(perm_count, local);
}

// ── Kernel 2: Memory-bound Poseidon2 (STARK Merkle simulation) ─
//
// Each thread reads 24 words from a large global memory array,
// runs Poseidon2, writes 8 words back.  For arrays >> L2 cache,
// this is memory-bandwidth-bound: the GPU stalls on HBM reads
// while the Poseidon2 ALU pipeline is underutilized.
//
// This simulates STARK Merkle tree building where each compression
// reads two child nodes (16 words) + h_H (8 words from constant)
// and writes one parent node (8 words).

__global__ void mem_kernel(
    uint32_t* __restrict__ data,    // large array [n_elems]
    uint64_t n_elems,               // must be >> L2 cache
    volatile int* stop,
    unsigned long long* perm_count,
    int* found_slot
) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * (uint64_t)blockDim.x;
    uint64_t local = 0;

    // Each thread processes chunks of W elements, striding over the array.
    // Read pattern: sequential with large stride → L2 misses for big arrays.
    uint64_t chunks = n_elems / W;
    uint64_t idx = tid;

    while (!*stop) {
        // Wrap around the array
        uint64_t base = (idx % chunks) * W;

        // Read W words from global memory into registers
        uint32_t s[W];
        for (int i = 0; i < 16; i++) s[i] = data[base + i];
        for (int i = 0; i < 8; i++) s[16 + i] = d_h_h[i];

        poseidon2(s);
        local++;

        // Write 8 words back (parent hash)
        for (int i = 0; i < 8; i++) data[base + i] = s[i];

        // Ticket check (symbiotic PoW from STARK work)
        int slot = check_tickets(s);
        if (slot >= 0) {
            if (atomicCAS((int*)stop, 0, 1) == 0)
                *found_slot = slot;
        }

        idx += stride;
    }

    atomicAdd(perm_count, local);
}

// ── Host helpers ────────────────────────────────────────────

void generate_rc(uint32_t* rc_ext, uint32_t* rc_int) {
    uint64_t z = 0x5A4B3C2D1E0FA9B8ULL;
    auto next = [&]() -> uint32_t {
        z += 0x9E3779B97F4A7C15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x ^= x >> 31;
        return (uint32_t)x & P;
    };
    for (int r = 0; r < RF; r++)
        for (int i = 0; i < W; i++)
            rc_ext[r * W + i] = next();
    for (int i = 0; i < RP; i++)
        rc_int[i] = next();
}

double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

const char* fmt(uint64_t n, char* buf) {
    if (n >= 1000000000ULL) sprintf(buf, "%.2fG", n / 1e9);
    else if (n >= 1000000ULL) sprintf(buf, "%.2fM", n / 1e6);
    else if (n >= 1000ULL) sprintf(buf, "%.1fK", n / 1e3);
    else sprintf(buf, "%lu", (unsigned long)n);
    return buf;
}

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ── Run benchmark phase ─────────────────────────────────────

struct BenchResult {
    double elapsed;
    uint64_t perms;
    double rate;    // Mperm/s
};

// Phase 1: Pure PoW — all GPU compute for Poseidon2
BenchResult run_pure_pow(int blocks, int threads, double run_sec) {
    int *d_stop;
    unsigned long long *d_perms, *d_nonce;
    int *d_slot;

    CHECK(cudaMalloc(&d_stop, sizeof(int)));
    CHECK(cudaMalloc(&d_perms, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_nonce, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_slot, sizeof(int)));
    CHECK(cudaMemset(d_stop, 0, sizeof(int)));
    CHECK(cudaMemset(d_perms, 0, sizeof(unsigned long long)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Launch kernel
    pow_kernel<<<blocks, threads, 0, stream>>>(0, d_stop, d_perms, d_nonce, d_slot);

    // Let it run for run_sec
    double t0 = now_sec();
    while (now_sec() - t0 < run_sec) {
        usleep(50000); // 50ms poll
    }
    int one = 1;
    CHECK(cudaMemcpy(d_stop, &one, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaStreamSynchronize(stream));
    double elapsed = now_sec() - t0;

    unsigned long long perms;
    CHECK(cudaMemcpy(&perms, d_perms, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_stop));
    CHECK(cudaFree(d_perms));
    CHECK(cudaFree(d_nonce));
    CHECK(cudaFree(d_slot));

    return { elapsed, (uint64_t)perms, perms / elapsed / 1e6 };
}

// Phase 2: Memory-bound — Poseidon2 with global memory R/W
BenchResult run_mem_bound(int blocks, int threads, uint32_t log_arr, double run_sec) {
    uint64_t n_elems = 1ULL << log_arr;
    uint64_t bytes = n_elems * sizeof(uint32_t);

    uint32_t *d_data;
    int *d_stop, *d_slot;
    unsigned long long *d_perms;

    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMalloc(&d_stop, sizeof(int)));
    CHECK(cudaMalloc(&d_slot, sizeof(int)));
    CHECK(cudaMalloc(&d_perms, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_stop, 0, sizeof(int)));
    CHECK(cudaMemset(d_perms, 0, sizeof(unsigned long long)));

    // Initialize array with pseudo-random data
    uint32_t* h_data = (uint32_t*)malloc(bytes);
    uint64_t rng = 12345;
    for (uint64_t i = 0; i < n_elems; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        h_data[i] = (uint32_t)rng & P;
    }
    CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    free(h_data);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    mem_kernel<<<blocks, threads, 0, stream>>>(d_data, n_elems, d_stop, d_perms, d_slot);

    double t0 = now_sec();
    while (now_sec() - t0 < run_sec) {
        usleep(50000);
    }
    int one = 1;
    CHECK(cudaMemcpy(d_stop, &one, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaStreamSynchronize(stream));
    double elapsed = now_sec() - t0;

    unsigned long long perms;
    CHECK(cudaMemcpy(&perms, d_perms, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_stop));
    CHECK(cudaFree(d_slot));
    CHECK(cudaFree(d_perms));

    return { elapsed, (uint64_t)perms, perms / elapsed / 1e6 };
}

// Phase 3: Symbiotic — both kernels on different CUDA streams
BenchResult run_symbiotic(int blocks, int threads, uint32_t log_arr, double run_sec) {
    // Split GPU: half blocks for PoW, half for Merkle
    int pow_blocks = blocks / 2;
    int mem_blocks = blocks - pow_blocks;
    if (pow_blocks < 1) pow_blocks = 1;
    if (mem_blocks < 1) mem_blocks = 1;

    // PoW resources
    int *d_pow_stop;
    unsigned long long *d_pow_perms, *d_pow_nonce;
    int *d_pow_slot;
    CHECK(cudaMalloc(&d_pow_stop, sizeof(int)));
    CHECK(cudaMalloc(&d_pow_perms, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_pow_nonce, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_pow_slot, sizeof(int)));
    CHECK(cudaMemset(d_pow_stop, 0, sizeof(int)));
    CHECK(cudaMemset(d_pow_perms, 0, sizeof(unsigned long long)));

    // Mem resources
    uint64_t n_elems = 1ULL << log_arr;
    uint64_t bytes = n_elems * sizeof(uint32_t);
    uint32_t *d_data;
    int *d_mem_stop, *d_mem_slot;
    unsigned long long *d_mem_perms;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMalloc(&d_mem_stop, sizeof(int)));
    CHECK(cudaMalloc(&d_mem_slot, sizeof(int)));
    CHECK(cudaMalloc(&d_mem_perms, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_mem_stop, 0, sizeof(int)));
    CHECK(cudaMemset(d_mem_perms, 0, sizeof(unsigned long long)));

    // Init array
    uint32_t* h_data = (uint32_t*)malloc(bytes);
    uint64_t rng = 67890;
    for (uint64_t i = 0; i < n_elems; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        h_data[i] = (uint32_t)rng & P;
    }
    CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    free(h_data);

    // Two streams for concurrent execution
    cudaStream_t s_pow, s_mem;
    CHECK(cudaStreamCreate(&s_pow));
    CHECK(cudaStreamCreate(&s_mem));

    // Launch both simultaneously
    pow_kernel<<<pow_blocks, threads, 0, s_pow>>>(
        0, d_pow_stop, d_pow_perms, d_pow_nonce, d_pow_slot);
    mem_kernel<<<mem_blocks, threads, 0, s_mem>>>(
        d_data, n_elems, d_mem_stop, d_mem_perms, d_mem_slot);

    double t0 = now_sec();
    while (now_sec() - t0 < run_sec) {
        usleep(50000);
    }
    int one = 1;
    CHECK(cudaMemcpy(d_pow_stop, &one, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mem_stop, &one, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaStreamSynchronize(s_pow));
    CHECK(cudaStreamSynchronize(s_mem));
    double elapsed = now_sec() - t0;

    unsigned long long pow_perms, mem_perms;
    CHECK(cudaMemcpy(&pow_perms, d_pow_perms, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&mem_perms, d_mem_perms, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    printf("  PoW  perms: %llu (%.2f Mperm/s)\n", pow_perms, pow_perms / elapsed / 1e6);
    printf("  Mem  perms: %llu (%.2f Mperm/s)\n", mem_perms, mem_perms / elapsed / 1e6);

    uint64_t total = pow_perms + mem_perms;

    CHECK(cudaStreamDestroy(s_pow));
    CHECK(cudaStreamDestroy(s_mem));
    CHECK(cudaFree(d_pow_stop));
    CHECK(cudaFree(d_pow_perms));
    CHECK(cudaFree(d_pow_nonce));
    CHECK(cudaFree(d_pow_slot));
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_mem_stop));
    CHECK(cudaFree(d_mem_slot));
    CHECK(cudaFree(d_mem_perms));

    return { elapsed, total, total / elapsed / 1e6 };
}

// ── Main ────────────────────────────────────────────────────

int main(int argc, char** argv) {
    uint32_t difficulty = argc > 1 ? atoi(argv[1]) : 24;
    uint32_t log_arr    = argc > 2 ? atoi(argv[2]) : 22;  // array = 2^22 u32 = 16MB
    double   run_sec    = argc > 3 ? atof(argv[3]) : 5.0;

    // GPU info
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;

    printf("ZK-SPoW GPU Benchmark — Poseidon2 Width-24 / M31\n");
    printf("════════════════════════════════════════════════════\n");
    printf("GPU          : %s\n", prop.name);
    printf("SMs          : %d\n", sms);
    printf("Clock        : %d MHz\n", prop.clockRate / 1000);
    printf("Memory       : %.1f GB (%.0f GB/s)\n",
        prop.totalGlobalMem / 1e9,
        2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("L2 cache     : %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("────────────────────────────────────────────────────\n");
    printf("Difficulty   : %u bits\n", difficulty);
    printf("Array size   : 2^%u = %.1f MB (should be >> L2)\n",
        log_arr, (1ULL << log_arr) * 4.0 / 1e6);
    printf("Run time     : %.1f sec per phase\n", run_sec);
    printf("Poseidon2    : W=%d, Rf=%d, Rp=%d\n", W, RF, RP);
    printf("════════════════════════════════════════════════════\n");

    // Generate and upload constants
    uint32_t rc_ext[RF * W], rc_int[RP];
    generate_rc(rc_ext, rc_int);

    uint32_t diag[W] = {
        P - 2, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
        2048, 4096, 8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576, 2097152, 4194304
    };
    uint32_t h_h[8] = {
        0x5EADBEEF & P, 0x4AFEBABE & P, 0x12345678, 0x1ABCDEF0 & P,
        42, 2026, 2, 19
    };
    uint32_t target[8];
    for (int i = 0; i < 8; i++) target[i] = P - 1;
    target[0] = difficulty >= 31 ? 0 : P >> difficulty;

    CHECK(cudaMemcpyToSymbol(d_rc_ext, rc_ext, sizeof(rc_ext)));
    CHECK(cudaMemcpyToSymbol(d_rc_int, rc_int, sizeof(rc_int)));
    CHECK(cudaMemcpyToSymbol(d_diag, diag, sizeof(diag)));
    CHECK(cudaMemcpyToSymbol(d_h_h, h_h, sizeof(h_h)));
    CHECK(cudaMemcpyToSymbol(d_target, target, sizeof(target)));

    int blocks = sms * 4;   // 4 blocks per SM for good occupancy
    int threads = 256;

    char buf1[32], buf2[32], buf3[32];

    // Phase 1: Pure PoW
    printf("\n── Phase 1: Pure PoW (compute-only) ───────────\n");
    BenchResult r1 = run_pure_pow(blocks, threads, run_sec);
    printf("Perms        : %s\n", fmt(r1.perms, buf1));
    printf("Time         : %.3fs\n", r1.elapsed);
    printf("Rate         : %.2f Mperm/s\n", r1.rate);

    // Phase 2: Memory-bound
    printf("\n── Phase 2: Memory-bound (Merkle/LDE sim) ─────\n");
    BenchResult r2 = run_mem_bound(blocks, threads, log_arr, run_sec);
    printf("Perms        : %s\n", fmt(r2.perms, buf2));
    printf("Time         : %.3fs\n", r2.elapsed);
    printf("Rate         : %.2f Mperm/s\n", r2.rate);

    // Phase 3: Symbiotic
    printf("\n── Phase 3: Symbiotic (PoW + Merkle concurrent) \n");
    BenchResult r3 = run_symbiotic(blocks, threads, log_arr, run_sec);
    printf("Total perms  : %s\n", fmt(r3.perms, buf3));
    printf("Time         : %.3fs\n", r3.elapsed);
    printf("Total rate   : %.2f Mperm/s\n", r3.rate);

    // Summary
    printf("\n════════════════════════════════════════════════════\n");
    printf("RESULTS SUMMARY\n");
    printf("────────────────────────────────────────────────────\n");
    printf("Pure PoW     : %8.2f Mperm/s  (baseline)\n", r1.rate);
    printf("Mem-bound    : %8.2f Mperm/s  (%.1f%% of baseline)\n",
        r2.rate, r2.rate / r1.rate * 100);
    printf("Symbiotic    : %8.2f Mperm/s  (%.1f%% of baseline)\n",
        r3.rate, r3.rate / r1.rate * 100);
    printf("────────────────────────────────────────────────────\n");

    double overhead = (1.0 - r3.rate / r1.rate) * 100;
    if (overhead < 0) overhead = 0;
    printf("Symbiotic overhead: %.1f%%\n", overhead);

    if (r3.rate >= r1.rate * 0.90)
        printf("VERDICT      : Complementary bottleneck CONFIRMED\n");
    else if (r3.rate >= r1.rate * 0.70)
        printf("VERDICT      : Partial overlap — some resource contention\n");
    else
        printf("VERDICT      : Bottleneck NOT complementary on this GPU\n");

    printf("════════════════════════════════════════════════════\n");
    return 0;
}
