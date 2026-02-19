// ZK-SPoW Full GPU STARK Prover + Symbiotic Benchmark
//
// Complete Circle STARK over M31 on GPU:
//   iCFFT/CFFT (butterfly FFT), Merkle tree (Poseidon2),
//   constraint quotient, FRI fold — all as GPU kernels.
//
// Benchmark measures per-component and end-to-end throughput,
// then compares Pure PoW vs Symbiotic (STARK + PoW concurrent).
//
// Build:  nvcc -O3 -arch=sm_80 stark.cu -o stark
// Run:    ./stark [log_trace] [difficulty] [seconds]
// Sweep:  ./stark sweep [difficulty] [seconds]
// Example: ./stark 16 24 10
// Example: ./stark sweep 24 3

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>

#define P       0x7FFFFFFFu
#define W       24
#define RF      8
#define RP      22
#define HRF     4
#define LOG_BLOWUP  2
#define N_QUERIES   24
#define FRI_STOP    4

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); \
    } } while(0)

// ═══════════════════════════════════════════════════════════════
// M31 field — host + device
// ═══════════════════════════════════════════════════════════════

__host__ __device__ __forceinline__ uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b, r = (s & P) + (s >> 31);
    return r >= P ? r - P : r;
}
__host__ __device__ __forceinline__ uint32_t m31_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : P - b + a;
}
__host__ __device__ __forceinline__ uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t p = (uint64_t)a * b;
    uint32_t lo = (uint32_t)(p & P), hi = (uint32_t)(p >> 31);
    uint32_t s = lo + hi, r = (s & P) + (s >> 31);
    return r >= P ? r - P : r;
}
__host__ __device__ __forceinline__ uint32_t m31_pow5(uint32_t x) {
    uint32_t x2 = m31_mul(x, x); return m31_mul(m31_mul(x2, x2), x);
}
__host__ __device__ uint32_t m31_pow(uint32_t b, uint32_t e) {
    uint32_t r = 1;
    while (e > 0) { if (e & 1) r = m31_mul(r, b); b = m31_mul(b, b); e >>= 1; }
    return r;
}
__host__ __device__ uint32_t m31_inv(uint32_t a) { return m31_pow(a, P - 2); }
__host__ __device__ uint32_t m31_neg(uint32_t a) { return a == 0 ? 0 : P - a; }
__host__ __device__ uint32_t m31_double_x(uint32_t x) {
    return m31_sub(m31_add(m31_mul(x, x), m31_mul(x, x)), 1);
}

// ═══════════════════════════════════════════════════════════════
// Poseidon2 Width-24 — device (constant memory) + host (params)
// ═══════════════════════════════════════════════════════════════

static const uint32_t DIAG[W] = {
    P-2,1,2,4,8,16,32,64,128,256,512,1024,
    2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304
};

struct RC { uint32_t ext[RF][W]; uint32_t intr[RP]; };

void gen_rc(RC* rc) {
    uint64_t z = 0x5A4B3C2D1E0FA9B8ULL;
    auto next = [&]() -> uint32_t {
        z += 0x9E3779B97F4A7C15ULL; uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return (uint32_t)(x ^ (x >> 31)) & P;
    };
    for (int r = 0; r < RF; r++) for (int i = 0; i < W; i++) rc->ext[r][i] = next();
    for (int i = 0; i < RP; i++) rc->intr[i] = next();
}

__host__ __device__ void _m4(uint32_t x[4]) {
    uint32_t ab = m31_add(x[0], x[1]), cd = m31_add(x[2], x[3]);
    uint32_t t2 = m31_add(m31_add(x[1], x[1]), cd);
    uint32_t t3 = m31_add(m31_add(x[3], x[3]), ab);
    uint32_t ab4 = m31_add(m31_add(ab, ab), m31_add(ab, ab));
    uint32_t cd4 = m31_add(m31_add(cd, cd), m31_add(cd, cd));
    uint32_t t5 = m31_add(ab4, t2), t4 = m31_add(cd4, t3);
    x[0] = m31_add(t3, t5); x[1] = t5; x[2] = m31_add(t2, t4); x[3] = t4;
}

__host__ __device__ void _ext_mds(uint32_t s[W]) {
    uint32_t sum[4] = {};
    for (int g = 0; g < 6; g++) for (int i = 0; i < 4; i++) sum[i] = m31_add(sum[i], s[g*4+i]);
    for (int g = 0; g < 6; g++) {
        uint32_t t[4]; for (int i = 0; i < 4; i++) t[i] = m31_add(s[g*4+i], sum[i]);
        _m4(t); for (int i = 0; i < 4; i++) s[g*4+i] = t[i];
    }
}

__host__ __device__ void _int_mds(uint32_t s[W], const uint32_t diag[W]) {
    uint32_t sum = 0; for (int i = 0; i < W; i++) sum = m31_add(sum, s[i]);
    uint32_t s0 = s[0]; s[0] = m31_sub(sum, m31_add(s0, s0));
    for (int i = 1; i < W; i++) { uint32_t o = s[i]; s[i] = m31_add(sum, m31_mul(diag[i], o)); }
}

// Host Poseidon2
void h_poseidon2(uint32_t s[W], const RC* rc) {
    _ext_mds(s);
    for (int r = 0; r < HRF; r++) {
        for (int i = 0; i < W; i++) s[i] = m31_add(s[i], rc->ext[r][i]);
        for (int i = 0; i < W; i++) s[i] = m31_pow5(s[i]); _ext_mds(s);
    }
    for (int r = 0; r < RP; r++) {
        s[0] = m31_add(s[0], rc->intr[r]); s[0] = m31_pow5(s[0]); _int_mds(s, DIAG);
    }
    for (int r = HRF; r < RF; r++) {
        for (int i = 0; i < W; i++) s[i] = m31_add(s[i], rc->ext[r][i]);
        for (int i = 0; i < W; i++) s[i] = m31_pow5(s[i]); _ext_mds(s);
    }
}

// Device round constants + Poseidon2
__constant__ uint32_t d_rc_ext[RF * W];
__constant__ uint32_t d_rc_int[RP];
__constant__ uint32_t d_diag[W];
__constant__ uint32_t d_h_h[8];
__constant__ uint32_t d_target[8];

__device__ void d_poseidon2(uint32_t s[W]) {
    _ext_mds(s);
    for (int r = 0; r < HRF; r++) {
        for (int i = 0; i < W; i++) s[i] = m31_add(s[i], d_rc_ext[r * W + i]);
        for (int i = 0; i < W; i++) s[i] = m31_pow5(s[i]); _ext_mds(s);
    }
    for (int r = 0; r < RP; r++) {
        s[0] = m31_add(s[0], d_rc_int[r]); s[0] = m31_pow5(s[0]); _int_mds(s, d_diag);
    }
    for (int r = HRF; r < RF; r++) {
        for (int i = 0; i < W; i++) s[i] = m31_add(s[i], d_rc_ext[r * W + i]);
        for (int i = 0; i < W; i++) s[i] = m31_pow5(s[i]); _ext_mds(s);
    }
}

__device__ bool d_ticket_lt(const uint32_t* t) {
    for (int i = 0; i < 8; i++) { if (t[i] < d_target[i]) return true; if (t[i] > d_target[i]) return false; }
    return false;
}

// ═══════════════════════════════════════════════════════════════
// Circle group + domain (host)
// ═══════════════════════════════════════════════════════════════

struct CPoint { uint32_t x, y; };

CPoint cp_cadd(CPoint a, CPoint b) {
    return { m31_sub(m31_mul(a.x, b.x), m31_mul(a.y, b.y)),
             m31_add(m31_mul(a.x, b.y), m31_mul(a.y, b.x)) };
}
CPoint cp_double(CPoint p) {
    uint32_t xx = m31_mul(p.x, p.x);
    return { m31_sub(m31_add(xx, xx), 1), m31_add(m31_mul(p.x, p.y), m31_mul(p.x, p.y)) };
}
CPoint cp_rep_double(CPoint p, int n) { for (int i = 0; i < n; i++) p = cp_double(p); return p; }

static const CPoint CP_GEN = { 2, 1268011823u };  // order 2^31

CPoint cp_subgroup_gen(int log_size) { return cp_rep_double(CP_GEN, 31 - log_size); }

struct CDomain {
    uint32_t* half_x;    // x-coordinates of half-coset
    uint32_t* half_y;    // y-coordinates of half-coset
    int log_size;
    int half_n;
};

CDomain cd_new(int log_size) {
    int half_n = 1 << (log_size - 1);
    CDomain d;
    d.log_size = log_size;
    d.half_n = half_n;
    d.half_x = (uint32_t*)malloc(half_n * sizeof(uint32_t));
    d.half_y = (uint32_t*)malloc(half_n * sizeof(uint32_t));
    CPoint init = cp_subgroup_gen(log_size + 1);
    CPoint step = cp_subgroup_gen(log_size - 1);
    CPoint cur = init;
    for (int i = 0; i < half_n; i++) {
        d.half_x[i] = cur.x; d.half_y[i] = cur.y;
        cur = cp_cadd(cur, step);
    }
    return d;
}

void cd_free(CDomain* d) { free(d->half_x); free(d->half_y); }

// ═══════════════════════════════════════════════════════════════
// Twiddle factors (host → device)
// ═══════════════════════════════════════════════════════════════

struct Twiddles {
    uint32_t* flat;       // all levels concatenated
    int* offsets;         // offset[k] = start of level k
    int n_levels;
    int total;
    uint32_t* d_flat;    // device copy
};

// Inverse twiddles for iCFFT / FRI fold
Twiddles make_inv_twiddles(CDomain* dom) {
    int half_n = dom->half_n;
    int n_levels = 0, total = 0;
    { int s = half_n; while (s >= 1) { total += s; n_levels++; s /= 2; } }

    Twiddles tw;
    tw.n_levels = n_levels;
    tw.total = total;
    tw.flat = (uint32_t*)malloc(total * sizeof(uint32_t));
    tw.offsets = (int*)malloc(n_levels * sizeof(int));

    // Level 0: inv(y_i)
    tw.offsets[0] = 0;
    for (int i = 0; i < half_n; i++) tw.flat[i] = m31_inv(dom->half_y[i]);

    // Level k≥1: inv of projected x-coordinates
    uint32_t* xs = (uint32_t*)malloc(half_n * sizeof(uint32_t));
    memcpy(xs, dom->half_x, half_n * sizeof(uint32_t));
    int off = half_n, size = half_n / 2;
    for (int k = 1; k < n_levels; k++) {
        tw.offsets[k] = off;
        for (int i = 0; i < size; i++) tw.flat[off + i] = m31_inv(xs[i]);
        for (int i = 0; i < half_n; i++) xs[i] = m31_double_x(xs[i]);
        off += size; size /= 2;
    }
    free(xs);

    CHECK(cudaMalloc(&tw.d_flat, total * sizeof(uint32_t)));
    CHECK(cudaMemcpy(tw.d_flat, tw.flat, total * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return tw;
}

// Forward twiddles for CFFT
Twiddles make_fwd_twiddles(CDomain* dom) {
    int half_n = dom->half_n;
    int n_levels = 0, total = 0;
    { int s = half_n; while (s >= 1) { total += s; n_levels++; s /= 2; } }

    Twiddles tw;
    tw.n_levels = n_levels;
    tw.total = total;
    tw.flat = (uint32_t*)malloc(total * sizeof(uint32_t));
    tw.offsets = (int*)malloc(n_levels * sizeof(int));

    // Level 0: y_i
    tw.offsets[0] = 0;
    for (int i = 0; i < half_n; i++) tw.flat[i] = dom->half_y[i];

    uint32_t* xs = (uint32_t*)malloc(half_n * sizeof(uint32_t));
    memcpy(xs, dom->half_x, half_n * sizeof(uint32_t));
    int off = half_n, size = half_n / 2;
    for (int k = 1; k < n_levels; k++) {
        tw.offsets[k] = off;
        for (int i = 0; i < size; i++) tw.flat[off + i] = xs[i];
        for (int i = 0; i < half_n; i++) xs[i] = m31_double_x(xs[i]);
        off += size; size /= 2;
    }
    free(xs);

    CHECK(cudaMalloc(&tw.d_flat, total * sizeof(uint32_t)));
    CHECK(cudaMemcpy(tw.d_flat, tw.flat, total * sizeof(uint32_t), cudaMemcpyHostToDevice));
    return tw;
}

void tw_free(Twiddles* tw) {
    free(tw->flat); free(tw->offsets); cudaFree(tw->d_flat);
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernels: FFT butterfly
// ═══════════════════════════════════════════════════════════════

// DIF butterfly stage: pairs at stride `half` within groups of `group_size`.
// input[base+i] and input[base+half+i] → output[base+i], output[base+half+i]
__global__ void k_butterfly_dif(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
    const uint32_t* __restrict__ twiddles, int tw_offset,
    int n_groups, int group_size, int half
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_groups * half;
    if (tid >= total) return;
    int g = tid / half, i = tid % half;
    int base = g * group_size;
    uint32_t a = in[base + i], b = in[base + half + i];
    uint32_t tw = twiddles[tw_offset + i];
    out[base + i] = m31_add(a, b);
    out[base + half + i] = m31_mul(m31_sub(a, b), tw);
}

// DIT butterfly stage (for CFFT): pairs at stride `half` within groups.
__global__ void k_butterfly_dit(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
    const uint32_t* __restrict__ twiddles, int tw_offset,
    int n_groups, int group_size, int half
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_groups * half;
    if (tid >= total) return;
    int g = tid / half, i = tid % half;
    int base = g * group_size;
    uint32_t ee = in[base + i], oe = in[base + half + i];
    uint32_t t = m31_mul(twiddles[tw_offset + i], oe);
    out[base + i] = m31_add(ee, t);
    out[base + half + i] = m31_sub(ee, t);
}

// Bit-reversal permutation
__global__ void k_bitrev(const uint32_t* in, uint32_t* out, int n, int log_n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int rev = 0, x = tid;
    for (int i = 0; i < log_n; i++) { rev = (rev << 1) | (x & 1); x >>= 1; }
    out[rev] = in[tid];
}

// Scale all elements by constant
__global__ void k_scale(uint32_t* data, int n, uint32_t factor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] = m31_mul(data[tid], factor);
}

// ═══════════════════════════════════════════════════════════════
// GPU iCFFT / CFFT orchestration (host launches kernels)
// ═══════════════════════════════════════════════════════════════

int div_up(int a, int b) { return (a + b - 1) / b; }

// iCFFT: evaluations → coefficients (in-place with scratch buffer)
void gpu_icfft(uint32_t* d_data, uint32_t* d_scratch, int log_n,
               Twiddles* inv_tw, cudaStream_t stream)
{
    int n = 1 << log_n;
    int threads = 256;
    uint32_t *A = d_data, *B = d_scratch;

    // DIF stages: stride halves each time
    int n_groups = 1, group_size = n, half = n / 2;
    for (int k = 0; k < log_n; k++) {
        int total = n_groups * half;
        k_butterfly_dif<<<div_up(total, threads), threads, 0, stream>>>(
            A, B, inv_tw->d_flat, inv_tw->offsets[k], n_groups, group_size, half);
        uint32_t* tmp = A; A = B; B = tmp;
        n_groups *= 2; group_size /= 2; half /= 2;
    }

    // Bit-reversal permutation
    k_bitrev<<<div_up(n, threads), threads, 0, stream>>>(A, B, n, log_n);
    if (B != d_data)
        CHECK(cudaMemcpyAsync(d_data, B, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));

    // Scale by 1/N
    uint32_t inv_n = m31_inv(n);
    k_scale<<<div_up(n, threads), threads, 0, stream>>>(d_data, n, inv_n);
}

// CFFT: coefficients → evaluations
void gpu_cfft(uint32_t* d_data, uint32_t* d_scratch, int log_n,
              Twiddles* fwd_tw, cudaStream_t stream)
{
    int n = 1 << log_n;
    int threads = 256;
    uint32_t *A = d_data, *B = d_scratch;

    // Bit-reversal first (undo interleaved coefficient order)
    k_bitrev<<<div_up(n, threads), threads, 0, stream>>>(A, B, n, log_n);
    uint32_t* tmp = A; A = B; B = tmp;

    // DIT stages: stride doubles each time
    int n_groups = n / 2, group_size = 2, half = 1;
    for (int k = log_n - 1; k >= 0; k--) {
        int total = n_groups * half;
        k_butterfly_dit<<<div_up(total, threads), threads, 0, stream>>>(
            A, B, fwd_tw->d_flat, fwd_tw->offsets[k], n_groups, group_size, half);
        uint32_t* tmp2 = A; A = B; B = tmp2;
        n_groups /= 2; group_size *= 2; half *= 2;
    }

    if (A != d_data)
        CHECK(cudaMemcpyAsync(d_data, A, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernel: Merkle tree build (one level)
// ═══════════════════════════════════════════════════════════════

// Build one Merkle level: parents[i] = compress(children[2i], children[2i+1])
// Each node = 8 uint32. h_H from constant memory.
// Returns ticket via atomics.
__global__ void k_merkle_level(
    const uint32_t* __restrict__ children, // [2*n_parents][8]
    uint32_t* __restrict__ parents,         // [n_parents][8]
    int n_parents,
    unsigned long long* perm_count,
    volatile int* found, int* found_slot, uint32_t* found_state
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_parents) return;

    uint32_t s[W];
    for (int j = 0; j < 8; j++) s[j]      = children[tid * 16 + j];
    for (int j = 0; j < 8; j++) s[8 + j]  = children[tid * 16 + 8 + j];
    for (int j = 0; j < 8; j++) s[16 + j] = d_h_h[j];

    d_poseidon2(s);
    atomicAdd(perm_count, 1ULL);

    for (int j = 0; j < 8; j++) parents[tid * 8 + j] = s[j];

    // Check PoW tickets (symbiotic)
    for (int slot = 0; slot < 3; slot++) {
        if (d_ticket_lt(&s[slot * 8]) && atomicCAS((int*)found, 0, 1) == 0) {
            *found_slot = slot;
            for (int j = 0; j < W; j++) found_state[j] = s[j];
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernel: Constraint quotient
// ═══════════════════════════════════════════════════════════════

// Q[j] = C_lde[j] * Z_boundary(pt_j) / V_coset(pt_j)
// Pre-uploaded: d_lde_x[j] = x-coordinate of LDE point j
__global__ void k_quotient(
    const uint32_t* __restrict__ c_lde,  // constraint on LDE domain
    uint32_t* __restrict__ q_evals,       // output quotient
    const uint32_t* __restrict__ lde_x,   // x-coordinates of LDE domain
    uint32_t x_b0, uint32_t x_b1, int log_trace, int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    uint32_t x = lde_x[j];
    uint32_t z = m31_mul(m31_sub(x, x_b0), m31_sub(x, x_b1));

    // vanish_coset: apply double_x (log_trace-1) times
    uint32_t v = x;
    for (int k = 1; k < log_trace; k++) v = m31_double_x(v);

    q_evals[j] = m31_mul(m31_mul(c_lde[j], z), m31_inv(v));
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernel: Fibonacci constraint from trace (for pre-alloc prover)
// ═══════════════════════════════════════════════════════════════

__global__ void k_constraint_vals(
    const uint32_t* __restrict__ trace, uint32_t* __restrict__ out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = m31_sub(trace[(i + 2) % n], m31_add(trace[(i + 1) % n], trace[i]));
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernel: FRI fold (circle + line, same operation)
// ═══════════════════════════════════════════════════════════════

// fold: out[i] = (in[i]+in[i+half]) + alpha * (in[i]-in[i+half]) * tw[i]
__global__ void k_fri_fold(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
    const uint32_t* __restrict__ twiddles, int tw_offset,
    uint32_t alpha, int half
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    uint32_t a = in[i], b = in[i + half];
    uint32_t f0 = m31_add(a, b);
    uint32_t f1 = m31_mul(m31_sub(a, b), twiddles[tw_offset + i]);
    out[i] = m31_add(f0, m31_mul(alpha, f1));
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernel: Pure PoW (compute-only, for baseline)
// ═══════════════════════════════════════════════════════════════

__global__ void k_pow(
    uint64_t nonce_base, volatile int* stop,
    unsigned long long* perm_count,
    unsigned long long* found_nonce, int* found_slot
) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * (uint64_t)blockDim.x;
    uint64_t local = 0, nonce = nonce_base + tid;

    while (!*stop) {
        uint32_t s[W] = {};
        s[0] = (uint32_t)(nonce) & P;
        s[1] = (uint32_t)(nonce >> 31) & P;
        s[8] = (uint32_t)(tid) & P;
        for (int i = 0; i < 8; i++) s[16 + i] = d_h_h[i];
        d_poseidon2(s);
        local++;
        for (int slot = 0; slot < 3; slot++)
            if (d_ticket_lt(&s[slot * 8]) && atomicCAS((int*)stop, 0, 1) == 0) {
                *found_nonce = nonce; *found_slot = slot;
            }
        nonce += stride;
    }
    atomicAdd(perm_count, local);
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernel: Batched PoW (non-persistent, same pattern as Merkle)
//
// One thread = one Poseidon2 perm, then exit.
// Same d_poseidon2() as k_merkle_level.  Apples-to-apples.
// ═══════════════════════════════════════════════════════════════

__global__ void k_pow_batch(
    uint64_t nonce_base, int batch_size,
    volatile int* found, int* found_slot, uint32_t* found_state
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    uint64_t nonce = nonce_base + (uint64_t)tid;
    uint32_t s[W] = {};
    s[0] = (uint32_t)(nonce) & P;
    s[1] = (uint32_t)(nonce >> 31) & P;
    s[8] = (uint32_t)(tid) & P;
    for (int i = 0; i < 8; i++) s[16 + i] = d_h_h[i];

    d_poseidon2(s);

    for (int slot = 0; slot < 3; slot++) {
        if (d_ticket_lt(&s[slot * 8]) && atomicCAS((int*)found, 0, 1) == 0) {
            *found_slot = slot;
            for (int j = 0; j < W; j++) found_state[j] = s[j];
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Host: Merkle tree utilities
// ═══════════════════════════════════════════════════════════════

struct MerkleGPU {
    uint32_t* d_tree;   // flat array: [leaves | level1 | level2 | ... | root]
    int* level_offsets;  // offset of each level in d_tree (in uint32 count)
    int n_leaves;
    int n_levels;
    uint64_t perms;
};

// Pack evaluations into 8-word leaves
void pack_to_leaves(const uint32_t* evals, int n, uint32_t* leaves) {
    int n_leaves = (n + 7) / 8;
    memset(leaves, 0, n_leaves * 8 * sizeof(uint32_t));
    memcpy(leaves, evals, n * sizeof(uint32_t));
}

// Build Merkle tree on GPU, return tree in device memory
MerkleGPU gpu_merkle_build(const uint32_t* d_evals, int n_evals,
                           cudaStream_t stream,
                           unsigned long long* d_perm_count,
                           volatile int* d_found, int* d_found_slot,
                           uint32_t* d_found_state)
{
    int n_leaves_raw = (n_evals + 7) / 8;
    int n_leaves = 1;
    while (n_leaves < n_leaves_raw) n_leaves <<= 1;

    // Total tree nodes
    int total_nodes = 0;
    { int s = n_leaves; while (s >= 1) { total_nodes += s; s /= 2; } }

    MerkleGPU m;
    m.n_leaves = n_leaves;
    m.n_levels = 0;
    { int s = n_leaves; while (s >= 1) { m.n_levels++; s /= 2; } }
    m.level_offsets = (int*)malloc(m.n_levels * sizeof(int));
    int off = 0;
    { int s = n_leaves; for (int k = 0; k < m.n_levels; k++) { m.level_offsets[k] = off * 8; off += s; s /= 2; } }

    CHECK(cudaMalloc(&m.d_tree, total_nodes * 8 * sizeof(uint32_t)));
    CHECK(cudaMemsetAsync(m.d_tree, 0, total_nodes * 8 * sizeof(uint32_t), stream));

    // Copy evaluations to leaf level (packed as 8-word nodes)
    CHECK(cudaMemcpyAsync(m.d_tree, d_evals, n_evals * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice, stream));

    // Build levels bottom-up
    int threads = 256;
    int cur_size = n_leaves;
    for (int k = 0; k < m.n_levels - 1; k++) {
        int n_parents = cur_size / 2;
        uint32_t* children = m.d_tree + m.level_offsets[k];
        uint32_t* parents = m.d_tree + m.level_offsets[k + 1];
        k_merkle_level<<<div_up(n_parents, threads), threads, 0, stream>>>(
            children, parents, n_parents, d_perm_count, d_found, d_found_slot, d_found_state);
        cur_size = n_parents;
    }

    return m;
}

void merkle_root(MerkleGPU* m, uint32_t root[8], cudaStream_t stream) {
    int last = m->level_offsets[m->n_levels - 1];
    CHECK(cudaMemcpyAsync(root, m->d_tree + last, 8 * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
}

void merkle_free(MerkleGPU* m) { cudaFree(m->d_tree); free(m->level_offsets); }

// ═══════════════════════════════════════════════════════════════
// Host: Full STARK prover on GPU
// ═══════════════════════════════════════════════════════════════

struct StarkTiming {
    double t_icfft, t_cfft, t_merkle_trace, t_quotient, t_merkle_quot;
    double t_fri, t_total;
    uint64_t poseidon_perms;
};

StarkTiming gpu_stark_prove(
    int log_trace, uint32_t a0, uint32_t a1,
    const RC* rc, cudaStream_t stream
) {
    StarkTiming tm = {};
    double t0, t1;
    int n = 1 << log_trace;
    int log_lde = log_trace + LOG_BLOWUP;
    int lde_n = 1 << log_lde;
    int threads = 256;

    // Shared counters
    unsigned long long *d_perms;
    int *d_found, *d_slot;
    uint32_t *d_state;
    CHECK(cudaMalloc(&d_perms, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_found, sizeof(int)));
    CHECK(cudaMalloc(&d_slot, sizeof(int)));
    CHECK(cudaMalloc(&d_state, W * sizeof(uint32_t)));
    CHECK(cudaMemsetAsync(d_perms, 0, sizeof(unsigned long long), stream));
    CHECK(cudaMemsetAsync(d_found, 0, sizeof(int), stream));

    // 1. Generate trace on CPU, upload
    uint32_t* trace = (uint32_t*)malloc(n * sizeof(uint32_t));
    trace[0] = a0; trace[1] = a1;
    for (int i = 2; i < n; i++) trace[i] = m31_add(trace[i-1], trace[i-2]);

    uint32_t *d_trace, *d_scratch;
    CHECK(cudaMalloc(&d_trace, lde_n * sizeof(uint32_t)));  // will reuse for LDE
    CHECK(cudaMalloc(&d_scratch, lde_n * sizeof(uint32_t)));
    CHECK(cudaMemcpyAsync(d_trace, trace, n * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    // Domains + twiddles
    CDomain trace_dom = cd_new(log_trace);
    CDomain lde_dom = cd_new(log_lde);
    Twiddles trace_inv_tw = make_inv_twiddles(&trace_dom);
    Twiddles lde_fwd_tw = make_fwd_twiddles(&lde_dom);
    Twiddles lde_inv_tw = make_inv_twiddles(&lde_dom);

    double t_start = (double)clock() / CLOCKS_PER_SEC;

    // 2. iCFFT: trace → coefficients
    CHECK(cudaStreamSynchronize(stream));
    t0 = (double)clock() / CLOCKS_PER_SEC;
    gpu_icfft(d_trace, d_scratch, log_trace, &trace_inv_tw, stream);
    CHECK(cudaStreamSynchronize(stream));
    t1 = (double)clock() / CLOCKS_PER_SEC;
    tm.t_icfft = t1 - t0;

    // 3. Pad to LDE size, CFFT → LDE evaluations
    // d_trace now has n coefficients. Pad to lde_n.
    CHECK(cudaMemsetAsync(d_trace + n, 0, (lde_n - n) * sizeof(uint32_t), stream));
    t0 = (double)clock() / CLOCKS_PER_SEC;
    gpu_cfft(d_trace, d_scratch, log_lde, &lde_fwd_tw, stream);
    CHECK(cudaStreamSynchronize(stream));
    t1 = (double)clock() / CLOCKS_PER_SEC;
    tm.t_cfft = t1 - t0;
    // d_trace now has LDE evaluations

    // 4. Merkle commit trace
    t0 = (double)clock() / CLOCKS_PER_SEC;
    MerkleGPU t_tree = gpu_merkle_build(d_trace, lde_n, stream,
                                        d_perms, d_found, d_slot, d_state);
    CHECK(cudaStreamSynchronize(stream));
    t1 = (double)clock() / CLOCKS_PER_SEC;
    tm.t_merkle_trace = t1 - t0;

    uint32_t trace_root[8];
    merkle_root(&t_tree, trace_root, stream);

    // 5. Constraint quotient
    // Recompute constraint polynomial on trace domain, LDE it, then quotient
    uint32_t* c_vals = (uint32_t*)malloc(n * sizeof(uint32_t));
    for (int i = 0; i < n; i++)
        c_vals[i] = m31_sub(trace[(i+2)%n], m31_add(trace[(i+1)%n], trace[i]));

    uint32_t *d_cvals;
    CHECK(cudaMalloc(&d_cvals, lde_n * sizeof(uint32_t)));
    CHECK(cudaMemcpyAsync(d_cvals, c_vals, n * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    free(c_vals);

    // iCFFT on constraint values
    gpu_icfft(d_cvals, d_scratch, log_trace, &trace_inv_tw, stream);
    // Pad + CFFT for constraint LDE
    CHECK(cudaMemsetAsync(d_cvals + n, 0, (lde_n - n) * sizeof(uint32_t), stream));
    gpu_cfft(d_cvals, d_scratch, log_lde, &lde_fwd_tw, stream);
    CHECK(cudaStreamSynchronize(stream));
    // d_cvals now has constraint LDE

    // Upload LDE x-coordinates for quotient kernel
    uint32_t* lde_x = (uint32_t*)malloc(lde_n * sizeof(uint32_t));
    for (int i = 0; i < lde_dom.half_n; i++) {
        lde_x[i] = lde_dom.half_x[i];
        lde_x[i + lde_dom.half_n] = lde_dom.half_x[i];  // conjugate has same x
    }
    uint32_t* d_lde_x;
    CHECK(cudaMalloc(&d_lde_x, lde_n * sizeof(uint32_t)));
    CHECK(cudaMemcpyAsync(d_lde_x, lde_x, lde_n * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    free(lde_x);

    int half_n = n / 2;
    uint32_t x_b0 = trace_dom.half_x[half_n - 2];
    uint32_t x_b1 = trace_dom.half_x[half_n - 1];

    uint32_t *d_qevals;
    CHECK(cudaMalloc(&d_qevals, lde_n * sizeof(uint32_t)));

    t0 = (double)clock() / CLOCKS_PER_SEC;
    k_quotient<<<div_up(lde_n, threads), threads, 0, stream>>>(
        d_cvals, d_qevals, d_lde_x, x_b0, x_b1, log_trace, lde_n);
    CHECK(cudaStreamSynchronize(stream));
    t1 = (double)clock() / CLOCKS_PER_SEC;
    tm.t_quotient = t1 - t0;

    // 6. Merkle commit quotient
    t0 = (double)clock() / CLOCKS_PER_SEC;
    MerkleGPU q_tree = gpu_merkle_build(d_qevals, lde_n, stream,
                                        d_perms, d_found, d_slot, d_state);
    CHECK(cudaStreamSynchronize(stream));
    t1 = (double)clock() / CLOCKS_PER_SEC;
    tm.t_merkle_quot = t1 - t0;

    // 7. FRI fold chain on quotient evaluations
    t0 = (double)clock() / CLOCKS_PER_SEC;

    // Fiat-Shamir: derive alphas
    uint32_t quot_root[8];
    merkle_root(&q_tree, quot_root, stream);

    uint32_t fs[W] = {};
    memcpy(fs, trace_root, 8 * sizeof(uint32_t));
    memcpy(fs + 8, quot_root, 8 * sizeof(uint32_t));
    fs[16] = 0x46524931;
    h_poseidon2(fs, rc);

    uint64_t rng_state = (uint64_t)fs[0] | ((uint64_t)fs[1] << 32) | 1;
    auto rng_next = [&]() -> uint64_t {
        rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17; return rng_state;
    };

    // FRI: circle fold, then line folds
    uint32_t alpha0 = (uint32_t)rng_next() & P;

    // d_qevals has the quotient on LDE domain. FRI fold it.
    uint32_t *d_fri_cur = d_qevals;  // reuse buffer
    int fri_size = lde_n;

    // Circle fold: lde_n → lde_n/2
    uint32_t *d_fri_out;
    CHECK(cudaMalloc(&d_fri_out, (lde_n / 2) * sizeof(uint32_t)));
    k_fri_fold<<<div_up(fri_size/2, threads), threads, 0, stream>>>(
        d_fri_cur, d_fri_out, lde_inv_tw.d_flat, lde_inv_tw.offsets[0],
        alpha0, fri_size / 2);
    d_fri_cur = d_fri_out;
    fri_size /= 2;
    int tw_idx = 1;

    // Line folds until FRI_STOP — with Merkle commit + Fiat-Shamir per layer
    while (fri_size > FRI_STOP) {
        // Merkle commit current FRI layer
        MerkleGPU fri_tree = gpu_merkle_build(d_fri_cur, fri_size, stream,
                                               d_perms, d_found, d_slot, d_state);
        uint32_t fri_mk_root[8];
        merkle_root(&fri_tree, fri_mk_root, stream);

        // Fiat-Shamir: hash(fri_root || "FRI2") → alpha
        uint32_t fs2[W] = {};
        memcpy(fs2, fri_mk_root, 8 * sizeof(uint32_t));
        fs2[8] = 0x46524932;  // "FRI2"
        h_poseidon2(fs2, rc);
        uint32_t alpha = fs2[0] & P;

        merkle_free(&fri_tree);

        uint32_t *d_next;
        CHECK(cudaMalloc(&d_next, (fri_size / 2) * sizeof(uint32_t)));
        k_fri_fold<<<div_up(fri_size/2, threads), threads, 0, stream>>>(
            d_fri_cur, d_next, lde_inv_tw.d_flat, lde_inv_tw.offsets[tw_idx],
            alpha, fri_size / 2);
        if (d_fri_cur != d_qevals) cudaFree(d_fri_cur);
        d_fri_cur = d_next;
        fri_size /= 2;
        tw_idx++;
    }

    CHECK(cudaStreamSynchronize(stream));
    t1 = (double)clock() / CLOCKS_PER_SEC;
    tm.t_fri = t1 - t0;

    // Read total Poseidon2 perms
    unsigned long long total_perms;
    CHECK(cudaMemcpy(&total_perms, d_perms, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    tm.poseidon_perms = total_perms;

    double t_end = (double)clock() / CLOCKS_PER_SEC;
    tm.t_total = t_end - t_start;

    // Cleanup
    if (d_fri_cur != d_qevals) cudaFree(d_fri_cur);
    cudaFree(d_fri_out);
    cudaFree(d_qevals);
    cudaFree(d_cvals);
    cudaFree(d_lde_x);
    merkle_free(&t_tree);
    merkle_free(&q_tree);
    cudaFree(d_trace);
    cudaFree(d_scratch);
    cudaFree(d_perms);
    cudaFree(d_found);
    cudaFree(d_slot);
    cudaFree(d_state);
    free(trace);
    tw_free(&trace_inv_tw);
    tw_free(&lde_fwd_tw);
    tw_free(&lde_inv_tw);
    cd_free(&trace_dom);
    cd_free(&lde_dom);

    return tm;
}

// ═══════════════════════════════════════════════════════════════
// Pre-allocated STARK prover (complete: Fiat-Shamir + FRI Merkle + query)
// ═══════════════════════════════════════════════════════════════

struct FRILayerInfo {
    int input_size;       // values in this layer (before fold)
    int mk_n_leaves, mk_n_levels, mk_total;
    int *mk_offsets;      // level offsets in uint32_t count
    int pool_offset;      // word offset into d_fri_pool
};

struct StarkBufs {
    int log_trace, log_lde, n, lde_n;

    CDomain trace_dom, lde_dom;
    Twiddles trace_inv_tw, lde_fwd_tw, lde_inv_tw;

    uint32_t *d_a, *d_b, *d_c, *d_q;   // each lde_n words
    uint32_t *d_lde_x;                  // lde_n words (static)

    // Pre-allocated Merkle trees (trace + quotient)
    uint32_t *d_tree_t, *d_tree_q;
    int mk_n_leaves, mk_n_levels, mk_total;
    int *mk_offsets;  // level offsets in uint32_t count

    // FRI intermediate Merkle trees (pre-allocated)
    int n_fri_layers;
    FRILayerInfo *fri;
    uint32_t *d_fri_pool;   // single GPU alloc for all FRI trees

    // Counters
    unsigned long long *d_perms;
    int *d_found, *d_slot;
    uint32_t *d_state;

    uint32_t *h_trace;
    uint32_t x_b0, x_b1;

    // Fiat-Shamir
    const RC *rc;
};

StarkBufs stark_bufs_alloc(int log_trace, const RC* rc) {
    StarkBufs b = {};
    b.log_trace = log_trace;
    b.log_lde = log_trace + LOG_BLOWUP;
    b.n = 1 << log_trace;
    b.lde_n = 1 << b.log_lde;

    // Domains + twiddles (computed once, uploaded to device)
    b.trace_dom = cd_new(log_trace);
    b.lde_dom = cd_new(b.log_lde);
    b.trace_inv_tw = make_inv_twiddles(&b.trace_dom);
    b.lde_fwd_tw = make_fwd_twiddles(&b.lde_dom);
    b.lde_inv_tw = make_inv_twiddles(&b.lde_dom);

    // GPU work buffers
    int lde_n = b.lde_n;
    CHECK(cudaMalloc(&b.d_a, lde_n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&b.d_b, lde_n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&b.d_c, lde_n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&b.d_q, lde_n * sizeof(uint32_t)));

    // LDE x-coordinates (static — upload once)
    CHECK(cudaMalloc(&b.d_lde_x, lde_n * sizeof(uint32_t)));
    uint32_t* lde_x = (uint32_t*)malloc(lde_n * sizeof(uint32_t));
    for (int i = 0; i < b.lde_dom.half_n; i++) {
        lde_x[i] = b.lde_dom.half_x[i];
        lde_x[i + b.lde_dom.half_n] = b.lde_dom.half_x[i];
    }
    CHECK(cudaMemcpy(b.d_lde_x, lde_x, lde_n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    free(lde_x);

    // Merkle tree pre-allocation
    int n_leaves_raw = (lde_n + 7) / 8;
    b.mk_n_leaves = 1;
    while (b.mk_n_leaves < n_leaves_raw) b.mk_n_leaves <<= 1;
    b.mk_n_levels = 0;
    b.mk_total = 0;
    { int s = b.mk_n_leaves; while (s >= 1) { b.mk_total += s; b.mk_n_levels++; s /= 2; } }
    b.mk_offsets = (int*)malloc(b.mk_n_levels * sizeof(int));
    { int off = 0, s = b.mk_n_leaves;
      for (int k = 0; k < b.mk_n_levels; k++) { b.mk_offsets[k] = off * 8; off += s; s /= 2; } }

    CHECK(cudaMalloc(&b.d_tree_t, b.mk_total * 8 * sizeof(uint32_t)));
    CHECK(cudaMalloc(&b.d_tree_q, b.mk_total * 8 * sizeof(uint32_t)));

    // Counters
    CHECK(cudaMalloc(&b.d_perms, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&b.d_found, sizeof(int)));
    CHECK(cudaMalloc(&b.d_slot, sizeof(int)));
    CHECK(cudaMalloc(&b.d_state, W * sizeof(uint32_t)));

    // Host trace buffer
    b.h_trace = (uint32_t*)malloc(b.n * sizeof(uint32_t));

    // Boundary points (fixed for this trace size)
    int half_n = b.n / 2;
    b.x_b0 = b.trace_dom.half_x[half_n - 2];
    b.x_b1 = b.trace_dom.half_x[half_n - 1];

    // FRI intermediate Merkle trees (pre-allocated)
    // After circle fold: size = lde_n/2. Line folds commit while size > FRI_STOP.
    b.n_fri_layers = 0;
    { int s = lde_n / 2; while (s > FRI_STOP) { b.n_fri_layers++; s /= 2; } }

    b.fri = (FRILayerInfo*)calloc(b.n_fri_layers, sizeof(FRILayerInfo));
    int fri_pool_total = 0;
    { int s = lde_n / 2;
      for (int k = 0; k < b.n_fri_layers; k++) {
          FRILayerInfo& f = b.fri[k];
          f.input_size = s;
          int raw = (s + 7) / 8;
          f.mk_n_leaves = 1;
          while (f.mk_n_leaves < raw) f.mk_n_leaves <<= 1;
          f.mk_n_levels = 0;
          f.mk_total = 0;
          { int t = f.mk_n_leaves; while (t >= 1) { f.mk_total += t; f.mk_n_levels++; t /= 2; } }
          f.mk_offsets = (int*)malloc(f.mk_n_levels * sizeof(int));
          { int off2 = 0, t = f.mk_n_leaves;
            for (int j = 0; j < f.mk_n_levels; j++) { f.mk_offsets[j] = off2 * 8; off2 += t; t /= 2; } }
          f.pool_offset = fri_pool_total * 8;  // word offset
          fri_pool_total += f.mk_total;
          s /= 2;
      }
    }
    if (fri_pool_total > 0)
        CHECK(cudaMalloc(&b.d_fri_pool, fri_pool_total * 8 * sizeof(uint32_t)));
    else
        b.d_fri_pool = nullptr;

    b.rc = rc;

    return b;
}

// Run one COMPLETE STARK proof — Fiat-Shamir + FRI Merkle commits + query phase.
// Returns total Poseidon2 perm count (trace + quotient + FRI Merkle trees + Fiat-Shamir).
uint64_t stark_prove_fast(StarkBufs* b, uint32_t a0, uint32_t a1, cudaStream_t stream) {
    int n = b->n, lde_n = b->lde_n;
    int threads = 256;

    // Reset counters (async, no sync)
    CHECK(cudaMemsetAsync(b->d_perms, 0, sizeof(unsigned long long), stream));
    CHECK(cudaMemsetAsync(b->d_found, 0, sizeof(int), stream));

    // 1. Generate trace on CPU, upload
    b->h_trace[0] = a0; b->h_trace[1] = a1;
    for (int i = 2; i < n; i++) b->h_trace[i] = m31_add(b->h_trace[i-1], b->h_trace[i-2]);
    CHECK(cudaMemcpyAsync(b->d_a, b->h_trace, n * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    // 2. Compute constraint on GPU BEFORE iCFFT modifies trace
    k_constraint_vals<<<div_up(n, threads), threads, 0, stream>>>(b->d_a, b->d_c, n);

    // 3. iCFFT trace → coefficients
    gpu_icfft(b->d_a, b->d_b, b->log_trace, &b->trace_inv_tw, stream);

    // 4. Pad to LDE, CFFT → trace LDE evaluations
    CHECK(cudaMemsetAsync(b->d_a + n, 0, (lde_n - n) * sizeof(uint32_t), stream));
    gpu_cfft(b->d_a, b->d_b, b->log_lde, &b->lde_fwd_tw, stream);

    // 5. Merkle commit trace (pre-allocated tree)
    CHECK(cudaMemsetAsync(b->d_tree_t, 0, b->mk_total * 8 * sizeof(uint32_t), stream));
    CHECK(cudaMemcpyAsync(b->d_tree_t, b->d_a, lde_n * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice, stream));
    { int cur = b->mk_n_leaves;
      for (int k = 0; k < b->mk_n_levels - 1; k++) {
          int np = cur / 2;
          k_merkle_level<<<div_up(np, threads), threads, 0, stream>>>(
              b->d_tree_t + b->mk_offsets[k], b->d_tree_t + b->mk_offsets[k+1],
              np, b->d_perms, b->d_found, b->d_slot, b->d_state);
          cur = np;
      }
    }

    // 5a. SYNC + read trace_root (required for Fiat-Shamir)
    CHECK(cudaStreamSynchronize(stream));
    uint32_t trace_root[8];
    int t_root_off = b->mk_offsets[b->mk_n_levels - 1];
    CHECK(cudaMemcpy(trace_root, b->d_tree_t + t_root_off, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 6. iCFFT constraint → coefficients
    gpu_icfft(b->d_c, b->d_b, b->log_trace, &b->trace_inv_tw, stream);

    // 7. Pad + CFFT → constraint LDE
    CHECK(cudaMemsetAsync(b->d_c + n, 0, (lde_n - n) * sizeof(uint32_t), stream));
    gpu_cfft(b->d_c, b->d_b, b->log_lde, &b->lde_fwd_tw, stream);

    // 8. Constraint quotient
    k_quotient<<<div_up(lde_n, threads), threads, 0, stream>>>(
        b->d_c, b->d_q, b->d_lde_x, b->x_b0, b->x_b1, b->log_trace, lde_n);

    // 9. Merkle commit quotient (pre-allocated)
    CHECK(cudaMemsetAsync(b->d_tree_q, 0, b->mk_total * 8 * sizeof(uint32_t), stream));
    CHECK(cudaMemcpyAsync(b->d_tree_q, b->d_q, lde_n * sizeof(uint32_t),
                          cudaMemcpyDeviceToDevice, stream));
    { int cur = b->mk_n_leaves;
      for (int k = 0; k < b->mk_n_levels - 1; k++) {
          int np = cur / 2;
          k_merkle_level<<<div_up(np, threads), threads, 0, stream>>>(
              b->d_tree_q + b->mk_offsets[k], b->d_tree_q + b->mk_offsets[k+1],
              np, b->d_perms, b->d_found, b->d_slot, b->d_state);
          cur = np;
      }
    }

    // 9a. SYNC + read quotient_root
    CHECK(cudaStreamSynchronize(stream));
    uint32_t quot_root[8];
    CHECK(cudaMemcpy(quot_root, b->d_tree_q + t_root_off, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 10. Fiat-Shamir: derive FRI alphas from trace_root + quotient_root
    uint32_t fs[W] = {};
    memcpy(fs, trace_root, 8 * sizeof(uint32_t));
    memcpy(fs + 8, quot_root, 8 * sizeof(uint32_t));
    fs[16] = 0x46524931;  // "FRI1"
    h_poseidon2(fs, b->rc);

    uint64_t rng_state = (uint64_t)fs[0] | ((uint64_t)fs[1] << 32) | 1;
    auto rng_next = [&]() -> uint64_t {
        rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17; return rng_state;
    };

    // 11. Circle fold with REAL alpha0 (quotient → d_a, lde_n → lde_n/2)
    uint32_t alpha0 = (uint32_t)rng_next() & P;
    k_fri_fold<<<div_up(lde_n/2, threads), threads, 0, stream>>>(
        b->d_q, b->d_a, b->lde_inv_tw.d_flat, b->lde_inv_tw.offsets[0],
        alpha0, lde_n / 2);

    // 12. FRI loop: Merkle commit + Fiat-Shamir + fold (matches CPU cstark_prove)
    uint32_t *fri_cur = b->d_a;
    int tw_idx = 1;
    for (int k = 0; k < b->n_fri_layers; k++) {
        FRILayerInfo& f = b->fri[k];
        uint32_t *d_tree = b->d_fri_pool + f.pool_offset;

        // 12a. Merkle commit current FRI layer (pre-allocated tree)
        CHECK(cudaMemsetAsync(d_tree, 0, f.mk_total * 8 * sizeof(uint32_t), stream));
        CHECK(cudaMemcpyAsync(d_tree, fri_cur, f.input_size * sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice, stream));
        { int cur = f.mk_n_leaves;
          for (int j = 0; j < f.mk_n_levels - 1; j++) {
              int np = cur / 2;
              k_merkle_level<<<div_up(np, threads), threads, 0, stream>>>(
                  d_tree + f.mk_offsets[j], d_tree + f.mk_offsets[j+1],
                  np, b->d_perms, b->d_found, b->d_slot, b->d_state);
              cur = np;
          }
        }

        // 12b. SYNC + read FRI root
        CHECK(cudaStreamSynchronize(stream));
        uint32_t fri_root[8];
        int root_off = f.mk_offsets[f.mk_n_levels - 1];
        CHECK(cudaMemcpy(fri_root, d_tree + root_off, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // 12c. Fiat-Shamir: hash(fri_root || "FRI2") → alpha
        uint32_t fs2[W] = {};
        memcpy(fs2, fri_root, 8 * sizeof(uint32_t));
        fs2[8] = 0x46524932;  // "FRI2"
        h_poseidon2(fs2, b->rc);
        uint32_t alpha = fs2[0] & P;

        // 12d. FRI line fold with real alpha (ping-pong fri_cur → other buffer)
        uint32_t *next = (fri_cur == b->d_a) ? b->d_c : b->d_a;
        int half = f.input_size / 2;
        k_fri_fold<<<div_up(half, threads), threads, 0, stream>>>(
            fri_cur, next, b->lde_inv_tw.d_flat, b->lde_inv_tw.offsets[tw_idx],
            alpha, half);
        fri_cur = next;
        tw_idx++;
    }

    // 13. Read FRI final values (<=FRI_STOP values)
    int fri_final_size = b->n_fri_layers > 0 ? b->fri[b->n_fri_layers - 1].input_size / 2 : lde_n / 2;
    uint32_t fri_last[FRI_STOP];
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(fri_last, fri_cur, fri_final_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 14. Query phase: derive positions and extract Merkle authentication paths
    //     This is the decommitment — proof is incomplete without it.
    for (int q = 0; q < N_QUERIES; q++) {
        int pos = (int)(rng_next() % (uint64_t)lde_n);
        int t_leaf = (pos / 8) < b->mk_n_leaves ? (pos / 8) : b->mk_n_leaves - 1;

        // Extract trace auth path (read sibling at each level)
        { int idx = t_leaf;
          for (int lv = 0; lv < b->mk_n_levels - 1; lv++) {
              int sib = (idx ^ 1) < (b->mk_n_leaves >> lv) ? (idx ^ 1) : idx;
              uint32_t sib_node[8];
              CHECK(cudaMemcpy(sib_node, b->d_tree_t + b->mk_offsets[lv] + sib * 8,
                               8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
              idx >>= 1;
          }
        }

        // Extract quotient auth path (same tree structure)
        { int idx = t_leaf;
          for (int lv = 0; lv < b->mk_n_levels - 1; lv++) {
              int sib = (idx ^ 1) < (b->mk_n_leaves >> lv) ? (idx ^ 1) : idx;
              uint32_t sib_node[8];
              CHECK(cudaMemcpy(sib_node, b->d_tree_q + b->mk_offsets[lv] + sib * 8,
                               8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
              idx >>= 1;
          }
        }

        // Extract quotient conjugate path (circle fold partner)
        { int conj_pos = pos < lde_n / 2 ? pos + lde_n / 2 : pos - lde_n / 2;
          int conj_leaf = (conj_pos / 8) < b->mk_n_leaves ? (conj_pos / 8) : b->mk_n_leaves - 1;
          int idx = conj_leaf;
          for (int lv = 0; lv < b->mk_n_levels - 1; lv++) {
              int sib = (idx ^ 1) < (b->mk_n_leaves >> lv) ? (idx ^ 1) : idx;
              uint32_t sib_node[8];
              CHECK(cudaMemcpy(sib_node, b->d_tree_q + b->mk_offsets[lv] + sib * 8,
                               8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
              idx >>= 1;
          }
        }

        // Extract FRI layer openings + fold partner paths
        int cur_pos = pos % (lde_n / 2);
        int cur_size = lde_n / 2;
        for (int k = 0; k < b->n_fri_layers; k++) {
            FRILayerInfo& f = b->fri[k];
            uint32_t *d_tree = b->d_fri_pool + f.pool_offset;
            int leaf = (cur_pos / 8) < f.mk_n_leaves ? (cur_pos / 8) : f.mk_n_leaves - 1;

            // Opening path
            { int idx = leaf;
              for (int lv = 0; lv < f.mk_n_levels - 1; lv++) {
                  int sib = (idx ^ 1) < (f.mk_n_leaves >> lv) ? (idx ^ 1) : idx;
                  uint32_t sib_node[8];
                  CHECK(cudaMemcpy(sib_node, d_tree + f.mk_offsets[lv] + sib * 8,
                                   8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                  idx >>= 1;
              }
            }

            // Fold partner path
            int half = cur_size / 2;
            int p_pos = cur_pos < half ? cur_pos + half : cur_pos - half;
            int p_leaf = (p_pos / 8) < f.mk_n_leaves ? (p_pos / 8) : f.mk_n_leaves - 1;
            { int idx = p_leaf;
              for (int lv = 0; lv < f.mk_n_levels - 1; lv++) {
                  int sib = (idx ^ 1) < (f.mk_n_leaves >> lv) ? (idx ^ 1) : idx;
                  uint32_t sib_node[8];
                  CHECK(cudaMemcpy(sib_node, d_tree + f.mk_offsets[lv] + sib * 8,
                                   8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                  idx >>= 1;
              }
            }

            cur_pos %= half;
            cur_size = half;
        }
    }

    // 15. Total Poseidon2 perms (Merkle trees: trace + quotient + FRI + Fiat-Shamir)
    unsigned long long perms;
    CHECK(cudaMemcpy(&perms, b->d_perms, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    perms += 1 + b->n_fri_layers;  // Fiat-Shamir: 1 initial + n per FRI layer
    return perms;
}

void stark_bufs_free(StarkBufs* b) {
    cudaFree(b->d_a); cudaFree(b->d_b); cudaFree(b->d_c); cudaFree(b->d_q);
    cudaFree(b->d_lde_x);
    cudaFree(b->d_tree_t); cudaFree(b->d_tree_q);
    if (b->d_fri_pool) cudaFree(b->d_fri_pool);
    for (int k = 0; k < b->n_fri_layers; k++) free(b->fri[k].mk_offsets);
    free(b->fri);
    cudaFree(b->d_perms); cudaFree(b->d_found); cudaFree(b->d_slot); cudaFree(b->d_state);
    free(b->h_trace);
    free(b->mk_offsets);
    tw_free(&b->trace_inv_tw); tw_free(&b->lde_fwd_tw); tw_free(&b->lde_inv_tw);
    cd_free(&b->trace_dom); cd_free(&b->lde_dom);
}

// ═══════════════════════════════════════════════════════════════
// Host: Benchmark
// ═══════════════════════════════════════════════════════════════

double wall_time() {
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

// ── 3-Mode measurement result ──
struct SweepResult {
    int log_trace;
    double stark_ms;
    uint64_t stark_perms;
    int stark_count;
    double mode_a, mode_b, mode_c;  // perm/s
    double zk_overhead, sym_recovery, net_cost;  // percent
    double f_sym, spread;
};

// Run 3-mode comparison for a given trace size.
// Prints per-phase detail if verbose.
SweepResult run_3mode(int log_trace, int difficulty, double run_sec,
                      const RC* rc, int sms, bool verbose)
{
    char buf[32];
    cudaStream_t s_stark, s_pow;
    CHECK(cudaStreamCreate(&s_stark));
    CHECK(cudaStreamCreate(&s_pow));

    int pow_threads = 256;
    int lde_n = 1 << (log_trace + LOG_BLOWUP);
    int pow_batch = lde_n;
    int pow_batch_blocks = div_up(pow_batch, pow_threads);

    // PoW counters
    int *d_pow_found, *d_pow_slot;
    uint32_t *d_pow_state;
    CHECK(cudaMalloc(&d_pow_found, sizeof(int)));
    CHECK(cudaMalloc(&d_pow_slot, sizeof(int)));
    CHECK(cudaMalloc(&d_pow_state, W * sizeof(uint32_t)));

    // Phase 1: STARK timing (1 warm-up + 1 measured)
    gpu_stark_prove(log_trace, 1, 1, rc, s_stark);  // warm up
    StarkTiming st = gpu_stark_prove(log_trace, 100, 1, rc, s_stark);

    // Pre-allocate STARK buffers
    StarkBufs sbufs = stark_bufs_alloc(log_trace, rc);
    stark_prove_fast(&sbufs, 1, 1, s_stark);  // warm up

    // Helper: batched Pure PoW measurement
    auto run_pure_pow = [&]() -> double {
        CHECK(cudaMemset(d_pow_found, 0, sizeof(int)));
        k_pow_batch<<<pow_batch_blocks, pow_threads, 0, s_pow>>>(
            0, pow_batch, d_pow_found, d_pow_slot, d_pow_state);
        CHECK(cudaStreamSynchronize(s_pow));

        uint64_t total_perms = 0, nonce = 0;
        int launches = 0;
        double t0 = wall_time();
        while (wall_time() - t0 < run_sec) {
            k_pow_batch<<<pow_batch_blocks, pow_threads, 0, s_pow>>>(
                nonce, pow_batch, d_pow_found, d_pow_slot, d_pow_state);
            nonce += pow_batch;
            total_perms += pow_batch;
            launches++;
            if (launches % 50 == 0) CHECK(cudaStreamSynchronize(s_pow));
        }
        CHECK(cudaStreamSynchronize(s_pow));
        return total_perms / (wall_time() - t0);
    };

    // Phase 2: Pure PoW (pre)
    if (verbose) printf("\n  [2^%d] Phase 2: Pure PoW (pre)...", log_trace);
    double pre = run_pure_pow();
    if (verbose) printf(" %.2f Mperm/s\n", pre / 1e6);

    // Phase 3: Symbiotic
    if (verbose) printf("  [2^%d] Phase 3: Symbiotic...", log_trace);
    CHECK(cudaMemset(d_pow_found, 0, sizeof(int)));

    double tw0 = wall_time();
    uint64_t stark_perms_total = 0, sym_pow_perms = 0;
    int stark_count = 0, pow_launches = 0;
    uint64_t pow_nonce = 0;

    while (wall_time() - tw0 < run_sec) {
        uint64_t p = stark_prove_fast(&sbufs, 200 + stark_count, 1, s_stark);
        stark_perms_total += p;
        stark_count++;
        for (int b = 0; b < 5; b++) {
            k_pow_batch<<<pow_batch_blocks, pow_threads, 0, s_pow>>>(
                pow_nonce, pow_batch, d_pow_found, d_pow_slot, d_pow_state);
            pow_nonce += pow_batch;
            sym_pow_perms += pow_batch;
            pow_launches++;
        }
    }
    CHECK(cudaStreamSynchronize(s_pow));
    CHECK(cudaStreamSynchronize(s_stark));
    double sym_elapsed = wall_time() - tw0;

    double pow_only_rate = sym_pow_perms / sym_elapsed;
    double sym_rate = (sym_pow_perms + stark_perms_total) / sym_elapsed;
    double f_sym = (double)stark_perms_total / (sym_pow_perms + stark_perms_total);
    if (verbose) printf(" %.2f Mperm/s (%d proofs)\n", sym_rate / 1e6, stark_count);

    // Phase 4: Pure PoW (post)
    if (verbose) printf("  [2^%d] Phase 4: Pure PoW (post)...", log_trace);
    double post = run_pure_pow();
    if (verbose) printf(" %.2f Mperm/s\n", post / 1e6);

    // Cleanup
    cudaFree(d_pow_found); cudaFree(d_pow_slot); cudaFree(d_pow_state);
    stark_bufs_free(&sbufs);
    CHECK(cudaStreamDestroy(s_stark));
    CHECK(cudaStreamDestroy(s_pow));

    // Compute result
    double pure_avg = (pre + post) / 2.0;
    double spread = fabs(pre - post) / pure_avg * 100;

    SweepResult r = {};
    r.log_trace = log_trace;
    r.stark_ms = st.t_total * 1e3;
    r.stark_perms = st.poseidon_perms;
    r.stark_count = stark_count;
    r.mode_a = pure_avg;
    r.mode_b = pow_only_rate;
    r.mode_c = sym_rate;
    r.zk_overhead = (1.0 - r.mode_b / r.mode_a) * 100;
    r.sym_recovery = (r.mode_c / r.mode_b - 1.0) * 100;
    r.net_cost = (1.0 - r.mode_c / r.mode_a) * 100;
    r.f_sym = f_sym * 100;
    r.spread = spread;
    return r;
}

int main(int argc, char** argv) {
    // Detect sweep mode: ./stark sweep [difficulty] [seconds]
    bool sweep_mode = argc > 1 && strcmp(argv[1], "sweep") == 0;

    int log_trace = 16;
    int difficulty = 24;
    double run_sec = 5.0;

    if (sweep_mode) {
        difficulty = argc > 2 ? atoi(argv[2]) : 24;
        run_sec = argc > 3 ? atof(argv[3]) : 3.0;
    } else {
        log_trace = argc > 1 ? atoi(argv[1]) : 16;
        difficulty = argc > 2 ? atoi(argv[2]) : 24;
        run_sec = argc > 3 ? atof(argv[3]) : 5.0;
    }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;

    printf("ZK-SPoW Full GPU STARK — Poseidon2 Width-24 / M31\n");
    printf("════════════════════════════════════════════════════\n");
    printf("GPU          : %s (%d SMs, %d MHz)\n", prop.name, sms, prop.clockRate/1000);
    printf("Memory       : %.1f GB, L2 %.1f MB\n",
        prop.totalGlobalMem/1e9, prop.l2CacheSize/1e6);
    printf("────────────────────────────────────────────────────\n");
    if (sweep_mode) {
        printf("Mode         : SWEEP (k=8,10,12,...,22)\n");
    } else {
        printf("Trace        : 2^%d = %d\n", log_trace, 1 << log_trace);
        printf("LDE          : 2^%d = %d (%dx blowup)\n",
            log_trace + LOG_BLOWUP, 1 << (log_trace + LOG_BLOWUP), 1 << LOG_BLOWUP);
    }
    printf("Difficulty   : %d bits\n", difficulty);
    printf("FRI queries  : %d, fold to <= %d\n", N_QUERIES, FRI_STOP);
    printf("Poseidon2    : W=%d, Rf=%d, Rp=%d\n", W, RF, RP);
    printf("Run time     : %.1f sec/size\n", run_sec);
    printf("════════════════════════════════════════════════════\n");

    // Setup constants
    RC rc; gen_rc(&rc);
    uint32_t h_h[8] = {0x5EADBEEF & P, 0x4AFEBABE & P, 0x12345678, 0x1ABCDEF0 & P, 42,2026,2,19};
    uint32_t target[8]; for (int i = 0; i < 8; i++) target[i] = P-1;
    target[0] = difficulty >= 31 ? 0 : P >> difficulty;

    CHECK(cudaMemcpyToSymbol(d_rc_ext, rc.ext, sizeof(rc.ext)));
    CHECK(cudaMemcpyToSymbol(d_rc_int, rc.intr, sizeof(rc.intr)));
    CHECK(cudaMemcpyToSymbol(d_diag, DIAG, sizeof(DIAG)));
    CHECK(cudaMemcpyToSymbol(d_h_h, h_h, sizeof(h_h)));
    CHECK(cudaMemcpyToSymbol(d_target, target, sizeof(target)));

    char buf[32];

    if (sweep_mode) {
        // ═══════════════════════════════════════════════════════════
        // Sweep mode: run 3-mode comparison across multiple trace sizes
        // ═══════════════════════════════════════════════════════════
        const int sizes[] = {8, 10, 12, 14, 16, 18, 20, 22};
        const int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
        SweepResult results[n_sizes];

        for (int i = 0; i < n_sizes; i++) {
            int k = sizes[i];
            printf("\n════ Sweep k=%d (2^%d = %d) ════════════════════\n",
                k, k, 1 << k);
            results[i] = run_3mode(k, difficulty, run_sec, &rc, sms, true);
            SweepResult& r = results[i];
            printf("  → A=%.1f  B=%.1f  C=%.1f Mperm/s  ZK=%+.1f%%  Sym=%+.1f%%  Net=%+.1f%%\n",
                r.mode_a / 1e6, r.mode_b / 1e6, r.mode_c / 1e6,
                -r.zk_overhead, r.sym_recovery, -r.net_cost);
        }

        // Summary table
        printf("\n\n════════════════════════════════════════════════════════════════════════════\n");
        printf("ZK-SPoW Sweep Summary — 3-Mode Comparison\n");
        printf("════════════════════════════════════════════════════════════════════════════\n");
        printf("  k   trace   STARK     perms   A(Pure)   B(ZK+PoW)   C(ZK-SPoW)  ZK cost  Sym rec  Net cost  f_sym\n");
        printf("────────────────────────────────────────────────────────────────────────────────────────────────────────\n");
        for (int i = 0; i < n_sizes; i++) {
            SweepResult& r = results[i];
            printf(" %2d  %6d  %6.2fms  %6s  %7.1f   %7.1f     %7.1f    %+5.1f%%   %+5.1f%%   %+5.1f%%   %4.1f%%\n",
                r.log_trace, 1 << r.log_trace,
                r.stark_ms, fmt(r.stark_perms, buf),
                r.mode_a / 1e6, r.mode_b / 1e6, r.mode_c / 1e6,
                -r.zk_overhead, r.sym_recovery, -r.net_cost, r.f_sym);
        }
        printf("════════════════════════════════════════════════════════════════════════════\n");
        printf("  A = Pure PoW (Mperm/s)     B = ZK + PoW, no sym (Mperm/s)\n");
        printf("  C = ZK-SPoW, sym (Mperm/s) ZK cost = A→B   Sym rec = B→C   Net = A→C\n");
        printf("════════════════════════════════════════════════════════════════════════════\n");

        // Check if Sym recovery is positive across all sizes
        bool all_positive = true;
        for (int i = 0; i < n_sizes; i++)
            if (results[i].sym_recovery <= 0) all_positive = false;
        if (all_positive)
            printf("RESULT: Symbiotic recovery positive at ALL sizes — ZK-SPoW universally beneficial\n");
        else
            printf("RESULT: Symbiotic recovery NOT universally positive — check small sizes\n");

    } else {
        // ═══════════════════════════════════════════════════════════
        // Single mode: detailed Phase 1 timing + 3-mode comparison
        // ═══════════════════════════════════════════════════════════
        cudaStream_t s_stark;
        CHECK(cudaStreamCreate(&s_stark));

        // ── Phase 1: Full STARK proof timing ──
        printf("\n── Phase 1: Full GPU STARK Proof ───────────────\n");

        // Warm up
        gpu_stark_prove(log_trace, 1, 1, &rc, s_stark);

        // Actual measurement (average of 3 runs)
        StarkTiming avg = {};
        int runs = 3;
        for (int r = 0; r < runs; r++) {
            StarkTiming t = gpu_stark_prove(log_trace, 100 + r, 1, &rc, s_stark);
            avg.t_icfft += t.t_icfft;
            avg.t_cfft += t.t_cfft;
            avg.t_merkle_trace += t.t_merkle_trace;
            avg.t_quotient += t.t_quotient;
            avg.t_merkle_quot += t.t_merkle_quot;
            avg.t_fri += t.t_fri;
            avg.t_total += t.t_total;
            avg.poseidon_perms += t.poseidon_perms;
        }
        avg.t_icfft /= runs; avg.t_cfft /= runs;
        avg.t_merkle_trace /= runs; avg.t_quotient /= runs;
        avg.t_merkle_quot /= runs; avg.t_fri /= runs;
        avg.t_total /= runs; avg.poseidon_perms /= runs;

        printf("Component breakdown (avg of %d runs):\n", runs);
        printf("  iCFFT        : %8.3f ms\n", avg.t_icfft * 1e3);
        printf("  CFFT (LDE)   : %8.3f ms\n", avg.t_cfft * 1e3);
        printf("  Merkle trace : %8.3f ms  (Poseidon2)\n", avg.t_merkle_trace * 1e3);
        printf("  Quotient     : %8.3f ms\n", avg.t_quotient * 1e3);
        printf("  Merkle quot  : %8.3f ms  (Poseidon2)\n", avg.t_merkle_quot * 1e3);
        printf("  FRI fold     : %8.3f ms\n", avg.t_fri * 1e3);
        printf("  ─────────────────────────\n");
        printf("  Total        : %8.3f ms\n", avg.t_total * 1e3);
        printf("  Poseidon2    : %s perms\n", fmt(avg.poseidon_perms, buf));

        double stark_pps = avg.poseidon_perms / avg.t_total;
        printf("  STARK Mperm/s: %.2f\n", stark_pps / 1e6);

        double t_poseidon = avg.t_merkle_trace + avg.t_merkle_quot;
        double t_memory = avg.t_icfft + avg.t_cfft + avg.t_quotient;
        printf("\n  Poseidon2 time: %.1f%% (Merkle trace+quot)\n", t_poseidon / avg.t_total * 100);
        printf("  FRI time       : %.1f%% (Merkle commits + folds)\n", avg.t_fri / avg.t_total * 100);
        printf("  M31/memory time: %.1f%% (FFT + quotient)\n", t_memory / avg.t_total * 100);

        // ── Phases 2–4: 3-Mode Comparison ──
        SweepResult sr = run_3mode(log_trace, difficulty, run_sec, &rc, sms, true);

        // Summary
        printf("\n════════════════════════════════════════════════════\n");
        printf("ZK-SPoW 3-Mode Comparison (trace=2^%d)\n", log_trace);
        printf("════════════════════════════════════════════════════\n");
        printf("STARK proof      : %6.2f ms/proof  (%s Poseidon2 perms)\n",
            avg.t_total * 1e3, fmt(avg.poseidon_perms, buf));
        printf("  Merkle t+q     : %5.1f%%\n", t_poseidon / avg.t_total * 100);
        printf("  FRI (Mk+fold)  : %5.1f%%\n", avg.t_fri / avg.t_total * 100);
        printf("  FFT+quotient   : %5.1f%%\n", t_memory / avg.t_total * 100);
        printf("────────────────────────────────────────────────────\n");
        printf("A. Pure PoW        : %8.2f Mperm/s  (100.0%%)\n", sr.mode_a / 1e6);
        printf("   spread          : %.1f%%\n", sr.spread);
        printf("B. ZK + PoW (no sym): %7.2f Mperm/s  (%5.1f%%)\n",
            sr.mode_b / 1e6, sr.mode_b / sr.mode_a * 100);
        printf("C. ZK-SPoW (sym)   : %8.2f Mperm/s  (%5.1f%%)\n",
            sr.mode_c / 1e6, sr.mode_c / sr.mode_a * 100);
        printf("────────────────────────────────────────────────────\n");
        printf("ZK overhead  (A→B): %+.1f%%  (cost of running STARK proofs)\n", -sr.zk_overhead);
        printf("Sym recovery (B→C): %+.1f%%  (tickets from Merkle hashing)\n", sr.sym_recovery);
        printf("Net cost     (A→C): %+.1f%%\n", -sr.net_cost);
        printf("────────────────────────────────────────────────────\n");
        printf("f_sym              : %.1f%%  (STARK fraction of total perms)\n", sr.f_sym);
        printf("────────────────────────────────────────────────────\n");

        if (sr.net_cost <= 5.0)
            printf("RESULT: ZK nearly free (net cost %.1f%%)\n", sr.net_cost);
        else if (sr.sym_recovery > 1.0)
            printf("RESULT: Symbiotic recovers %+.1f%% — ZK-SPoW reduces ZK cost\n", sr.sym_recovery);
        else
            printf("RESULT: Symbiotic gain minimal (%.1f%%) — bottleneck not complementary\n", sr.sym_recovery);

        if (sr.spread > 10.0)
            printf("WARNING: Baseline unstable (%.1f%% spread) — results unreliable\n", sr.spread);

        printf("════════════════════════════════════════════════════\n");

        CHECK(cudaStreamDestroy(s_stark));
    }

    return 0;
}
