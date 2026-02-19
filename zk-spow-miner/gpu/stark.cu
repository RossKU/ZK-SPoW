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
// Example: ./stark 16 24 10

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

    // Line folds until FRI_STOP
    while (fri_size > FRI_STOP) {
        // Derive next alpha
        uint32_t alpha = (uint32_t)rng_next() & P;
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

int main(int argc, char** argv) {
    int log_trace = argc > 1 ? atoi(argv[1]) : 16;
    int difficulty = argc > 2 ? atoi(argv[2]) : 24;
    double run_sec = argc > 3 ? atof(argv[3]) : 5.0;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;

    printf("ZK-SPoW Full GPU STARK — Poseidon2 Width-24 / M31\n");
    printf("════════════════════════════════════════════════════\n");
    printf("GPU          : %s (%d SMs, %d MHz)\n", prop.name, sms, prop.clockRate/1000);
    printf("Memory       : %.1f GB, L2 %.1f MB\n",
        prop.totalGlobalMem/1e9, prop.l2CacheSize/1e6);
    printf("────────────────────────────────────────────────────\n");
    printf("Trace        : 2^%d = %d\n", log_trace, 1 << log_trace);
    printf("LDE          : 2^%d = %d (%dx blowup)\n",
        log_trace + LOG_BLOWUP, 1 << (log_trace + LOG_BLOWUP), 1 << LOG_BLOWUP);
    printf("Difficulty   : %d bits\n", difficulty);
    printf("FRI queries  : %d, fold to <= %d\n", N_QUERIES, FRI_STOP);
    printf("Poseidon2    : W=%d, Rf=%d, Rp=%d\n", W, RF, RP);
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
    cudaStream_t s_stark, s_pow;
    CHECK(cudaStreamCreate(&s_stark));
    CHECK(cudaStreamCreate(&s_pow));

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

    // Time breakdown: Poseidon2 (Merkle) vs M31-only (FFT/quotient/FRI)
    double t_poseidon = avg.t_merkle_trace + avg.t_merkle_quot;
    double t_memory = avg.t_icfft + avg.t_cfft + avg.t_quotient + avg.t_fri;
    printf("\n  Poseidon2 time: %.1f%% (Merkle)\n", t_poseidon / avg.t_total * 100);
    printf("  M31/memory time: %.1f%% (FFT + quotient + FRI)\n", t_memory / avg.t_total * 100);

    // ── Phase 2: Pure PoW baseline ──
    printf("\n── Phase 2: Pure PoW Baseline (%.1fs) ──────────\n", run_sec);

    int pow_blocks = sms * 4, pow_threads = 256;
    int *d_pow_stop, *d_pow_slot;
    unsigned long long *d_pow_perms, *d_pow_nonce;
    CHECK(cudaMalloc(&d_pow_stop, sizeof(int)));
    CHECK(cudaMalloc(&d_pow_perms, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_pow_nonce, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_pow_slot, sizeof(int)));
    CHECK(cudaMemset(d_pow_stop, 0, sizeof(int)));
    CHECK(cudaMemset(d_pow_perms, 0, sizeof(unsigned long long)));

    k_pow<<<pow_blocks, pow_threads, 0, s_pow>>>(0, d_pow_stop, d_pow_perms, d_pow_nonce, d_pow_slot);
    double tw0 = wall_time();
    while (wall_time() - tw0 < run_sec) usleep(50000);
    int one = 1;
    CHECK(cudaMemcpy(d_pow_stop, &one, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaStreamSynchronize(s_pow));
    double pow_elapsed = wall_time() - tw0;

    unsigned long long pure_pow_perms;
    CHECK(cudaMemcpy(&pure_pow_perms, d_pow_perms, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    double pure_pow_rate = pure_pow_perms / pow_elapsed;

    printf("Perms        : %s\n", fmt(pure_pow_perms, buf));
    printf("Time         : %.3fs\n", pow_elapsed);
    printf("Rate         : %.2f Mperm/s\n", pure_pow_rate / 1e6);

    cudaFree(d_pow_stop); cudaFree(d_pow_perms); cudaFree(d_pow_nonce); cudaFree(d_pow_slot);

    // ── Phase 3: Symbiotic (STARK proofs + PoW concurrent) ──
    printf("\n── Phase 3: Symbiotic (STARK + PoW, %.1fs) ─────\n", run_sec);

    // Launch PoW on half the GPU
    int sym_pow_blocks = sms * 2;  // half GPU for PoW
    CHECK(cudaMalloc(&d_pow_stop, sizeof(int)));
    CHECK(cudaMalloc(&d_pow_perms, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_pow_nonce, sizeof(unsigned long long)));
    CHECK(cudaMalloc(&d_pow_slot, sizeof(int)));
    CHECK(cudaMemset(d_pow_stop, 0, sizeof(int)));
    CHECK(cudaMemset(d_pow_perms, 0, sizeof(unsigned long long)));

    k_pow<<<sym_pow_blocks, pow_threads, 0, s_pow>>>(
        0, d_pow_stop, d_pow_perms, d_pow_nonce, d_pow_slot);

    // Run STARK proofs on s_stark stream simultaneously
    tw0 = wall_time();
    uint64_t stark_perms_total = 0;
    int stark_count = 0;
    while (wall_time() - tw0 < run_sec) {
        StarkTiming t = gpu_stark_prove(log_trace, 200 + stark_count, 1, &rc, s_stark);
        stark_perms_total += t.poseidon_perms;
        stark_count++;
    }

    // Stop PoW
    CHECK(cudaMemcpy(d_pow_stop, &one, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaStreamSynchronize(s_pow));
    double sym_elapsed = wall_time() - tw0;

    unsigned long long sym_pow_perms;
    CHECK(cudaMemcpy(&sym_pow_perms, d_pow_perms, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    uint64_t sym_total = sym_pow_perms + stark_perms_total;
    double sym_rate = sym_total / sym_elapsed;

    printf("STARK proofs : %d (in %.1fs)\n", stark_count, sym_elapsed);
    printf("STARK perms  : %s (Poseidon2)\n", fmt(stark_perms_total, buf));
    printf("PoW perms    : %s\n", fmt(sym_pow_perms, buf));
    printf("Total perms  : %s\n", fmt(sym_total, buf));
    printf("Rate         : %.2f Mperm/s\n", sym_rate / 1e6);

    double f_sym = (double)stark_perms_total / sym_total * 100;
    printf("f_sym        : %.1f%%\n", f_sym);

    cudaFree(d_pow_stop); cudaFree(d_pow_perms); cudaFree(d_pow_nonce); cudaFree(d_pow_slot);

    // ── Summary ──
    printf("\n════════════════════════════════════════════════════\n");
    printf("RESULTS SUMMARY (trace=2^%d, diff=%d)\n", log_trace, difficulty);
    printf("────────────────────────────────────────────────────\n");
    printf("STARK proof  : %8.2f ms/proof (%s Poseidon2 perms)\n",
        avg.t_total * 1e3, fmt(avg.poseidon_perms, buf));
    printf("  Poseidon2  : %5.1f%% of proof time (Merkle trees)\n",
        t_poseidon / avg.t_total * 100);
    printf("  Memory ops : %5.1f%% of proof time (FFT/quotient/FRI)\n",
        t_memory / avg.t_total * 100);
    printf("────────────────────────────────────────────────────\n");
    printf("Pure PoW     : %8.2f Mperm/s  (baseline)\n", pure_pow_rate / 1e6);
    printf("Symbiotic    : %8.2f Mperm/s  (%.1f%% of baseline)\n",
        sym_rate / 1e6, sym_rate / pure_pow_rate * 100);
    printf("────────────────────────────────────────────────────\n");

    double overhead = (1.0 - sym_rate / pure_pow_rate) * 100;
    if (overhead < 0) overhead = 0;
    printf("Symbiotic overhead : %.1f%%\n", overhead);
    printf("f_sym (STARK frac) : %.1f%%\n", f_sym);

    if (sym_rate >= pure_pow_rate * 0.90)
        printf("VERDICT : Complementary bottleneck CONFIRMED (<10%% overhead)\n");
    else if (sym_rate >= pure_pow_rate * 0.70)
        printf("VERDICT : Partial overlap (%.0f%% overhead)\n", overhead);
    else
        printf("VERDICT : Bottleneck NOT complementary (%.0f%% overhead)\n", overhead);

    printf("════════════════════════════════════════════════════\n");

    CHECK(cudaStreamDestroy(s_stark));
    CHECK(cudaStreamDestroy(s_pow));
    return 0;
}
