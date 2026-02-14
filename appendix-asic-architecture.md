# Appendix A: ASIC Architecture Details

**Companion to: ZK-PoUW v0.1**

---

## A.1 ZK-Symbiotic ASIC Block Diagram

```
┌─ ZK-PoUW ASIC (7nm, ~200W) ──────────────────────────────────────────┐
│                                                                        │
│  ┌─ Poseidon2 Core Array (50% die) ────────────────────────────────┐  │
│  │                                                                  │  │
│  │  [Core 0] [Core 1] [Core 2] ... [Core N-1]                      │  │
│  │                                                                  │  │
│  │  Each core: Width-24 Poseidon2 pipeline (M31)                   │  │
│  │  Throughput: 1 hash / pipeline_depth cycles per core             │  │
│  │  Mode: PoUW (STARK Merkle) or PoW (nonce), per-cycle MUX       │  │
│  │                                                                  │  │
│  │  ┌────────────────────────────────────────────────────────┐     │  │
│  │  │  Per-Core Architecture:                                 │     │  │
│  │  │                                                         │     │  │
│  │  │  ┌─────────┐     ┌──────────────────┐   ┌──────────┐  │     │  │
│  │  │  │Input MUX│────→│ Poseidon2 Pipeline│──→│Output RTR│  │     │  │
│  │  │  └──┬──┬───┘     │ [R1][R2]...[R22] │   └──┬───┬───┘  │     │  │
│  │  │     │  │          └──────────────────┘      │   │       │     │  │
│  │  │     │  │                                    │   │       │     │  │
│  │  │  SRAM  nonce                           SRAM  target    │     │  │
│  │  │  data  counter                         write comparator│     │  │
│  │  │  (STARK) (PoW)                        (STARK) (PoW)    │     │  │
│  │  └────────────────────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─ NTT Butterfly Unit (25% die) ──────────────────────────────────┐  │
│  │  M31 multiply-accumulate array                                   │  │
│  │  STARK: polynomial LDE + FRI folding                             │  │
│  │  PoW mode: idle (future: repurpose for other useful computation) │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─ SRAM (20% die) ────────────────────────────────────────────────┐  │
│  │  32 MB on-chip, ~200 GB/s bandwidth                              │  │
│  │  STARK: eval values, Merkle tree nodes, FRI intermediate data    │  │
│  │  PoW mode: unused (Poseidon2 runs from registers only)           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─ Control & I/O (5% die) ────────────────────────────────────────┐  │
│  │  PoW controller: header_state registers, nonce counter, target   │  │
│  │  STARK controller: state machine (trace→NTT→Merkle→FRI)         │  │
│  │  Scheduler: per-cycle STARK/PoW arbitration                      │  │
│  │  Network interface                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## A.2 Poseidon2 Core Circuit (M31)

### A.2.1 M31 Field Arithmetic

**Modular multiplication:**

```
a, b ∈ F_p  where p = 2^31 - 1

c = a × b                     // 31 × 31 → 62-bit product
c_hi = c[61:31]               // upper 31 bits
c_lo = c[30:0]                // lower 31 bits
result = c_hi + c_lo          // Mersenne reduction
if result ≥ p: result -= p    // final correction
```

Mersenne property: `2^31 ≡ 1 (mod p)`, so the upper bits fold directly into the lower bits with a single addition.

| Metric | M31 | Goldilocks (2^64 - 2^32 + 1) |
|--------|-----|------|
| Multiplier width | 31 × 31 → 62 bit | 64 × 64 → 128 bit |
| Reduction | 1 addition | shift + sub + add |
| Gate count | ~1,000 | ~3,500 |
| Latency | 1 cycle | 2–3 cycles |
| Area ratio | **1×** | 3.5× |

**M31 advantage:** 3.5× more multipliers per die → 3.5× more Poseidon2 cores → proportionally higher hashrate.

### A.2.2 S-box: x^5 mod p

```
x ──→ [MUL] ──→ x²
       [MUL] ──→ x⁴ = (x²)²     // parallel with x³ if needed
       [MUL] ──→ x⁵ = x⁴ · x

Critical path: 3 sequential multiplications
Multipliers per S-box: 3 (with x² reuse)
Gate count (M31): 3 × 1,000 = ~3,000 gates
```

### A.2.3 MDS Matrix (Poseidon2)

**External rounds** use a block-circulant structure based on M4 = circ(2, 3, 1, 1):

```
For each 4-element group:
  s₀' = 2s₀ + 3s₁ + s₂ + s₃
  s₁' = s₀ + 2s₁ + 3s₂ + s₃
  s₂' = s₀ + s₁ + 2s₂ + 3s₃
  s₃' = 3s₀ + s₁ + s₂ + 2s₃

  2x = x + x,  3x = x + x + x  → additions only, no multipliers
```

Width 24 = 6 groups of 4 → 6 M4 applications → **additions only** in external MDS.

**Internal rounds** use sparse MDS: M\_I = diag(μ₁, ..., μ\_t) + **1**·**1**^T:

```
sum = s₀ + s₁ + ... + s_{t-1}        // t-1 additions
s_i' = μ_i · s_i + sum                // t multiplications + t additions

Width 16 (standard): 16 multiplications per internal round
Width 24 (extended): 24 multiplications per internal round  (+50%)
```

### A.2.4 Full Round vs Partial Round

| Component | External round (×8) | Internal round (×14) |
|-----------|--------------------|--------------------|
| S-box | t S-boxes (24 × 3K = 72K gates) | 1 S-box (3K gates) |
| MDS | M4 blocks (additions, ~12K gates) | diag + 1·1^T (24 × 1K = 24K gates) |
| Round constants | t additions (24 × ~100 = 2.4K) | t additions (2.4K) |
| **Subtotal** | **~86K gates** | **~29K gates** |

**Total core (pipelined):**

```
8 × 86K + 14 × 29K = 688K + 406K = ~1,094K gates per core (Width 24)

Standard Width 16:
8 × 58K + 14 × 21K = 464K + 294K = ~758K gates per core

Overhead: +44% core logic (Width 16 → 24)
```

---

## A.3 Pipeline Design Options

### A.3.1 Folded (Area Minimal)

```
1 round of hardware × 22 iterations

Area: ~86K gates (one external round circuit, shared)
Throughput: 1 hash / 22 cycles per core
@ 1 GHz: ~45M perm/sec per core → 90M effective hash/sec (2 tickets)
```

### A.3.2 Full Pipeline (Throughput Maximal)

```
22 stages, all rounds instantiated

Area: ~1.2M gates per core (1,094K logic + 78K registers + 3K control)
Throughput: 1 perm / cycle per core (after pipeline fill)
Latency: 22 cycles
@ 1 GHz: 1G perm/sec → 2G effective hash/sec (2 tickets)
```

### A.3.3 Partial Pipeline (Balanced)

```
k stages × (22/k) iterations

k=4: ~300K gates, 1 perm / 5.5 cycles → ~180M perm/sec → 360M eff
k=8: ~600K gates, 1 perm / 2.75 cycles → ~360M perm/sec → 720M eff
k=11: ~660K gates, 1 perm / 2 cycles → ~500M perm/sec → 1G eff
```

### A.3.4 Die-Level Comparison

Assuming 60M gate die, 50% allocated to Poseidon2 cores (30M gates). Each permutation produces **2 PoW tickets**:

| Pipeline style | Gates/core | Cores | Perm/sec/core | **Effective hash/sec** |
|---------------|-----------|-------|--------------|----------------------|
| Folded (×1) | 86K | 348 | 45M | **31.3G** |
| 4-stage | 300K | 100 | 180M | **36G** |
| 8-stage | 600K | 50 | 360M | **36G** |
| Full (×22) | 1.2M | 25 | 1G | **50G** |

Full pipeline achieves highest throughput. The 2-ticket multiplier makes full pipeline particularly effective.

---

## A.4 Header Pre-absorption

Block header H is variable-length (parents, tx\_root, timestamp). Hashing it every nonce attempt is wasteful.

**Optimization: pre-absorb header into Poseidon2 sponge state.**

```
On header change (~100 times/sec at 100 BPS):
  h_H = PoseidonSponge(parent_hashes || tx_merkle_root || timestamp)
  → 8 M31 elements, stored in header_state register

Per nonce attempt (compression function mode, width 24):
  State initialization:
    S[0..7]   ← v₁  (nonce part 1)
    S[8..15]  ← v₂  (nonce part 2)
    S[16..23] ← h_H (from register, constant)
  → 1 Poseidon2 permutation (22 rounds)
  → output S[8..15] and S[16..23] compared against target (2 tickets)
```

**Cost per nonce: exactly 1 Poseidon2 permutation.** Header pre-hash is amortized across ~10M nonce attempts per header change.

Hardware for pre-absorption:
- header\_state register: 8 × 31 bits = 248 bits (~400 gates)
- Negligible compared to Poseidon2 core (~1.2M gates)

---

## A.5 SRAM Bandwidth and Throughput Allocation

### A.5.1 STARK Memory Access Pattern

Each Poseidon2 Merkle hash requires:

```
Read:  left_child  = 8 × 4 bytes = 32 bytes
Read:  right_child = 8 × 4 bytes = 32 bytes
Write: parent_node = 8 × 4 bytes = 32 bytes
Total: 96 bytes per hash
```

### A.5.2 Throughput Calculation

```
SRAM bandwidth:          ~200 GB/s
Bytes per STARK hash:    96 bytes
STARK hash throughput:   200G / 96 ≈ 2.08G hashes/sec

Total Poseidon2 throughput: ~25G perm/sec (25 cores @ 1GHz, Width 24)
Effective PoW hashrate:    ~50G hash/sec (2 tickets per permutation)

STARK allocation:  2.08G / 25G ≈ 8.3%
PoW allocation:    remaining ≈ 91.7%
```

### A.5.3 Interpretation

| Metric | Value | Note |
|--------|-------|------|
| Hardware STARK fraction (f) | ~8% | SRAM-bandwidth limited |
| Hardware PoW fraction | ~92% | Fills idle Poseidon2 cycles |
| U (usefulness) | **100%** | ZK proof computation = useful work (see §2.2 of main document) |
| STARK proofs/sec | ~260 | 2.08G / 8M hashes per proof |
| PoW hashrate | ~50G effective | 25G perm/sec × 2 tickets |

**f is not waste — it is a throughput allocation metric.** It describes how Poseidon2 cycles are allocated between STARK (memory-bandwidth-limited) and PoW (compute-limited). U = 100% because the ASIC is executing ZK proof computation — useful work by definition. The PoW fill cycles provide additional network security as a costless byproduct. The two workloads are complementary: PoW is compute-bound, STARK is memory-bound. They share Poseidon2 cores but bottleneck on different resources, achieving near-perfect utilization. Higher SRAM bandwidth increases ZK proof *economic throughput* (more proofs/sec) but does not change U.

### A.5.4 Increasing STARK Throughput

| Memory technology | Bandwidth | f (STARK fraction) | STARK proofs/sec |
|------------------|-----------|-------------------|-----------------|
| SRAM 32 MB | 200 GB/s | ~8% | ~260 |
| SRAM 64 MB | 400 GB/s | ~17% | ~520 |
| HBM3 8 GB | 1.2 TB/s | ~50% | ~1,560 |
| HBM3E 16 GB | 2.4 TB/s | ~100% | ~3,120 |

With HBM, the STARK fraction approaches 100%, and nearly all Poseidon2 cycles serve STARK computation simultaneously with PoW. Note: this increases ZK proof *economic throughput* but does not change U (which is already 100% when Stwo is active) nor does it satisfy Ball et al.'s strict PoUW definition (which requires protocol-level verification of usefulness — absent under Option C).

---

## A.6 Die Area: kHeavyHash vs Poseidon2-PoW

### A.6.1 kHeavyHash Core

```
cSHAKE256 hash (×2):        ~80K gates (2× Keccak-f[1600], ~40K each, mid-pipeline)
64×64 nibble matrix mul:    ~65K gates (integer matmul + XOR)
Control:                     ~5K gates
─────────────────────────────────────
Total:                      ~150K gates per core
Throughput:                 ~1G hashes/sec per core @ 1GHz

Note: Keccak-f[1600] gate counts range from ~12K (compact/folded) to ~120K
(fully pipelined). The ~40K estimate assumes a mid-pipeline design (4–8
round stages) balancing area and throughput for ASIC mining.
```

### A.6.2 Poseidon2-PoW Core (M31, Width 24, Full Pipeline)

```
S-box circuits:             ~618K gates
  External: 24 × 3K × 8 rounds = 576K
  Internal: 1 × 3K × 14 rounds = 42K
MDS circuits:               ~432K gates
  External: 6 M4 blocks × ~2K × 8 rounds = 96K (additions only)
  Internal: 24 × 1K × 14 rounds = 336K
Round constant storage:      ~53K gates
Pipeline registers:          ~78K gates (24 × 31 bits × 21 stage boundaries)
Input MUX + output router:   ~1K gates
PoW controller:              ~2K gates (header reg, nonce counter, dual target comparator)
─────────────────────────────────────
Total:                       ~1.2M gates per core
Throughput:                  ~1G perm/sec → 2G effective hash/sec (2 tickets) @ 1GHz
```

### A.6.3 Chip-Level Comparison

| Metric | kHeavyHash ASIC | Poseidon2-PoW ASIC (Width 24) |
|--------|----------------|-------------------------------|
| Core area | ~150K gates | ~1.2M gates |
| Cores (60M gate die) | ~380 (95% utilized) | ~25 (50% allocated) |
| Throughput per core | ~1G/s | ~1G perm/s → 2G eff/s (2 tickets) |
| Total chip hashrate | ~380G/s | ~50G/s effective |
| ZK proof capability | None | ~260 proofs/sec |
| Additional components | None | NTT, SRAM, controller |

Poseidon2 has ~7.6× lower PoW hashrate per die than kHeavyHash. **This is absorbed by difficulty adjustment** — all miners use the same hash function, so per-miner revenue is determined by hashrate share, not absolute hashrate. The ZK proof capability provides additional revenue unavailable to kHeavyHash miners.

---

## A.7 M31 vs Goldilocks ASIC Comparison

| Metric | Goldilocks (2^64 - 2^32 + 1) | M31 (2^31 - 1) |
|--------|-----|-----|
| Element size | 64 bits | 31 bits |
| Multiplier gates | ~3,500 | ~1,000 |
| Multiplier latency | 2–3 cycles | 1 cycle |
| Poseidon2 width (extended) | 13 (rate 9, cap 4) | 24 (compression function) |
| Hash output | 4 × 64 = 256 bits | 8 × 31 = 248 bits |
| Cores per die (30M gates) | ~23 | ~25 |
| STARK ecosystem | Plonky2/Plonky3 | **Stwo (potential Kaspa choice)** |

**M31 is the natural choice** if Kaspa adopts Stwo. The smaller multiplier (1/3.5 area) enables higher core density and hashrate per die, while matching Stwo's field arithmetic exactly.
