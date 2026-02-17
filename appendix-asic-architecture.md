# Appendix A: ASIC Architecture Details

**Companion to: ZK-SPoW v0.3**

---

## A.1 ZK-Symbiotic ASIC Block Diagram

```
┌─ ZK-SPoW ASIC (7nm, ~200W) ──────────────────────────────────────────┐
│                                                                        │
│  ┌─ Poseidon2 Core Array (50% die) ────────────────────────────────┐  │
│  │                                                                  │  │
│  │  [Core 0] [Core 1] [Core 2] ... [Core N-1]                      │  │
│  │                                                                  │  │
│  │  Each core: Width-24 Poseidon2 pipeline (M31)                   │  │
│  │  Throughput: 1 hash / pipeline_depth cycles per core             │  │
│  │  Mode: Symbiotic (STARK Merkle) or PoW (nonce), per-cycle MUX  │  │
│  │                                                                  │  │
│  │  ┌────────────────────────────────────────────────────────┐     │  │
│  │  │  Per-Core Architecture:                                 │     │  │
│  │  │                                                         │     │  │
│  │  │  ┌─────────┐     ┌──────────────────┐   ┌──────────┐  │     │  │
│  │  │  │Input MUX│────→│ Poseidon2 Pipeline│──→│Output RTR│  │     │  │
│  │  │  └──┬──┬───┘     │ [R1][R2]...[R30] │   └──┬───┬───┘  │     │  │
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

**External rounds** use a block-circulant structure based on the Stwo/HorizenLabs M4:

```
M4 = [[5, 7, 1, 3],
      [4, 6, 1, 1],
      [1, 3, 5, 7],
      [1, 1, 4, 6]]

For each 4-element group:
  s₀' = 5s₀ + 7s₁ + s₂ + 3s₃
  s₁' = 4s₀ + 6s₁ + s₂ + s₃
  s₂' = s₀ + 3s₁ + 5s₂ + 7s₃
  s₃' = s₀ + s₁ + 4s₂ + 6s₃

  Coefficients ∈ {1,3,4,5,6,7} — all shift+add, no multipliers
```

Width 24 = 6 groups of 4 → 6 M4 applications → **shift+add only** in external MDS.

**Internal rounds** use sparse MDS: M\_I = **1**·**1**^T + diag(V), where V = [−2, 1, 2, 4, ..., 2²²] (Plonky3 production values [10]):

```
sum = s₀ + s₁ + ... + s_{t-1}        // t-1 additions
s_i' = V[i] · s_i + sum               // V[i] ∈ {powers of 2, −2} → shift+add

Width 16 (standard): 16 multiplications per internal round
Width 24 (extended): 24 multiplications per internal round  (+50%)
```

### A.2.4 Full Round vs Partial Round

| Component | External round (×8) | Internal round (×22) |
|-----------|--------------------|--------------------|
| S-box | t S-boxes (24 × 3K = 72K gates) | 1 S-box (3K gates) |
| MDS | M4 blocks (shift+add, ~12K gates) | diag + 1·1^T (24 × 1K = 24K gates) |
| Round constants | t additions (24 × ~100 = 2.4K) | t additions (2.4K) |
| **Subtotal** | **~86K gates** | **~29K gates** |

**Total core (pipelined):**

```
8 × 86K + 22 × 29K = 688K + 638K = ~1,326K gates per core (Width 24)

Standard Width 16:
8 × 58K + 14 × 21K = 464K + 294K = ~758K gates per core

Overhead: +75% core logic (Width 16 → 24)
```

---

## A.3 Pipeline Design Options

### A.3.1 Folded (Area Minimal)

```
1 round of hardware × 30 iterations

Area: ~86K gates (one external round circuit, shared)
Throughput: 1 hash / 30 cycles per core
@ 1 GHz: ~33M perm/sec per core → 100M effective hash/sec (3 tickets)
```

### A.3.2 Full Pipeline (Throughput Maximal)

```
30 stages, all rounds instantiated

Area: ~1.4M gates per core (1,326K logic + 108K registers + 3K control)
Throughput: 1 perm / cycle per core (after pipeline fill)
Latency: 30 cycles
@ 1 GHz: 1G perm/sec → 3G effective hash/sec (3 tickets)
```

### A.3.3 Partial Pipeline (Balanced)

```
k stages × (30/k) iterations

k=5: ~300K gates, 1 perm / 6 cycles → ~167M perm/sec → 500M eff
k=10: ~550K gates, 1 perm / 3 cycles → ~333M perm/sec → 1G eff
k=15: ~700K gates, 1 perm / 2 cycles → ~500M perm/sec → 1.5G eff
```

### A.3.4 Die-Level Comparison

Assuming 60M gate die, 50% allocated to Poseidon2 cores (30M gates). Each permutation produces **3 PoW tickets**:

| Pipeline style | Gates/core | Cores | Perm/sec/core | **Effective hash/sec** |
|---------------|-----------|-------|--------------|----------------------|
| Folded (×1) | 86K | 348 | 33M | **35G** |
| 5-stage | 300K | 100 | 167M | **50G** |
| 10-stage | 550K | 54 | 333M | **54G** |
| Full (×30) | 1.4M | 21 | 1G | **63G** |

Full pipeline achieves highest throughput. The 3-ticket multiplier makes full pipeline particularly effective.

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
  → 1 Poseidon2 permutation (30 rounds)
  → output S[0..7], S[8..15], and S[16..23] compared against target (3 tickets)
```

**Cost per nonce: exactly 1 Poseidon2 permutation.** Header pre-hash is amortized across ~10M nonce attempts per header change.

Hardware for pre-absorption:
- header\_state register: 8 × 31 bits = 248 bits (~400 gates)
- Negligible compared to Poseidon2 core (~1.4M gates)

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

Total Poseidon2 throughput: ~21G perm/sec (21 cores @ 1GHz, Width 24)
Effective PoW hashrate:    ~63G hash/sec (3 tickets per permutation)

STARK allocation:  2.08G / 21G ≈ 9.9%
PoW allocation:    remaining ≈ 90.1%
```

### A.5.3 Interpretation

| Metric | Value | Note |
|--------|-------|------|
| Hardware STARK fraction (f) | ~10% | SRAM-bandwidth limited |
| Hardware PoW fraction | ~90% | Fills idle Poseidon2 cycles |
| U (usefulness) | **≈67%** | t₀/t = 16/24; width extension overhead = 33% (see §2.2) |
| STARK proofs/sec | ~260 | 2.08G / 8M hashes per proof |
| PoW hashrate | ~63G effective | 21G perm/sec × 3 tickets |
| U\_avg | **≈6.7%** | f × U = 0.10 × 0.67; time-averaged usefulness (see §2.2) |

**Time-averaged usefulness.** U\_avg = f\_sym × U ≈ 0.10 × 0.67 ≈ **6.7%** under 200 GB/s SRAM bandwidth with the STARK prover continuously active. This means 6.7% of all Poseidon2 cycles advance ZK proofs; the remaining 93.3% provide PoW security only. In a conventional PoW network, U\_avg = 0%: all mining energy produces security and nothing else. ZK-SPoW's 6.7% represents computation that would otherwise have no useful output beyond block validation. U\_avg scales with memory bandwidth: ~13% at 400 GB/s, ~40% at 1.2 TB/s (§A.5.4). When the STARK prover is inactive (no ZK demand), U\_avg = 0% and the ASIC operates as a conventional PoW miner.

**f is not waste — it is a throughput allocation metric.** It describes how Poseidon2 cycles are allocated between STARK (memory-bandwidth-limited) and PoW (compute-limited). U ≈ 67% because 8 of 24 input state elements carry header\_digest rather than ZK data. The PoW cycles provide additional network security as a low-marginal-cost byproduct. The two workloads are complementary: PoW is compute-bound, STARK is memory-bound. They share Poseidon2 cores but bottleneck on different resources, achieving near-perfect utilization.

**Width-24 efficiency.** Because STARK Merkle throughput is SRAM-bandwidth-limited at ~2.08G hash/sec, Width-24 compression (1 perm/hash) delivers the same ZK proof rate as Width-16 sponge (2 perm/hash) while consuming half the Poseidon2 cycles (~10% vs ~20% of core capacity). The freed cycles serve PoW. Higher SRAM bandwidth increases ZK proof *economic throughput* (more proofs/sec) but does not change U.

### A.5.4 Increasing STARK Throughput

| Memory technology | Bandwidth | f (STARK fraction) | U\_avg (f × 67%) | STARK proofs/sec |
|------------------|-----------|-------------------|-----------------|-----------------|
| SRAM 32 MB | 200 GB/s | ~10% | ~6.7% | ~260 |
| SRAM 64 MB | 400 GB/s | ~20% | ~13% | ~520 |
| HBM3 8 GB | 1.2 TB/s | ~60% | ~40% | ~1,560 |
| HBM3E 16 GB | 2.4 TB/s | ~100% | ~67% | ~3,120 |

With HBM, the STARK fraction approaches 100%, and nearly all Poseidon2 cycles serve STARK computation simultaneously with PoW. Note: this increases ZK proof *economic throughput* but does not change U (which is bounded by t₀/t = 16/24 ≈ 67%, the width extension overhead). Higher memory bandwidth cannot overcome the fundamental width ratio cost.

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
S-box circuits:             ~642K gates
  External: 24 × 3K × 8 rounds = 576K
  Internal: 1 × 3K × 22 rounds = 66K
MDS circuits:               ~624K gates
  External: 6 M4 blocks × ~2K × 8 rounds = 96K (shift+add only)
  Internal: 24 × 1K × 22 rounds = 528K
Round constant storage:      ~55K gates (214 constants × 31 bits)
Pipeline registers:          ~108K gates (24 × 31 bits × 29 stage boundaries)
Input MUX + output router:   ~1K gates
PoW controller:              ~2K gates (header reg, nonce counter, triple target comparator)
─────────────────────────────────────
Total:                       ~1.4M gates per core
Throughput:                  ~1G perm/sec → 3G effective hash/sec (3 tickets) @ 1GHz
```

### A.6.3 Chip-Level Comparison

| Metric | kHeavyHash ASIC | Poseidon2-PoW ASIC (Width 24) |
|--------|----------------|-------------------------------|
| Core area | ~150K gates | ~1.4M gates |
| Cores (60M gate die) | ~380 (95% utilized) | ~21 (50% allocated) |
| Throughput per core | ~1G/s | ~1G perm/s → 3G eff/s (3 tickets) |
| Total chip hashrate | ~380G/s | ~63G/s effective |
| ZK proof capability | None | ~260 proofs/sec |
| Additional components | None | NTT, SRAM, controller |

Poseidon2 has ~6× lower PoW hashrate per die than kHeavyHash. **This is absorbed by difficulty adjustment** — all miners use the same hash function, so per-miner revenue is determined by hashrate share, not absolute hashrate. The ZK proof capability provides additional revenue unavailable to kHeavyHash miners.

---

## A.7 M31 vs Goldilocks ASIC Comparison

| Metric | Goldilocks (2^64 - 2^32 + 1) | M31 (2^31 - 1) |
|--------|-----|-----|
| Element size | 64 bits | 31 bits |
| Multiplier gates | ~3,500 | ~1,000 |
| Multiplier latency | 2–3 cycles | 1 cycle |
| Poseidon2 width (extended) | 13 (rate 9, cap 4) | 24 (compression function) |
| Hash output | 4 × 64 = 256 bits | 8 × 31 = 248 bits |
| Cores per die (30M gates) | ~23 | ~21 |
| STARK ecosystem | Plonky2/Plonky3 | **Stwo (potential Kaspa choice)** |

**M31 is the natural choice** if Kaspa adopts Stwo. The smaller multiplier (1/3.5 area) enables higher core density and hashrate per die, while matching Stwo's field arithmetic exactly.

---

