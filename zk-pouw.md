# ZK-PoUW: ZK-Symbiotic Proof of Work for Kaspa

**Version 0.1 — Draft — 2026-02-14**

---

## 1. Abstract

We propose replacing Kaspa's kHeavyHash proof-of-work function with Poseidon2 over the Mersenne field M31, the same algebraic hash function used internally by the Stwo STARK prover. By extending the Poseidon2 permutation width from 16 to 24 elements and operating in compression function mode, every Poseidon2 invocation accepts a Merkle child pair plus block\_header as input and simultaneously produces two independent PoW tickets. This unification means that a mining ASIC and a ZK prover share the same Poseidon2 core design — a property we call **ZK-Symbiotic Proof of Work**. The width extension carries a usefulness cost of U = t₀/t = 16/24 ≈ 67% — the remaining 33% of each core's computation serves PoW integration rather than ZK proving. In return, the same hardware performs both network security (PoW) and useful computation (ZK proofs) with cycle-level switching and no hashrate degradation.

---

## 2. Introduction

### 2.1 The PoW Energy Problem

Kaspa uses kHeavyHash — cSHAKE256 (Keccak/SHA-3 family) composed with a 64×64 matrix multiplication over 4-bit nibbles — for proof of work. Like all traditional PoW schemes, the computational work produces no output beyond network security. The entire energy expenditure is justified solely by the security guarantees it provides.

### 2.2 Proof of Useful Work

Ball et al. [1] formalize **Proof of Useful Work (PoUW)** as a PoW scheme where the mining computation simultaneously produces useful output verifiable by the network. Their strict definition requires:

1. The PoW computation itself produces useful output
2. The verifier can confirm the usefulness
3. The useful output is bound to the PoW evidence

We adopt a **practical definition** that accounts for the hardware cost of PoW integration:

> **Definition (Usefulness).**
> - U = t₀/t = 16/24 ≈ **67%** when the ASIC is executing ZK proof computation (PoUW mode)
> - U = **0%** when the ASIC is grinding nonces without concurrent ZK computation (Pure PoW mode)
>
> The upper bound reflects the width extension cost: each Width-24 permutation carries 8 state elements (33% of the core) dedicated to PoW integration (header\_hash input, dual ticket output) that do not advance ZK proof computation. A hypothetical Width-16 Stwo prover would achieve U = 100%, but cannot produce PoW tickets. This cost is a fundamental property of the width extension (t₀/t), independent of die-level design choices (SRAM, NTT allocation, etc.).

The key insight: ZK proof computation *is* work. When an ASIC runs the Stwo prover, every Poseidon2 cycle produces economically valuable ZK proofs, with PoW tickets as a low-marginal-cost byproduct. The per-permutation cost of PoW tickets is zero, but the width extension from 16 to 24 imposes a fixed 33% core overhead (see §4.2). If a PoW solution is found mid-proof, the block is submitted without interrupting ZK computation. When no ZK demand exists, the ASIC reverts to pure PoW (U = 0%), identical to any conventional miner.

### 2.3 Stwo and Poseidon2

Kaspa is evaluating StarkWare's **Stwo** [4] as a potential STARK backend for verifiable programs (vProgs). Stwo operates over the Mersenne field M31 and uses **Poseidon2** [3] as its internal hash function for Merkle tree commitments and Fiat-Shamir challenges.

This creates a unique opportunity: if the PoW hash function is also Poseidon2 over M31, then the mining ASIC's primary computational element — the Poseidon2 pipeline — is *identical* to what is needed for STARK proof generation. Mining hardware becomes ZK proving hardware, with the cost being a Poseidon2 width extension from 16 to 24 elements (+44% core area, ~+22% die area; see §4.2). ZK proof throughput is unaffected by this extension because STARK Merkle hashing is SRAM-bandwidth-bound, not Poseidon2-compute-bound (see §5.6).

---

## 3. Notation and Definitions

| Symbol | Definition |
|--------|-----------|
| F\_p | Finite field, p = 2^31 - 1 (Mersenne prime M31) |
| Poseidon2\_π | Poseidon2 permutation over F\_p^t |
| t | State width (number of field elements in permutation) |
| r | Rate: number of input/output elements (sponge mode) |
| c | Capacity: security parameter (sponge mode; hidden elements) |
| n | Hash output size in field elements (n = 8, giving 248 bits) |
| H | Block header: (parent\_hashes, tx\_merkle\_root, timestamp) |
| h\_H | Header hash: PoseidonSponge(H) ∈ F\_p^k |
| k | Header hash element count (k = 8 for symmetric I/O and dual PoW tickets) |
| (v\_1, v\_2) | Nonce: v\_1, v\_2 ∈ F\_p^8 |
| T | Target ∈ F\_p^8 (difficulty-adjusted) |
| S | Poseidon2 state after permutation, S ∈ F\_p^t |

**Stwo baseline parameters (confirmed from source code [4]):**

| Parameter | Value |
|-----------|-------|
| Field | M31, p = 2^31 - 1 |
| Hash output | 8 elements = 248 bits |
| Standard width | t₀ = 16 (sponge mode: rate 8, capacity 8) |
| External rounds R\_f | 8 (4 + 4) |
| Internal rounds R\_p | 14 |
| S-box exponent | α = 5 |
| Merkle hash | 2 permutations per node (sponge: absorb left[8], absorb right[8]) |
| Commitment hash | Blake2s (base layer), Poseidon2 (recursive proofs) |

---

## 4. Protocol Specification

### 4.1 PoW Function Replacement

**Current (kHeavyHash):**

```
pow_hash = cSHAKE256(M · cSHAKE256(H || nonce) ⊕ cSHAKE256(H || nonce))
valid iff pow_hash < target
```

where M is a 64×64 matrix (generated from header, full-rank over nibbles), nonce is 8 bytes.

**Proposed (Poseidon2-PoW):**

```
h_H = PoseidonSponge(H)                          // header pre-hash (8 M31 elements)
S   = Poseidon2_π(v₁ || v₂ || h_H)              // single permutation, width 24
pow_hash₁ = (S[8], S[9], ..., S[15])              // 8 M31 elements = 248 bits
pow_hash₂ = (S[16], S[17], ..., S[23])            // 8 M31 elements = 248 bits
valid iff pow_hash₁ < target OR pow_hash₂ < target
```

where (v₁, v₂) are 8 M31 elements each (64 bytes total nonce). The permutation operates in **compression function mode** — all 24 input elements are visible (no hidden capacity). This differs from Stwo's standard sponge mode (width 16, rate 8, capacity 8) but is a recommended Poseidon2 usage mode [3]. Each permutation produces **two independent PoW tickets**, doubling the effective hashrate per permutation.

**Verification cost:** One Poseidon2 permutation (width 24) + two target comparisons + one header pre-hash (amortized).

### 4.2 Poseidon2 Width Extension

The core design change is extending the Poseidon2 permutation width to accommodate block\_header as an additional input, and switching from sponge mode to compression function mode.

#### 4.2.1 Stwo Baseline

Stwo's Poseidon2 operates in **sponge mode** with width 16:

```
Width t₀ = 16
Rate  r  = 8    ← absorbs 8 elements per permutation
Capacity c = 8  ← hidden elements (security)
```

For Merkle tree commitments, each node hash requires **two sponge absorptions**:

```
Absorb left_child[8]  → Permute → (state updated)
Absorb right_child[8] → Permute → Squeeze output[8]

Total: 2 permutations per Merkle hash
```

#### 4.2.2 Design Alternatives

Three approaches to integrate header\_hash into the Merkle hash:

| Design | Width | Mode | Perm/hash | PoW tickets | Core area Δ | Die area Δ |
|--------|-------|------|-----------|-------------|-------------|------------|
| A: 3rd absorb | 16 | Sponge | 3 (+50%) | 1 | 0% | 0% |
| B: Width 20 | 20 | Compression | 1 | 1 (+4 wasted) | +22% | ~+11% |
| **C: Width 24** | **24** | **Compression** | **1** | **2** | **+44%** | **~+22%** |

**Design A:** No Poseidon2 modification. Standard width-16 sponge with 3 absorptions (left, right, header[4]). +50% permutations per hash, reducing both PoW and ZK throughput by 33%.

**Design B:** Extend to width 20 with header\_hash[4]. Compression function mode, 1 permutation per hash. But 4 output elements are unused (asymmetric 8+8+4 I/O).

**Design C (selected):** Extend to width 24 with header\_hash[8]. Symmetric 8+8+8 I/O — all output elements are useful. **2 PoW tickets per permutation** (S[8..15] and S[16..23]), yielding +67% effective hashrate over Design B despite 17% fewer cores. Header hash security doubles (248 vs 124 bits). Width 24 is within the Poseidon2 paper's analyzed parameter range [3]. **ZK throughput is unaffected**: STARK Merkle hashing is SRAM-bandwidth-bound (~2.08G hash/sec regardless of width), so the larger cores do not reduce ZK proof generation rate (see §5.6). The width extension cost manifests purely as U = 16/24 ≈ 67% (§2.2).

#### 4.2.3 Proposed Extension (Design C)

```
Standard Stwo:    Width t₀ = 16,  Sponge (rate 8, capacity 8)
Proposed ZK-PoUW: Width t  = 24,  Compression function (all 24 visible)
```

**Impact on Poseidon2 internals:**

| Component | Width 16 (standard) | Width 24 (proposed) | Change |
|-----------|--------------------|--------------------|--------|
| External MDS | 4 M4 blocks (additions) | 6 M4 blocks (additions) | +50% additions |
| Internal MDS | 16 multiplications (diag + 1·1^T) | 24 multiplications | +50% multiplications |
| S-box (external) | 16 per round | 24 per round | +50% |
| S-box (internal) | 1 per round | 1 per round | unchanged |
| State registers | 16 × 31 = 496 bits | 24 × 31 = 744 bits | +50% |

Internal round MDS scales as O(t), not O(t²), because the sparse MDS structure (diagonal + rank-1) requires only t multiplications per round. This makes the width extension significantly cheaper than for standard Poseidon.

**Core area overhead: +44%.** Total die area impact: **~+22%** (Poseidon2 is 50% of die; see Appendix A). The 2-ticket property yields **+67% effective hashrate**, partially offsetting the core count reduction. The usefulness cost is U = t₀/t = 16/24 ≈ 67% — the 33% overhead per permutation serves PoW integration, not ZK computation.

#### 4.2.4 Compression Function vs Sponge

Standard Stwo uses sponge mode with 8 hidden capacity elements. This proposal uses **compression function mode** where all 24 state elements are visible. Key differences:

| Property | Sponge (Stwo) | Compression (ZK-PoUW) |
|----------|--------------|----------------------|
| Hidden state | 8 capacity elements | None (all visible) |
| Security model | Indifferentiability | Collision/preimage resistance of π |
| Permutations per Merkle hash | 2 | **1** |
| Width | 16 | 24 |
| PoW tickets per hash | 0 | **2** |

Both modes are established Poseidon2 usage modes [3]. The Poseidon2 paper recommends compression function mode for Merkle trees, noting up to 5× efficiency over sponge mode **in compute-bound settings**. On the ZK-PoUW ASIC, STARK Merkle throughput is SRAM-bandwidth-bound (~2.08G hash/sec; see §A.5), so this per-permutation efficiency gain does not increase ZK proof rate — it instead frees Poseidon2 cycles for PoW fill. Width 24 is within the paper's analyzed parameter range, though compression function mode over M31 requires dedicated security review (see §7.3, §9).

### 4.3 I/O Mapping

For a single Poseidon2 permutation with state S ∈ F\_p^24 (compression function mode):

```
INPUT (24 = 8+8+8, all visible):
  S[0..7]    ← left_child     8 M31 elements (248 bits)
  S[8..15]   ← right_child    8 M31 elements (248 bits)
  S[16..23]  ← header_hash    8 M31 elements (248 bits)

          ┌──────────────────────────────────┐
          │   Poseidon2 permutation (t = 24) │
          │   R_f = 8 external + R_p = 14 int│
          └──────────────────────────────────┘

OUTPUT (24 = 8+8+8, all visible):
  S[0..7]    → merkle_node    STARK Merkle parent node (useful)
  S[8..15]   → pow_hash₁      PoW ticket 1 (248 bits)
  S[16..23]  → pow_hash₂      PoW ticket 2 (248 bits)
```

Symmetric 8+8+8 I/O with zero waste. No capacity elements — compression function mode exposes all state elements. Security relies on the Poseidon2 permutation's collision resistance and PRP properties (see §4.2.4).

**Dual-use property:**

| Mode | S[0..7] in | S[8..15] in | S[16..23] in | S[0..7] out | S[8..15] out | S[16..23] out |
|------|-----------|------------|-------------|-----------|-------------|--------------|
| STARK (PoUW) | left child | right child | header\_hash | merkle parent | PoW ticket₁ | PoW ticket₂ |
| Pure PoW | v₁ | v₂ | header\_hash | (discarded) | PoW ticket₁ | PoW ticket₂ |

The same Poseidon2 hardware computes both modes. Only the input source for S[0..15] differs (SRAM Merkle data vs random nonces). S[16..23] is always header\_hash. **Each permutation produces 2 independent PoW tickets**, doubling effective hashrate.

### 4.4 Block Structure

```
Block {
  header: {
    parent_hashes:   [Hash; D]        // D DAGKNIGHT parents
    tx_merkle_root:  Hash
    timestamp:       u64
  }
  nonce: {
    val1: [F_p; 8]                    // 32 bytes
    val2: [F_p; 8]                    // 32 bytes
  }                                    // Total: 64 bytes
  transactions: [Tx]
}
```

| Field | kHeavyHash (current) | Poseidon2-PoW (proposed) |
|-------|---------------------|------------------------|
| Nonce size | 8 bytes | 64 bytes (+56 bytes) |
| PoW hash size | 256 bits (32 bytes) | 248 bits (31 bytes) |
| Block overhead | — | +56 bytes (~0.04% of 125KB block) |
| Verification | kHeavyHash × 1 | Poseidon2 × 1 |

**STARK proofs are NOT included in the block.** They are submitted as independent transactions in the mempool, providing economic value to the ZK ecosystem (vProgs fees). This eliminates the +3–5 MB/s bandwidth overhead that would result from mandatory per-block STARK proofs at 100 BPS.

### 4.5 Header Hash Collision Resistance

The header hash h\_H compresses the variable-length block header into k field elements. If k is too small, an attacker can find two headers H\_A ≠ H\_B with identical h\_H, allowing PoW solutions to be transplanted between conflicting blocks.

| k (elements) | Bits | Birthday bound | Security |
|-------------|------|---------------|----------|
| 1 | 31 | 2^15 ≈ 32K | INSECURE |
| 2 | 62 | 2^31 ≈ 2 × 10^9 | INSECURE |
| 4 | 124 | 2^62 ≈ 4.6 × 10^18 | SECURE |
| 8 | 248 | 2^124 | Overkill |

**Minimum: k = 4** (124-bit collision resistance). We choose **k = 8** (248-bit, matching PoW hash size) to enable symmetric 8+8+8 I/O and 2 PoW tickets per permutation. Width extension: t = 16 + 8 = 24.

---

## 5. Operating Modes

### 5.1 PoUW Mode (Stwo Prover Active)

When ZK proof demand exists, the ASIC runs the Stwo prover. The STARK proof generation pipeline:

```
1. Trace generation       → circuit evaluation
2. NTT (LDE)              → polynomial domain extension
3. Merkle tree            → Poseidon2 hashing (this is where PoW tickets appear)
4. Fiat-Shamir challenge  → derived from Merkle root
5. FRI rounds             → folding + commitment
```

At step 3, every Merkle hash is:

```
S = Poseidon2_π(n_L || n_R || h_H)       // width 24, compression function
merkle_parent = S[0..7]       ← advances STARK proof (useful)
pow_ticket₁   = S[8..15]      ← checked against PoW target (security)
pow_ticket₂   = S[16..23]     ← checked against PoW target (security)

if pow_ticket₁ < target OR pow_ticket₂ < target → BLOCK FOUND
```

Every Poseidon2 invocation in the STARK computation simultaneously:
- **(a)** Advances the ZK proof (economic value — useful work)
- **(b)** Produces PoW tickets (network security — low-marginal-cost byproduct)

**U = t₀/t = 16/24 ≈ 67%.** The ASIC is performing ZK proof computation with PoW tickets as a byproduct. The 33% usefulness gap is the width extension overhead: 8 of 24 state elements per permutation serve PoW (header\_hash input, dual ticket output) rather than ZK. This is the inherent cost of PoW integration. Note that there is no "ZK only" mode — every Width-24 permutation produces PoW tickets, regardless of whether the input is STARK Merkle data or random nonces.

**STARK Merkle hashing modes:** The ASIC can perform STARK Merkle hashing in two ways:

| Mode | Method | Perm/hash | Stwo verifier | PoW tickets during STARK |
|------|--------|-----------|---------------|-------------------------|
| A | Width-24 compression (h\_H in S[16..23]) | 1 | Requires modification | Yes (from every STARK hash) |
| B | Width-16 sponge emulation (8 elements unused) | 2 | Standard compatible | No (only from PoW fill cycles) |

Both modes produce identical ZK throughput (SRAM-bandwidth-bound at ~2.08G hash/sec). Mode A yields ~10% more PoW fill during STARK operation. This proposal assumes Mode A; Mode B is available as a fallback for Stwo compatibility.

### 5.2 Pure PoW Mode (No ZK Demand)

When no ZK proofs are requested:

```
loop:
  v₁ = next_nonce_1()
  v₂ = next_nonce_2()
  S = Poseidon2_π(v₁ || v₂ || h_H)       // width 24, compression function
  if S[8..15] < target OR S[16..23] < target → BLOCK FOUND
```

Identical Poseidon2 pipeline, identical throughput. The only difference is the input source: random/sequential nonces instead of STARK Merkle children.

**U = 0%.** No ZK proof is being computed. The ASIC provides network security only, equivalent to any conventional PoW miner. This is not waste — security has value — but it is not useful work in the PoUW sense.

### 5.3 Linear Mode Transition

```
┌─ Poseidon2 Pipeline (always 100% utilized) ──────────────┐
│                                                            │
│  Input MUX (per-cycle decision):                           │
│    SRAM data ready  → STARK Merkle hash   (PoUW)         │
│    SRAM not ready   → PoW nonce hash      (PoW)          │
│                                                            │
│  Switching cost: 0 cycles (combinational MUX, ~300 gates) │
│  Hashrate: invariant across all modes                      │
└────────────────────────────────────────────────────────────┘
```

The transition between PoUW and Pure PoW is **per-cycle and linear**, not a discrete mode switch. When the Poseidon2 MDS is executing STARK computation, that cycle is PoUW. When SRAM data is not ready, a random nonce hash is substituted and that cycle is PoW. The pipeline is always full — the ratio of PoUW to PoW cycles is determined by SRAM bandwidth, not by difficulty or protocol parameters.

### 5.4 Hashrate Invariance

> **Proposition.** Total PoW hashrate H is independent of the operating mode.
>
> *Argument.* H = N\_cores × throughput\_per\_core. Each core's throughput is 1 hash per pipeline\_depth cycles (fully pipelined), regardless of whether the input is a STARK Merkle pair or a random nonce. Input MUX adds zero latency (combinational logic). Therefore H is constant across PoUW, Pure PoW, and any mixed state.

### 5.5 Difficulty Independence

U is determined by ZK demand and width ratio, not by PoW difficulty.

| Condition | U | Rationale |
|-----------|---|-----------|
| Stwo Prover active, any difficulty | ≈67% (16/24) | ZK proof computation, minus width extension overhead |
| No ZK demand, any difficulty | 0% | Pure PoW = security only |

Difficulty affects how many hashes are needed to find a block, but does not change U. Whether difficulty is 8M or 188G, the ASIC either computes ZK proofs (U ≈ 67%) or grinds nonces (U = 0%). The ratio of STARK-to-PoW cycles within the pipeline is determined by SRAM bandwidth (a hardware constant), not by the network's difficulty target.

### 5.6 Complementary Bottleneck Structure

The simultaneous execution of PoW and STARK is possible because they bottleneck on different resources:

| Resource | PoW | STARK | Combined |
|----------|-----|-------|----------|
| Poseidon2 cores | **100%** (compute-bound) | ~8% (data-starved) | **~100%** |
| NTT unit | 0% (unused) | **100%** | **100%** |
| SRAM bandwidth | 0% (registers only) | **100%** (memory-bound) | **100%** |

PoW is compute-bound (limited by Poseidon2 throughput). STARK is memory-bound (limited by SRAM bandwidth feeding Merkle data to Poseidon2). They share Poseidon2 cores but contend on different bottleneck resources, achieving near-perfect utilization of all hardware components simultaneously. This complementary structure is the foundation of the economic analysis in §6.

**Width-neutrality.** Because STARK is memory-bound, the Poseidon2 core width (16 or 24) does not affect ZK proof throughput. Width-24 reduces the fraction of Poseidon2 cycles consumed by STARK (from ~17% with sponge to ~8% with compression), freeing more cycles for PoW fill — but the STARK proof generation rate remains identical at ~2.08G Merkle hashes/sec (see §A.5).

---

## 6. Pareto Analysis

### 6.1 Competing Designs

Four designs compared under identical die area and power budget:

| Design | Die allocation | Hashrate | ZK Throughput | U | Can mine? |
|--------|---------------|----------|--------------|---|-----------|
| Pure PoW | 95% Poseidon2, 5% control | ~1.9 H (die area: 95/50) | 0 | 0% | Yes |
| Pure Stwo | 20% Poseidon2, 40% NTT, 35% SRAM | 0 | Z\_max | 100% (t₀/t₀) | No |
| **ZK-PoUW** | **50% Poseidon2, 25% NTT, 20% SRAM** | **H** | **Z** | **≈67% (t₀/t)** | **Yes** |
| ZK-PoUW+HBM | 50% Poseidon2, 25% NTT, 20% HBM I/F | H | Z\_hbm | ≈67% (t₀/t) | Yes |

Pure PoW achieves ~1.9× hashrate on a die-area basis (95%/50% Poseidon2 allocation) but produces no ZK proofs (U = 0%). On a hashes-per-watt basis, the gap narrows to ~1.1–1.2× because idle NTT and SRAM contribute static leakage (~10–20% of dynamic power). Pure Stwo uses standard Width-16 cores (U = 100%) but cannot mine. ZK-PoUW achieves U ≈ 67% and mining capability — the 33% usefulness gap is the cost of PoW integration via width extension.

### 6.2 Economic Dominance

Difficulty adjusts to total network hashrate. When all N miners use the same design, per-ASIC mining revenue is B/N regardless of absolute hashrate. The differentiator is ZK revenue:

```
ZK-PoUW revenue = B/N + Z×F   >   B/N = Pure PoW revenue    (for any ZF > 0)
```

Pure PoW's ~1.1–1.2× power-efficiency advantage yields at most ~10–20% more mining revenue per watt in a mixed network. Once ZK fee income Z×F exceeds this margin, ZK-PoUW strictly dominates. As the network grows (N → ∞), per-miner mining reward → 0 while ZK fees remain constant, making ZK-PoUW the eventual equilibrium. In the interim (ZF ≈ 0), ZK-PoUW ASICs function as standard PoW miners with no penalty.

---

## 7. Security Considerations

### 7.1 Poseidon2 Cryptographic Properties

Poseidon2 [3] maintains 128-bit security with capacity c = 8 M31 elements (248 bits). Known algebraic attack vectors (Gröbner basis, interpolation, differential) have been analyzed for standard parameters. StarkWare has adopted Poseidon2 for production use in Starknet, representing significant implicit endorsement of its security.

### 7.2 Single Primitive Dependency

| Component | Current Kaspa | Proposed |
|-----------|--------------|----------|
| PoW | kHeavyHash (cSHAKE256) | Poseidon2 |
| STARK | Poseidon2 | Poseidon2 |
| Independence | PoW ≠ STARK | PoW = STARK |

**Risk:** A cryptographic break in Poseidon2 compromises both PoW security and STARK validity simultaneously.

**Mitigating factors:**
- StarkWare's Starknet accepts identical risk (entire L2 depends on Poseidon2)
- Active cryptanalysis community monitors algebraic hash functions
- Kaspa's modular architecture permits hash function upgrade via hard fork
- No known viable attack exists against Poseidon2 with recommended parameters

**Emergency response:** If a Poseidon2 weakness is discovered:
- Immediate: Increase difficulty to offset any attack advantage
- Short-term: Hard fork to fallback hash (e.g., Blake3-PoW) using the control logic already present in the ASIC's I/O path
- Medium-term: Deploy replacement algebraic hash with ASIC firmware update (if the vulnerability is specific to Poseidon2 constants rather than the algebraic structure)

**Assessment:** Accepted risk, shared with the broader Poseidon2 ecosystem. The benefit (mining ASIC = ZK prover) outweighs the concentration risk, particularly if Kaspa adopts Stwo as its ZK backend.

### 7.3 Non-Standard Width and Mode

| Width | Mode | Deployment | Security analysis |
|-------|------|-----------|------------------|
| 8 | Sponge | Filecoin, Mina | Extensively analyzed |
| 12 | Sponge | Plonky2 | Well analyzed |
| 16 | Sponge | **Stwo (standard)** | Analyzed by StarkWare |
| 16 | Compression | Various Merkle trees | Analyzed (recommended by [3]) |
| **24** | **Compression** | **This proposal** | **Width analyzed; compression mode needs review** |

**Round number adequacy (preliminary):** The Poseidon2 paper [3] specifies minimum rounds as a function of width t and security level. For t ≤ 24 over a prime field with α = 5, the recommended minimum is R\_f = 8, R\_p ≥ ⌈log\_5(2 · security\_bits)⌉ + partial\_round\_margin. With R\_p = 14 (same as Stwo's width 16), the margin decreases as width grows because the algebraic degree per round increases more slowly relative to the state size. A dedicated analysis must confirm R\_p = 14 remains sufficient for width 24, or determine if R\_p should increase (e.g., to 16–18).

**Required:** Commission dedicated security analysis for Poseidon2 with t = 24 over M31 in **compression function mode**, covering:
- Algebraic attack resistance (Gröbner basis complexity bounds for width 24)
- Differential and linear cryptanalysis (larger state → more diffusion paths)
- Round number adequacy (R\_f = 8, R\_p = 14) — verify sufficient margin for width 24
- MDS matrix security properties for the 24×24 case
- Compression function mode security (all state elements visible, no hidden capacity)
- Independence of dual PoW tickets (S[8..15] and S[16..23] from same permutation)

### 7.4 PoW Hash Distribution

Both pow\_hash₁ = S[8..15] and pow\_hash₂ = S[16..23] are outputs of the same Poseidon2 permutation. In compression function mode, all 24 state elements are visible by design. Security relies on the permutation's PRP properties: given a random-looking input, all output elements should be indistinguishable from random. The two PoW tickets are deterministically linked (same permutation), but each is individually pseudorandom. An attacker who could predict pow₂ from pow₁ without computing the full permutation would violate the PRP assumption — equivalent to breaking Poseidon2. The full-round permutation (8 external + 14 internal) ensures all output elements are cryptographically mixed across all 24 state positions.

**Dual-ticket mining advantage:** With two independent 248-bit tickets per permutation, the probability of finding at least one valid PoW per permutation is:

```
P(valid) = 1 - (1 - T/2^248)² ≈ 2T/2^248    for small T/2^248
```

This yields +100% improvement in expected block-finding rate per permutation (equivalent to +67% per unit die area, after accounting for the larger width-24 core). Importantly, a miner cannot selectively publish one ticket while withholding the other — both tickets are deterministic outputs of the same permutation and can be independently verified by any node. This prevents ticket-grinding attacks where a miner might discard unfavorable tickets.

### 7.5 Quantum Resistance

Poseidon2's security against quantum adversaries:
- Grover's algorithm halves the effective hash bits: 248/2 = 124-bit quantum security
- Comparable to SHA-256 under quantum attack (256/2 = 128 bits)
- kHeavyHash: 256/2 = 128-bit quantum security
- **4-bit regression** from the transition (124 vs 128 quantum bits)

---

## 8. Comparison with Prior Work

### 8.1 Design Evolution

Five architectures were explored over multiple sessions before converging on ZK-PoUW:

| # | Architecture | Verdict | Primary rejection reason |
|---|-------------|---------|------------------------|
| 1 | Core Division (α = 1%) | Practical | U = 1%, cannot claim PoUW |
| 2 | Rate-4 Poseidon | Theoretical PoUW | FRI cascade cost, timing misalignment |
| 3 | ZK-Symbiotic (HW multithread) | Best engineering | Not PoUW by Ball et al. (nonce ≠ useful) |
| 4 | MatMul PoUW (Komargodski) | Domain-specific | O(n³) verification incompatible with 100 BPS |
| 5 | Direction C (Pure ZK PoUW) | Unsolved | Fiat-Shamir cascade barrier (open problem) |

The final design synthesizes insights from Architecture #2 (dual-use Poseidon outputs) and Architecture #3 (hardware multithreading with low-cost context switch), while avoiding their individual weaknesses:

- From #2: The idea that STARK intermediate hashes can serve as PoW tickets
- From #3: The MUX-based switching between STARK and PoW input sources
- Resolved #2's FRI cascade problem by treating re-computation as useful work
- Resolved #3's PoUW classification by adopting the practical U definition

### 8.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| STARK proof in block? | **No** (Option C) | Eliminates +3–5 MB/s bandwidth at 100 BPS |
| Hash function | **Poseidon2** | Stwo compatibility |
| Field | **M31** | Stwo standard (smallest multiplier, highest core density) |
| Permutation width | **24** (extended from 16) | PoW: 1 perm/hash + dual tickets; U = 16/24 ≈ 67%; ZK unaffected (SRAM-bound) |
| Operating mode | **Compression function** | 1 permutation/hash (vs 2 in sponge); recommended by [3] |
| PoW tickets per hash | **2** | S[8..15] and S[16..23]; +67% effective hashrate |
| PoW hash size | **248 bits** (8 M31 elements) | Close to 256-bit kHeavyHash/Bitcoin class (8-bit reduction) |
| Header hash elements | **8** (248 bits) | Birthday bound 2^124; matches PoW hash size |
| STARK enforcement | **None** | Market-driven ZK adoption; avoids bandwidth penalty |
| Nonce structure | **(v₁, v₂)** each 8 elements | 64 bytes, maps to Merkle child pair in STARK mode |

**Option C rationale:** Three options were evaluated for STARK enforcement:

| Option | STARK requirement | Bandwidth impact | PoUW (Ball et al.) |
|--------|------------------|-----------------|-------------------|
| A | Every block | +3–5 MB/s | Yes (strict) |
| B | Every N blocks | +50 KB/s (N=100) | Partial |
| **C** | **None** | **None** | **Practical (U≈67% when ZK active)** |

Option C was selected because it preserves Kaspa's existing bandwidth profile while enabling PoUW through economic incentives rather than protocol enforcement. U ≈ 67% when ZK demand exists (miners voluntarily run Stwo); U = 0% when no ZK demand (pure PoW fallback).

### 8.3 Relationship to Ball et al.

| Ball et al. requirement | Status |
|------------------------|--------|
| PoW computation produces useful output | Partial: STARK hashes produce useful output; PoW fill hashes provide security only (not "useful" per Ball et al.) |
| Verifier confirms usefulness | Not enforced (Option C) |
| Useful output bound to PoW evidence | Not enforced (STARK proof not in block) |

ZK-PoUW does not satisfy Ball et al.'s strict definition (0 of 3 criteria). It satisfies the **practical PoUW definition** (§2.2): when ZK demand exists, the ASIC performs ZK proof computation (U ≈ 67%) and PoW tickets emerge as byproducts. The 33% usefulness gap (= 8/24 of each permutation) is the inherent cost of PoW integration via width extension. When no ZK demand exists, U = 0% — the ASIC is a conventional miner. The protocol *enables* PoUW without *mandating* it, making the transition market-driven rather than protocol-enforced.

---

## 9. Open Questions

1. **Stwo production parameters.** The baseline parameters (Width 16, Rate 8, Capacity 8, R\_f = 8, R\_p = 14) are confirmed from source code [4]. However, round constants in the current Stwo implementation are explicit placeholders — both `EXTERNAL_ROUND_CONSTS` and `INTERNAL_ROUND_CONSTS` are uniformly set to `1234` with a TODO comment (`// TODO(shahars): Use poseidon's real constants.`). Final production constants are required before security analysis can be completed. The architectural parameters (width, round counts) are stable, but the concrete permutation is not yet defined for production use.

2. **Width 24 compression function security analysis.** Width 24 Poseidon2 has been analyzed in sponge mode, but compression function mode (all 24 elements visible) over M31 requires dedicated review. This analysis is a prerequisite for deployment, covering algebraic attacks, round number adequacy, and the dual-ticket independence property.

3. **Hard fork governance.** Transitioning from kHeavyHash to Poseidon2 renders existing kHeavyHash ASICs obsolete and requires community consensus.

4. **Mining pool protocol.** The 64-byte nonce (vs current 8-byte) requires updates to the stratum protocol for nonce range distribution. Stratum V2 may accommodate this natively.

5. **ZK market maturity.** The economic advantage of ZK-PoUW over Pure PoW depends on sufficient ZK proof demand. The timeline for this market to develop is uncertain, though ZK-PoUW ASICs function as standard PoW miners in the interim.

6. **Complementary bottleneck validation.** The claim that PoW (compute-bound) and STARK (memory-bound) can run simultaneously at full throughput requires hardware-level validation on actual ASIC designs.

7. **Compression function security.** This proposal uses Poseidon2 in compression function mode (all 24 state elements visible), unlike Stwo's standard sponge mode (8 capacity elements hidden). While compression mode is recommended by the Poseidon2 paper [3] and widely used for Merkle trees, the specific configuration (width 24 over M31, no capacity, dual PoW tickets) requires dedicated security analysis.

8. **Stwo verifier compatibility.** Standard Stwo verifiers using width-16 sponge cannot directly verify commitments produced by the width-24 compression function. Two integration modes are defined in §5.1: Mode A (width-24 compression, requires modified verifier) and Mode B (width-16 sponge emulation, standard compatible). Both produce identical ZK throughput (SRAM-bound). The choice depends on Stwo's upstream willingness to accept width-24 as a configuration option.

9. **Dual-ticket independence.** The two PoW tickets (S[8..15] and S[16..23]) are outputs of the same permutation and thus deterministically linked. While each is individually pseudorandom under PRP assumptions, the correlation should be formally analyzed to confirm no exploitable structure exists.

---

## 10. References

[1] Ball, M., Rosen, A., Sabin, M., & Vasudevan, P. N. (2017). "Proofs of Useful Work." *IACR Cryptology ePrint Archive*, 2017/203. https://eprint.iacr.org/2017/203

[2] Grassi, L., Khovratovich, D., Rechberger, C., Roy, A., & Schofnegger, M. (2021). "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems." *USENIX Security Symposium*. https://eprint.iacr.org/2019/458

[3] Grassi, L., Khovratovich, D., & Schofnegger, M. (2023). "Poseidon2: A Faster Version of the Poseidon Hash Function." *IACR Cryptology ePrint Archive*, 2023/323. https://eprint.iacr.org/2023/323

[4] StarkWare. "Stwo: A STARK Prover." https://github.com/starkware-libs/stwo

[5] Sompolinsky, Y., & Sutton, M. (2022). "The DAG KNIGHT Protocol: A Parameterless Generalization of Nakamoto Consensus." *IACR Cryptology ePrint Archive*, 2022/1494. https://eprint.iacr.org/2022/1494

[6] Kaspa. "kHeavyHash Specification." https://github.com/kaspanet/rusty-kaspa
