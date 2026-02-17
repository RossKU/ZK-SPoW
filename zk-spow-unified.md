# ZK-SPoW: ZK-Symbiotic Proof of Work for Kaspa

**Version 0.3 — Draft — 2026-02-17**

---

## Abstract

We propose replacing Kaspa's kHeavyHash proof-of-work function with a Width-24 Poseidon2 compression function over the Mersenne field M31. Traditional Proof of Useful Work (PoUW) attempts to make PoW computation produce useful results — a direction that Ball et al. [1] show requires careful construction with practical deployment constraints. ZK-SPoW (ZK-Symbiotic Proof of Work) inverts this relationship: useful ZK computation (STARK Merkle hashing) naturally produces PoW tickets as a mathematical byproduct. When the ASIC executes STARK proofs (Symbiotic mode), every Merkle hash accepts a child pair plus block header digest as input and simultaneously outputs a parent node advancing the ZK proof and three PoW tickets — from a single permutation. The miner does not choose the PoW input; the STARK computation determines it. This unification carries a cost of U = t₀/t = 16/24 ≈ 67% — the remaining 33% of each permutation serves PoW integration. In return, the same hardware performs both network security and useful computation with zero switching overhead. Width-24 requires a Kaspa-specific STARK verifier supporting Width-24 Poseidon2 compression — a parameter change within the Poseidon2 framework, and an incremental cost within the kHeavyHash → Poseidon2 hard fork. Security claims assume final Poseidon2 production parameters (§8.1).

---

## 1. Introduction

### 1.1 The PoW Energy Problem

Kaspa uses kHeavyHash — cSHAKE256 (Keccak/SHA-3 family) composed with a 64×64 matrix multiplication over 4-bit nibbles — for proof of work. Like all traditional PoW schemes, the computational work produces no output beyond network security. The entire energy expenditure is justified solely by the security guarantees it provides.

### 1.2 The PoUW Paradox and Its Inversion

Ball et al. [1] formalize **Proof of Useful Work (PoUW)** as a PoW scheme where the mining computation simultaneously produces useful output. Their strict definition requires:

1. The PoW computation itself produces useful output
2. The verifier can confirm the usefulness
3. The useful output is bound to the PoW evidence

The fundamental tension: PoW requires random exploration (nonce grinding), while useful computation requires specific, deterministic work. Prior PoUW constructions [1][7][8][9] achieve provable security for specific problem classes, but require pre-hashing, SNARGs, or domain-specific verification that limits practical deployment in high-throughput blockchains (100 BPS).

**ZK-SPoW inverts this relationship.** Instead of making PoW results useful, we start from useful computation (STARK proof generation) and observe that PoW tickets emerge as a natural mathematical byproduct:

> **Conventional PoUW:** PoW computation → try to make results useful → fundamental constraints [1]
>
> **ZK-SPoW:** Useful ZK computation → PoW tickets as mathematical byproduct → no contradiction

> **Definition (ZK-SPoW).** A PoW scheme where the hash function is a width-extended Poseidon2 compression function operating on STARK Merkle data, such that every permutation simultaneously advances a ZK proof and produces PoW tickets.

The mechanism: STARK proof generation requires millions of Merkle hashes. Each Width-24 Poseidon2 Merkle hash takes (left\_child, right\_child, header\_digest) as input and produces (pow\_ticket₀, pow\_ticket₁, pow\_ticket₂) as output, where pow\_ticket₀ = merkle\_parent simultaneously advances the ZK proof. All three output regions are checked against the difficulty target. The miner cannot choose the Merkle inputs — they are determined by the STARK computation. The "random exploration" required for PoW occurs naturally because STARK Merkle tree hashes are pseudorandom from the miner's perspective.

We define usefulness as the fraction of each permutation that advances ZK computation:

> **Definition (Usefulness).**
> - U = t₀/t = 16/24 ≈ **67%** when the ASIC is executing ZK proof computation (Symbiotic mode)
> - U = **0%** when the ASIC is grinding nonces without concurrent ZK computation (Pure PoW mode)
>
> The 33% overhead per permutation is the cost of PoW integration: 8 of 24 input state elements carry header\_digest rather than ZK data. This is the inherent price of unifying ZK and PoW into a single permutation — not waste, but the cost of symbiosis. On the output side, all three 8-element regions serve as PoW tickets, and S[0..7] simultaneously advances the ZK proof — the purest form of symbiosis.
>
> U is a per-permutation metric. The time-averaged usefulness across all Poseidon2 cycles is U\_avg = f\_sym × U, where f\_sym is the fraction of cycles executing STARK Merkle hashes (determined by memory bandwidth; see §4.6 and §A.5).

If a PoW solution is found mid-proof, the block is submitted without interrupting ZK computation. When no ZK demand exists, the ASIC reverts to pure PoW (U = 0%), identical to any conventional miner.

### 1.3 Stwo and Poseidon2

Kaspa is evaluating StarkWare's **Stwo** [4] as a potential STARK backend for verifiable programs (vProgs). Stwo operates over the Mersenne field M31 and uses **Poseidon2** [3] as its internal hash function for Merkle tree commitments and Fiat-Shamir challenges.

This creates a unique opportunity: if the PoW hash function is also Poseidon2 over M31, then the mining ASIC's primary computational element — the Poseidon2 pipeline — can serve both STARK proof generation and PoW mining. The cost is a Poseidon2 width extension from 16 to 24 elements with increased round count (+44–105% core area, ~+22–50% die area depending on implementation; see §3.2).

**Stwo-Kaspa verifier.** Standard Stwo uses Width-16 Poseidon2 in sponge mode. Width-24 Poseidon2 is a different cryptographic function (different MDS matrix, different round count, different S-box count per external round). ZK-SPoW therefore requires a Kaspa-specific verifier supporting Width-24 compression — a parameter change within Poseidon2's design framework [3], not a new cryptographic construction. Since the kHeavyHash → Poseidon2 transition already requires a hard fork with full-node verifier updates, the Width-24 adaptation is an incremental cost.

### 1.4 Contributions

1. **PoUW paradox inversion.** We formalize ZK-Symbiotic Proof of Work, a construction where useful ZK computation (STARK Merkle hashing) naturally produces PoW tickets as a cryptographically bound byproduct — inverting the traditional PoUW direction explored by Ball et al. [1], Ofelimos [7], and Komargodski et al. [8][9] (§1.2).

2. **Width-24 Poseidon2 parameterization.** We specify Width-24 Poseidon2 over M31 in compression function mode and verify its security parameters: R\_p = 22 internal rounds for 128-bit security with D = 5, computed via Plonky3's round number formula [10]. The per-ticket S-box cost decreases by 50% compared to Width-16 despite 51% more S-box operations per permutation, due to the triple-ticket structure (§6.3).

3. **Complementary bottleneck architecture.** We demonstrate that PoW mining (compute-bound) and STARK proof generation (memory-bound) can share Poseidon2 hardware with zero-cycle switching overhead, and provide gate-level ASIC architecture analysis for a 7nm implementation (§4, Appendix A).

---

## 2. Notation and Definitions

| Symbol | Definition |
|--------|-----------|
| F\_p | Finite field, p = 2^31 - 1 (Mersenne prime M31) |
| Poseidon2\_π | Poseidon2 permutation over F\_p^t |
| t | State width (number of field elements in permutation) |
| r | Rate: number of input/output elements (sponge mode) |
| c | Capacity: security parameter (sponge mode; hidden elements) |
| n | Hash output size in field elements (n = 8, giving 248 bits) |
| H | Block header (all consensus fields; see §3.4) |
| h\_H | Header digest: PoseidonSponge(H excluding nonce) ∈ F\_p^k |
| k | Header digest element count (k = 8 for symmetric I/O and three PoW tickets) |
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

**Note:** All security claims in this paper assume final production round constants. The current Stwo implementation uses placeholder values (`EXTERNAL_ROUND_CONSTS` and `INTERNAL_ROUND_CONSTS` uniformly set to `1234`; see §8.1).

---

## 3. Protocol Specification

### 3.1 PoW Function Replacement

**Current (kHeavyHash):**

```
pre_pow_hash = Blake2b(H excluding nonce and timestamp)
inner        = cSHAKE256_PoW(pre_pow_hash || timestamp || nonce)
pow_hash     = cSHAKE256_Heavy(M · inner ⊕ inner)
valid iff pow_hash < target
```

where M is a 64×64 full-rank matrix over 4-bit nibbles (generated from pre\_pow\_hash via XoShiRo256++), nonce is 8 bytes (u64). The inner hash splits into 64 nibbles for matrix-vector multiplication; `cSHAKE256_PoW` and `cSHAKE256_Heavy` use domain strings `"ProofOfWorkHash"` and `"HeavyHash"` respectively [6].

**Proposed (Poseidon2-PoW):**

```
h_H = PoseidonSponge(H excluding nonce)           // amortized pre-hash (8 M31 elements)
S   = Poseidon2_π(v₁ || v₂ || h_H)              // single permutation, width 24
pow_hash₀ = (S[0], S[1], ..., S[7])               // 8 M31 elements = 248 bits
pow_hash₁ = (S[8], S[9], ..., S[15])              // 8 M31 elements = 248 bits
pow_hash₂ = (S[16], S[17], ..., S[23])            // 8 M31 elements = 248 bits
valid iff pow_hash₀ < target OR pow_hash₁ < target OR pow_hash₂ < target
```

where (v₁, v₂) are 8 M31 elements each (64 bytes total nonce). The permutation operates in **compression function mode** — all 24 input elements are visible (no hidden capacity). This differs from Stwo's standard sponge mode (width 16, rate 8, capacity 8) but is a recommended Poseidon2 usage mode [3]. Each permutation produces **three PoW tickets** — one from each 8-element output region. The comparison `pow_hash < target` interprets both as 248-bit unsigned integers via big-endian concatenation of 8 M31 elements, each zero-padded to 31 bits. In Symbiotic mode, pow\_hash₀ = merkle\_parent: the same output advances the ZK proof and serves as a PoW ticket (reading the value does not modify it).

**Verification cost:** One Poseidon2 permutation (width 24) + three target comparisons + one header pre-hash (amortized).

**Standard PoW structure.** ZK-SPoW is conventional hash-based PoW: miners explore a nonce space by computing Poseidon2 permutations and comparing outputs against a difficulty target, identically to Nakamoto-style PoW. The ZK component (Symbiotic mode, §4.1) is an optional revenue source sharing the same hardware — it does not modify the PoW function, security model, or difficulty adjustment. When no ZK demand exists, the ASIC reverts to conventional PoW mining (§4.2) with identical security guarantees.

### 3.2 Poseidon2 Width Extension

The core design change is extending the Poseidon2 permutation width to accommodate header digest as an additional input, and switching from sponge mode to compression function mode.

#### 3.2.1 Stwo Baseline

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

#### 3.2.2 Design Alternatives

Three approaches to integrate header digest into the Merkle hash:

| Design | Width | Mode | Perm/hash | PoW tickets | Core area Δ | Die area Δ |
|--------|-------|------|-----------|-------------|-------------|------------|
| A: 3rd absorb | 16 | Sponge | 3 (+50%) | 1 | 0% | 0% |
| B: Width 20 | 20 | Compression | 1 | 1 (+4 wasted) | +22% | ~+11% |
| **C: Width 24** | **24** | **Compression** | **1** | **3** | **+44–105%** | **~+22–50%** |

**Design A:** No Poseidon2 modification. Standard width-16 sponge with 3 absorptions (left, right, header). 3 permutations per Merkle hash; ZK throughput is unaffected under SRAM-bandwidth-bound operation (§4.6).

**Design B:** Extend to width 20 with header digest[4]. Compression function mode, 1 permutation per hash. But 4 output elements are unused (asymmetric 8+8+4 I/O).

**Design C (selected):** Extend to width 24 with header digest[8]. Symmetric 8+8+8 I/O — all output elements are useful. **3 PoW tickets per permutation** (S[0..7], S[8..15], and S[16..23]; S[0..7] doubles as merkle\_parent in Symbiotic mode). Header digest security doubles (248 vs 124 bits). Width 24 is within the Poseidon2 paper's analyzed parameter range [3]. **ZK throughput is unaffected**: STARK Merkle hashing is SRAM-bandwidth-bound regardless of core width (see §4.6). The width extension cost manifests as U = 16/24 ≈ 67% (§1.2). **Requires Stwo-Kaspa verifier** supporting Width-24 compression function mode (see §1.3).

**Design A vs C tradeoff.** Under current SRAM bandwidth (~200 GB/s; §A.5), Design A's smaller Width-16 cores yield more cores per die and higher total PoW hashrate despite 3 perm/hash. However, Design A's 3× permutation cost per STARK hash becomes a bottleneck as memory bandwidth increases: at HBM-class bandwidth (>1 TB/s), STARK saturates Design A's Poseidon2 capacity, leaving minimal room for PoW. Design C's 1 perm/hash scales linearly with bandwidth, maintaining full PoW throughput at any memory tier. Design C also simplifies scheduling (each permutation is stateless, vs sponge state tracking across 3 absorptions in Design A). The tradeoff: Design A uses well-analyzed Width-16 parameters with no Stwo modification; Design C requires a Width-24 Stwo-Kaspa verifier (§8.3) but is future-proof for higher-bandwidth memory architectures.

#### 3.2.3 Proposed Extension (Design C)

```
Standard Stwo:    Width t₀ = 16,  Sponge (rate 8, capacity 8)
Proposed ZK-SPoW: Width t  = 24,  Compression function (all 24 visible)
```

**Impact on Poseidon2 internals:**

| Component | Width 16 (standard) | Width 24 (proposed) | Change |
|-----------|--------------------|--------------------|--------|
| External MDS | 4 M4 blocks (additions) | 6 M4 blocks (additions) | +50% additions |
| Internal MDS | 16 multiplications (diag + 1·1^T) | 24 multiplications | +50% multiplications |
| S-box (external) | 16 per round | 24 per round | +50% |
| S-box (internal) | 1 per round | 1 per round | unchanged |
| Internal rounds R\_p | 14 | 22 | +57% (see §6.3) |
| Total rounds | 22 (8+14) | 30 (8+22) | +36% |
| S-box operations | 142 | 214 | +51% |
| State registers | 16 × 31 = 496 bits | 24 × 31 = 744 bits | +50% |

Internal round MDS scales as O(t), not O(t²), because the sparse MDS structure (diagonal + rank-1) requires only t multiplications per round. This makes the width extension significantly cheaper than for standard Poseidon.

**Core area overhead: +44% (datapath width) to ~+105% (fully pipelined).** The datapath widens by 50% (24/16), and Width-24 requires R\_p = 22 internal rounds vs Width-16's R\_p = 14 (§6.3), adding +36% pipeline depth (30 vs 22 total rounds). In an iterative (round-reuse) design, core area increases by ~+50% (width only) with 36% more cycles per hash. In a fully pipelined design, core area approximately doubles. Total die area impact: **~+22% to ~+50%** depending on implementation (Poseidon2 is 50% of die; see Appendix A). The 3-ticket property yields S-box cost per PoW ticket of 214/3 ≈ 71, a *50% decrease* compared to Width-16 (142), partially offsetting the core count reduction. The usefulness cost is U = t₀/t = 16/24 ≈ 67% — the 33% overhead per permutation serves PoW integration, not ZK computation.

#### 3.2.4 Compression Function vs Sponge

Standard Stwo uses sponge mode with 8 hidden capacity elements. This proposal uses **compression function mode** where all 24 state elements are visible. Key differences:

| Property | Sponge (Stwo standard) | Compression (ZK-SPoW) |
|----------|--------------|----------------------|
| Hidden state | 8 capacity elements | None (all visible) |
| Security model | Indifferentiability | Collision/preimage resistance of π |
| Permutations per Merkle hash | 2 | **1** |
| Width | 16 | 24 |
| PoW tickets per hash | 0 | **3** |

Both modes are established Poseidon2 usage modes [3]. The Poseidon2 paper recommends compression function mode for Merkle trees, noting up to 5× efficiency over sponge mode in compute-bound settings. On ASIC implementations, STARK Merkle throughput is typically SRAM-bandwidth-bound (see §A.5), so the per-permutation efficiency gain manifests as more Poseidon2 cycles available for PoW rather than faster ZK proof generation. Width 24 is within the paper's analyzed parameter range [3][10].

### 3.3 I/O Mapping

For a single Poseidon2 permutation with state S ∈ F\_p^24 (compression function mode):

```
INPUT (24 = 8+8+8, all visible):
  S[0..7]    ← left_child      8 M31 elements (248 bits)
  S[8..15]   ← right_child     8 M31 elements (248 bits)
  S[16..23]  ← header_digest   8 M31 elements (248 bits)

          ┌──────────────────────────────────┐
          │   Poseidon2 permutation (t = 24) │
          │   R_f = 8 external + R_p = 22 int│
          └──────────────────────────────────┘

OUTPUT (24 = 8+8+8, all visible):
  S[0..7]    → pow_ticket₀     merkle_parent (STARK) AND PoW ticket 0 (248 bits)
  S[8..15]   → pow_ticket₁     PoW ticket 1 (248 bits)
  S[16..23]  → pow_ticket₂     PoW ticket 2 (248 bits)
```

Symmetric 8+8+8 I/O with no unused output elements. No capacity elements — compression function mode exposes all state elements. Security relies on the Poseidon2 permutation's collision resistance and PRP properties (see §3.2.4). S[0..7] serves dual roles: it advances the STARK Merkle tree (as merkle\_parent) and is checked against the PoW target. Reading S[0..7] for PoW comparison does not modify the value used by the STARK computation.

**Note on full diffusion:** The Poseidon2 permutation mixes all 24 state elements through its MDS matrix every round. All output elements are functions of all input elements — there is no structural separation between output regions. The 8+8+8 labeling is a convention for reading the output, not a property of the permutation. This means merkle\_parent depends on header\_digest, and the Stwo-Kaspa verifier must account for this (see §1.3).

**Dual-use property:**

| Mode | S[0..7] in | S[8..15] in | S[16..23] in | S[0..7] out | S[8..15] out | S[16..23] out |
|------|-----------|------------|-------------|-----------|-------------|--------------|
| Symbiotic | left child | right child | header\_digest | merkle parent + PoW ticket₀ | PoW ticket₁ | PoW ticket₂ |
| Pure PoW | v₁ | v₂ | header\_digest | PoW ticket₀ | PoW ticket₁ | PoW ticket₂ |

The same Poseidon2 hardware computes both modes. Only the input source for S[0..15] differs (SRAM Merkle data vs random nonces). S[16..23] is always header\_digest. **Each permutation produces 3 PoW tickets** — one from each 8-element output region.

**Note on ticket granularity.** The 3 × 8-element partition is a protocol convention, not a cryptographic constraint. Under PRP, any non-overlapping partition of the 24 output elements yields independent pseudorandom tickets. Finer partitions (e.g., 6 × 4 elements at 124 bits each) increase the ticket count but decrease per-ticket bit-length; difficulty adjustment absorbs the ticket count change, leaving per-miner revenue unchanged (§5.2). The 8-element grouping is selected for compatibility with Stwo's hash output convention (8 M31 elements = 248 bits) and sufficient difficulty target granularity at network scale.

### 3.4 Block Structure

**Proposed header** (change from current Kaspa in bold):

```
Header {
  version:                  u16
  parents_by_level:         [[Hash]]       // DAGKnight multi-level parents
  hash_merkle_root:         Hash           // transaction Merkle root
  accepted_id_merkle_root:  Hash
  utxo_commitment:          Hash
  timestamp:                u64            // milliseconds
  bits:                     u32            // difficulty target
  nonce:                    [F_p; 16]      // **64 bytes (currently u64 = 8 bytes)**
  daa_score:                u64
  blue_work:                BlueWorkType
  blue_score:               u64
  pruning_point:            Hash
}
```

The only structural change is the nonce expansion from `u64` (8 bytes) to `[F_p; 16]` (64 bytes, +56 bytes per block, ~0.04% of 125 KB). The nonce maps to Poseidon2 input positions S[0..15] as (v₁, v₂), each 8 M31 elements.

| | kHeavyHash (current) | Poseidon2-PoW (proposed) |
|---|---|---|
| Nonce | u64 (8 bytes) | [F\_p; 16] (64 bytes) |
| PoW function | kHeavyHash → 256-bit | Poseidon2 Width-24 → 248-bit |
| Block hash | Blake2b-256 → 256-bit | Blake2b-256 → 256-bit (**unchanged**) |

**Block hash vs PoW hash.** Kaspa computes block identity and PoW with separate hash functions. The block hash (Blake2b-256 over the full serialized header including nonce) provides DAG references and block identification — unchanged by this proposal. Only the PoW function is replaced: h\_H = PoseidonSponge(header excluding nonce) is the amortized pre-hash, combined with the nonce in Width-24 Poseidon2 (§3.1). The 248-bit PoW output does not affect the 256-bit block hash.

**STARK proofs are NOT included in the block.** They are submitted as independent transactions in the mempool, providing economic value to the ZK ecosystem (vProgs fees). This eliminates the +3–5 MB/s bandwidth overhead that would result from mandatory per-block STARK proofs at 100 BPS.

### 3.5 Header Digest Collision Resistance

The header digest h\_H compresses the block header (excluding nonce) into k field elements. If k is too small, an attacker can find two headers H\_A ≠ H\_B with identical h\_H, allowing PoW solutions to be transplanted between conflicting blocks.

| k (elements) | Bits | Birthday bound | Security |
|-------------|------|---------------|----------|
| 1 | 31 | 2^15 ≈ 32K | INSECURE |
| 2 | 62 | 2^31 ≈ 2 × 10^9 | INSECURE |
| 4 | 124 | 2^62 ≈ 4.6 × 10^18 | SECURE |
| 8 | 248 | 2^124 | Conservative but justified |

**Minimum: k = 4** (124-bit collision resistance). We choose **k = 8** (248-bit, matching PoW hash size) to enable symmetric 8+8+8 I/O and 3 PoW tickets per permutation. The triple-ticket structure and high permutation rate (~10⁹/sec per ASIC) make the conservative choice appropriate. Width extension: t = 16 + 8 = 24.

---

## 4. Operating Modes

### 4.1 Symbiotic Mode (Stwo Prover Active)

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
merkle_parent = S[0..7]       ← advances STARK proof AND checked as PoW ticket₀
pow_ticket₁   = S[8..15]      ← checked against PoW target
pow_ticket₂   = S[16..23]     ← checked against PoW target

if S[0..7] < target OR S[8..15] < target OR S[16..23] < target → BLOCK FOUND
```

Every Poseidon2 invocation in the STARK computation simultaneously:
- **(a)** Advances the ZK proof (economic value — the useful work that drives the permutation)
- **(b)** Produces PoW tickets (network security — mathematical byproduct of the same permutation)

The miner does not choose the Merkle inputs (n\_L, n\_R) — they are determined by the STARK computation. The "random exploration" required for PoW occurs naturally because STARK Merkle tree hashes are pseudorandom from the miner's perspective.

**U = t₀/t = 16/24 ≈ 67%.** The 33% usefulness gap is the width extension overhead: 8 of 24 input state elements carry header\_digest rather than ZK data. There is no "ZK only" mode — every Width-24 permutation produces PoW tickets, regardless of input.

**Header digest and Merkle tree.** Because Poseidon2's MDS matrix provides full diffusion, the merkle\_parent output depends on header\_digest. This means the STARK Merkle tree is bound to a specific block header. At ~2G hash/sec, a complete Merkle tree builds within one block interval (10ms at 100 BPS). The Stwo-Kaspa verifier reconstructs Merkle nodes using the same Width-24 compression with the known header\_digest.

**Header freshness.** A STARK proof spans multiple Merkle commitment phases (step 3 and FRI rounds) — typically O(10) phases — and takes seconds to complete (dominated by NTT and trace generation). The header\_digest is fixed per Merkle tree phase; each phase completes within one block interval (<10ms at ~2G hash/sec with ~2.08G hash throughput per phase). Between phases, the header\_digest register updates to the current block. PoW tickets from each phase are valid for that phase's header. Maximum staleness is 1 block (10ms): a PoW ticket found at the end of a Merkle phase references a header\_digest that is at most one block behind the DAG tip. In DAGKnight's DAG structure [5], this is well within tolerance — DAGKnight accepts blocks with parent sets up to k blocks deep (where k is the anticone parameter, typically k ≥ 10 at 100 BPS), and 1-block staleness is indistinguishable from normal parallel block production.

### 4.2 Pure PoW Mode (No ZK Demand)

When no ZK proofs are requested:

```
loop:
  v₁ = next_nonce_1()
  v₂ = next_nonce_2()
  S = Poseidon2_π(v₁ || v₂ || h_H)       // width 24, compression function
  if S[0..7] < target OR S[8..15] < target OR S[16..23] < target → BLOCK FOUND
```

Identical Poseidon2 pipeline, identical throughput. The only difference is the input source: random/sequential nonces instead of STARK Merkle children.

**U = 0%.** No ZK proof is being computed. The ASIC provides network security only, equivalent to any conventional PoW miner. This is not waste — security has value — but it is not useful work in the ZK-SPoW sense.

### 4.3 Linear Mode Transition

```
┌─ Poseidon2 Pipeline (always 100% utilized) ──────────────┐
│                                                            │
│  Input MUX (per-cycle decision):                           │
│    SRAM data ready  → STARK Merkle hash   (Symbiotic)    │
│    SRAM not ready   → PoW nonce hash      (PoW)          │
│                                                            │
│  Switching cost: 0 cycles (combinational MUX, ~300 gates) │
│  Hashrate: invariant across all modes                      │
└────────────────────────────────────────────────────────────┘
```

The transition between Symbiotic and Pure PoW is **per-cycle and linear**, not a discrete mode switch. When the Poseidon2 pipeline is executing STARK computation, that cycle is Symbiotic. When SRAM data is not ready, a random nonce hash is substituted and that cycle is PoW. The pipeline is always full — the ratio of Symbiotic to PoW cycles is determined by SRAM bandwidth, not by difficulty or protocol parameters.

### 4.4 Hashrate Invariance

> **Proposition.** Total PoW hashrate H is independent of the operating mode.
>
> *Argument.* H = N\_cores × throughput\_per\_core. Each core's throughput is 1 hash per pipeline\_depth cycles (fully pipelined), regardless of whether the input is a STARK Merkle pair or a random nonce. Input MUX adds zero latency (combinational logic). Therefore H is constant across Symbiotic, Pure PoW, and any mixed state.

### 4.5 Difficulty Independence

U is determined by ZK demand and width ratio, not by PoW difficulty.

| Condition | U | Rationale |
|-----------|---|-----------|
| Stwo Prover active, any difficulty | ≈67% (16/24) | ZK proof computation, minus width extension overhead |
| No ZK demand, any difficulty | 0% | Pure PoW = security only |

Difficulty affects how many hashes are needed to find a block, but does not change U. Whether difficulty is 8M or 188G, the ASIC either computes ZK proofs (U ≈ 67%) or grinds nonces (U = 0%). The ratio of STARK-to-PoW cycles within the pipeline is determined by SRAM bandwidth (a hardware constant), not by the network's difficulty target.

### 4.6 Complementary Bottleneck Structure

The simultaneous execution of PoW and STARK is possible because they bottleneck on different resources:

| Resource | PoW | STARK | Combined |
|----------|-----|-------|----------|
| Poseidon2 cores | **100%** (compute-bound) | Low (SRAM-starved) | **~100%** |
| NTT unit | 0% (unused) | **100%** | **100%** |
| SRAM bandwidth | 0% (registers only) | **100%** (memory-bound) | **100%** |

PoW is compute-bound (limited by Poseidon2 throughput). STARK is memory-bound (limited by SRAM bandwidth feeding Merkle data to Poseidon2). They share Poseidon2 cores but contend on different bottleneck resources, achieving near-perfect utilization of all hardware components simultaneously. Under 200 GB/s SRAM bandwidth, the STARK allocation is f\_sym ≈ 10% of Poseidon2 cycles, yielding U\_avg = f\_sym × U ≈ 6.7% time-averaged usefulness (see §A.5 for derivation). This complementary structure is the foundation of the economic analysis in §5.

**Width-24 efficiency.** Width-24 compression uses 1 permutation per Merkle hash versus Width-16 sponge's 2 permutations. This halves STARK's Poseidon2 cycle consumption, freeing more cycles for PoW. The STARK proof generation rate itself remains SRAM-bandwidth-bound (see §A.5 for quantitative analysis).

---

## 5. Pareto Analysis

### 5.1 Competing Designs

Four designs compared under identical die area and power budget:

| Design | Die allocation | Hashrate | ZK Throughput | U | Can mine? |
|--------|---------------|----------|--------------|---|-----------|
| Pure PoW | 95% Poseidon2, 5% control | ~1.9 H (die area: 95/50) | 0 | 0% | Yes |
| Pure Stwo | 20% Poseidon2, 40% NTT, 35% SRAM | 0 | Z\_max | 100% (t₀/t₀) | No |
| **ZK-SPoW** | **50% Poseidon2, 25% NTT, 20% SRAM** | **H** | **Z** | **≈67% (t₀/t)** | **Yes** |
| ZK-SPoW+HBM | 50% Poseidon2, 25% NTT, 20% HBM I/F | H | Z\_hbm | ≈67% (t₀/t) | Yes |

Pure PoW achieves ~1.9× hashrate on a die-area basis (95%/50% Poseidon2 allocation) but produces no ZK proofs (U = 0%). On a hashes-per-watt basis, the gap narrows to ~1.1–1.2× because idle NTT and SRAM contribute static leakage (~10–20% of dynamic power; varies by process node). Pure Stwo uses standard Width-16 cores (U = 100%) but cannot mine. ZK-SPoW achieves U ≈ 67% and mining capability — the 33% usefulness gap is the cost of PoW integration via width extension.

### 5.2 Economic Dominance

Difficulty adjusts to total network hashrate. When all N miners use the same design, per-ASIC mining revenue is B/N regardless of absolute hashrate. The differentiator is ZK revenue:

```
ZK-SPoW revenue = B/N + Z×F   >   B/N = Pure PoW revenue    (for any ZF > 0)
```

Pure PoW's ~1.1–1.2× power-efficiency advantage yields at most ~10–20% more mining revenue per watt in a mixed network. Once ZK fee income Z×F exceeds this margin, ZK-SPoW dominates. The crossover point depends on Kaspa's ZK market development, network growth trajectory, and adoption dynamics — detailed economic modeling is required for quantitative predictions.

**PoW security without ZK demand.** ZK-SPoW does not condition PoW security on ZK demand. When Z×F = 0 (no ZK market), the ASIC operates as a conventional PoW miner with a hash/watt disadvantage due to the die area overhead (§3.2.3). Difficulty adjustment absorbs this: in a homogeneous ZK-SPoW network, per-ASIC mining revenue is identical to a homogeneous Pure PoW network. The die overhead is the cost of optionality — it purchases the ability to capture ZK revenue when the market emerges, without sacrificing PoW security in the interim.

---

## 6. Security Considerations

### 6.1 Poseidon2 Cryptographic Properties

Poseidon2 [3] provides 128-bit security in sponge mode with capacity c = 8 M31 elements (248 bits). In compression function mode (ZK-SPoW), security relies on collision resistance and preimage resistance of the permutation — well-established properties within the Poseidon2 framework [3]. Known algebraic attack vectors (Gröbner basis, interpolation, differential) have been analyzed for standard parameters. StarkWare has adopted Poseidon2 for production use in Starknet, representing significant implicit endorsement of its security.

**Note:** All security claims in this paper assume final production round constants. The current Stwo implementation uses placeholder values (see §8.1).

### 6.2 Single Primitive Dependency

| Component | Current Kaspa | Proposed |
|-----------|--------------|----------|
| PoW | kHeavyHash (Blake2b + cSHAKE256) | Poseidon2 |
| STARK | Poseidon2 | Poseidon2 |
| Independence | PoW ≠ STARK | PoW = STARK |

**Risk:** A cryptographic break in Poseidon2 compromises both PoW security and STARK validity simultaneously. This concentration risk is shared with the broader Poseidon2 ecosystem (notably Starknet).

### 6.3 Width-24 Security

ZK-SPoW uses Width-24 Poseidon2 compression function for both STARK Merkle hashing and PoW. Security analysis:

**Collision resistance (STARK binding).** STARK Merkle tree integrity requires collision resistance of the Width-24 compression function. With 248-bit output (8 M31 elements), the birthday bound is 2^124, providing 124-bit collision security. Width-24 provides strictly more state than Width-16 (24 vs 16 elements), resulting in stronger diffusion and larger algebraic degree accumulation. This is within [3]'s analyzed parameter range.

**STARK soundness.** The header\_digest acts as a fixed salt in each Merkle hash: it is determined by the block header before proof generation begins and cannot be chosen adaptively by the prover. STARK soundness therefore reduces to collision resistance of the Width-24 compression function with a fixed third input — a strictly easier assumption than collision resistance under adversarially chosen inputs. The Stwo-Kaspa verifier reconstructs Merkle nodes using the same header\_digest, preserving the binding property.

**Preimage resistance (PoW security).** PoW requires only preimage resistance of the 248-bit output — trivially satisfied by 30 rounds of Poseidon2 (8 external + 22 internal).

**R\_p for Width-24.** The internal round count increases from R\_p = 14 (Width-16) to R\_p = 22 (Width-24) for 128-bit security at D = 5 over M31. This is computed via Plonky3's round number formula [10], which applies the security constraints from [2][3] plus the algebraic attack bound from Khovratovich et al. (ePrint 2023/537), with a security margin of R\_f += 2, R\_p × 1.075. The binding constraint is statistical (R\_f ≥ 6). Total rounds: 30 (8 + 22) vs 22 (8 + 14) for Width-16. Despite 51% more S-box operations per permutation (214 vs 142), the per-ticket cost *decreases* by 50% (214/3 ≈ 71 vs 142/1 = 142 S-boxes per PoW ticket) due to the triple-ticket structure. Supplementary verification: M\_I^k is invertible for all k = 1..48 (necessary condition for subspace trail resistance [3]). Diffusion analysis confirms full input-to-output dependency from the first external round. Algebraic degree after the full 30-round permutation exceeds 2^69, well above the 2^64 threshold for interpolation security at 128 bits.

### 6.4 PoW Hash Distribution

All three output regions — pow\_ticket₀ = S[0..7], pow\_ticket₁ = S[8..15], pow\_ticket₂ = S[16..23] — are outputs of the same Poseidon2 permutation. In compression function mode, all 24 state elements are visible by design. Security relies on the permutation's PRP properties: given a random-looking input, all output elements should be indistinguishable from random. The three PoW tickets are deterministically linked (same permutation), but each is individually pseudorandom. An attacker who could predict one ticket from another without computing the full permutation would violate the PRP assumption — equivalent to breaking Poseidon2. The full-round permutation (8 external + 22 internal) ensures all output elements are cryptographically mixed across all 24 state positions.

**Triple-ticket mining:** Under the PRP assumption on Poseidon2, the three tickets are mutually independent (Appendix B.7). With three 248-bit tickets per permutation, the per-permutation success probability is exact:

```
P(valid) = 1 - (1 - p)³ = 3p - 3p² + p³,    p = T/2^248
```

> **Note on 2^248 approximation.** Each M31 element ranges over [0, 2^31 − 2], so the exact denominator is (2^31 − 1)^8 rather than 2^248. The ratio (2^31 − 1)^8 / 2^248 = (1 − 2^{−31})^8 ≈ 1 − 3.7 × 10^{−9}. We use 2^248 throughout as a convenient approximation.

Note that the number of tickets per permutation does not affect mining economics in a homogeneous network: difficulty adjustment absorbs any change in per-permutation success probability, leaving per-miner revenue at B/N (§5.2). The triple-ticket structure is a natural consequence of Width-24's symmetric 8+8+8 output and Stwo's 8-element hash convention, not a hashrate optimization.

### 6.5 Quantum Resistance

Poseidon2's security against quantum adversaries:
- Grover's algorithm halves the effective hash bits: 248/2 = 124-bit quantum security
- Comparable to SHA-256 under quantum attack (256/2 = 128 bits)
- kHeavyHash: 256/2 = 128-bit quantum security
- **Delta:** −8 bits classical (248 vs 256) / −4 bits quantum (124 vs 128) from the transition. Both values remain well above the 100-bit security floor considered acceptable for PoW functions.

---

## 7. Comparison with Prior Work

### 7.1 Design Evolution

Five architectures were explored over multiple sessions before converging on ZK-SPoW:

| # | Architecture | Verdict | Primary rejection reason |
|---|-------------|---------|------------------------|
| 1 | Core Division (α = 1%) | Practical | U = 1%, cannot claim symbiosis |
| 2 | Rate-4 Poseidon | Theoretical | FRI cascade cost, timing misalignment |
| 3 | ZK-Symbiotic (HW multithread) | Best engineering | Not PoUW by Ball et al. (nonce ≠ useful) |
| 4 | MatMul PoUW (Komargodski [8]) | Domain-specific | O(n³) verification incompatible with 100 BPS |
| 5 | Direction C (Pure ZK PoUW) | Unsolved | Fiat-Shamir cascade barrier (open problem) |

The final design synthesizes insights from Architecture #2 (dual-use Poseidon outputs) and Architecture #3 (hardware multithreading with low-cost context switch), while avoiding their individual weaknesses:

- From #2: The idea that STARK intermediate hashes can serve as PoW tickets
- From #3: The MUX-based switching between STARK and PoW input sources
- Resolved #2's FRI cascade problem by treating re-computation as useful work
- Resolved #3's classification by the PoUW paradox inversion (§1.2)

### 7.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| STARK proof in block? | **No** (Option C) | Eliminates +3–5 MB/s bandwidth at 100 BPS |
| Hash function | **Poseidon2** | Stwo-Kaspa compatibility |
| Field | **M31** | Stwo standard (smallest multiplier, highest core density) |
| Permutation width | **24** (extended from 16) | 1 perm/hash + triple tickets; U = 16/24 ≈ 67%; ZK rate SRAM-bound |
| Operating mode | **Compression function** | 1 permutation/hash (vs 2 in sponge); recommended by [3] |
| PoW tickets per hash | **3** | S[0..7], S[8..15], S[16..23]; natural 8-element grouping |
| PoW hash size | **248 bits** (8 M31 elements) | Close to 256-bit kHeavyHash/Bitcoin class (8-bit reduction) |
| Header digest elements | **8** (248 bits) | Birthday bound 2^124; matches PoW hash size |
| STARK enforcement | **None** | Market-driven ZK adoption; avoids bandwidth penalty |
| Nonce structure | **(v₁, v₂)** each 8 elements | 64 bytes, maps to Merkle child pair in Symbiotic mode |
| Verifier | **Stwo-Kaspa** | Width-24 Poseidon2 compression; parameter change within [3]'s framework |

**Option C rationale:** Three options were evaluated for STARK enforcement:

| Option | STARK requirement | Bandwidth impact | ZK-SPoW benefit |
|--------|------------------|-----------------|-------------------|
| A | Every block | +3–5 MB/s | Maximum (strict ZK binding) |
| B | Every N blocks | +50 KB/s (N=100) | Partial |
| **C** | **None** | **None** | **Market-driven (U≈67% when ZK active)** |

Option C was selected because it preserves Kaspa's existing bandwidth profile while enabling symbiotic operation through economic incentives rather than protocol enforcement.

### 7.3 Relationship to Ball et al.

Ball et al. [1] formalize PoUW in the direction **PoW → useful output**. Their three criteria evaluate whether PoW computation can be redirected toward useful work. ZK-SPoW operates in the inverse direction: **useful computation → PoW output**.

| Ball et al. criterion | PoW → useful (their framework) | Useful → PoW (ZK-SPoW) |
|----------------------|-------------------------------|------------------------|
| PoW produces useful output | Partial: PoW-fill hashes are security-only | **Yes**: every Symbiotic permutation advances a STARK proof |
| Verifier confirms usefulness | Not enforced (Option C) | Verifiable: STARK proofs are publicly checkable |
| Useful output bound to PoW | Not enforced (proof not in block) | **Inherent**: same permutation produces both merkle\_parent and PoW tickets |

Under their original framework (PoW → useful), ZK-SPoW satisfies 0 of 3 criteria strictly. Under the inverted framing (useful → PoW), the evaluation changes: the STARK computation drives the permutation, PoW tickets are a cryptographically bound byproduct, and the proof is publicly verifiable.

The deeper point: Ball et al.'s hardness results [1] constrain the PoW → useful direction. ZK-SPoW sidesteps these constraints by never attempting to make PoW useful. Instead, useful computation (STARK proving) happens to produce PoW-valid outputs because Poseidon2's pseudorandom outputs naturally fall below the target at the expected rate.

Ofelimos [7] is the closest prior work, using SNARK proofs as useful work within a provably secure PoUW framework. Komargodski et al. [8][9] explore PoUW via matrix multiplication and external utility functions. ZK-SPoW differs from both: (1) the useful computation is market-driven rather than protocol-mandated (Option C), and (2) PoW tickets emerge as a byproduct of STARK hashing rather than through a separate verification mechanism.

---

## 8. Open Questions

1. **Stwo production parameters.** The baseline parameters (Width 16, Rate 8, Capacity 8, R\_f = 8, R\_p = 14) are confirmed from source code [4]. However, round constants in the current Stwo implementation are explicit placeholders — both `EXTERNAL_ROUND_CONSTS` and `INTERNAL_ROUND_CONSTS` are uniformly set to `1234` with a TODO comment (`// TODO(shahars): Use poseidon's real constants.`). Final production constants are required before security analysis can be completed.

2. **R\_p for Width-24 (resolved).** R\_p = 22 computed via Plonky3's round number formula [10]; independent cryptanalytic verification pending. See §6.3 for full security analysis.

3. **Stwo-Kaspa verifier.** Width-24 Poseidon2 compression is a different cryptographic function from Width-16 sponge (different MDS matrix, different round count). A Kaspa-specific verifier is required. Plonky3 [10] already provides a production Width-24 Poseidon2 implementation over M31 with verified parameters: external MDS = circ(2M4, M4, M4, M4, M4, M4) where M4 = [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]]; internal MDS = 1·1ᵀ + diag(V) with V = [−2, 1, 2, 4, ..., 2²²]; R\_f = 8, R\_p = 22. The remaining requirements are: (a) round constant finalization (Grain LFSR or PRNG; both Stwo and Plonky3 currently use non-production constants), (b) integration with Stwo's proof system, (c) independent security certification. StarkWare's willingness to accept Width-24 as an upstream configuration option determines whether this is a lightweight fork or a permanent maintenance burden.

4. **Hard fork governance.** Transitioning from kHeavyHash to Poseidon2 renders existing kHeavyHash ASICs obsolete and requires community consensus.

5. **Mining pool protocol.** The 64-byte nonce (vs current 8-byte) requires updates to the stratum protocol for nonce range distribution. Stratum V2 may accommodate this natively.

6. **ZK market maturity.** The economic advantage of ZK-SPoW over Pure PoW depends on sufficient ZK proof demand. The timeline for this market to develop is uncertain and requires dedicated economic modeling.

7. **Complementary bottleneck validation.** The claim that PoW (compute-bound) and STARK (memory-bound) can run simultaneously at full throughput requires hardware-level validation on actual ASIC designs.

8. **Triple-ticket independence (resolved).** Under the PRP assumption, for any fixed input x, the permutation output π(x) is uniformly distributed over F\_p^24. A uniform distribution on the product space F\_p^8 × F\_p^8 × F\_p^8 implies that S[0..7], S[8..15], S[16..23] are mutually independent. The joint success probability q = 1 − (1−p)³ = 3p − 3p² + p³ is therefore exact. Any detectable correlation would constitute a PRP distinguisher — equivalent to breaking Poseidon2. See Appendix B.7.

9. **Trace grinding (resolved).** Under the PRP assumption on Poseidon2, trace selection does not affect the PoW ticket success distribution. The total number of permutations across all STARK commitment phases (initial Merkle tree plus FRI rounds) is determined by protocol parameters and is invariant under trace selection. Each permutation produces three PoW tickets whose joint success probability q = 1−(1−p)³ ≈ 3p, p = T/2^248, is input-independent under PRP. The distribution of valid tickets follows Binomial(P, q) where P is the total permutation count — invariant across trace choices. Multi-trial grinding (k distinct traces, selecting the best outcome) incurs (k−1)/k waste from discarded proofs, yielding net loss for k ≥ 2. Header digest (h\_H) selection is equivalent to nonce grinding under PRP. See Appendix B for the full proof.

---

## Appendix A: ASIC Architecture Details

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
SRAM bandwidth:          ~200 GB/s (ideal; see note below)
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
| U (usefulness) | **≈67%** | t₀/t = 16/24; width extension overhead = 33% (see §1.2) |
| STARK proofs/sec | ~260 | 2.08G / 8M hashes per proof |
| PoW hashrate | ~63G effective | 21G perm/sec × 3 tickets |
| U\_avg | **≈6.7%** | f × U = 0.10 × 0.67; time-averaged usefulness (see §1.2) |

**Time-averaged usefulness.** U\_avg = f\_sym × U ≈ 0.10 × 0.67 ≈ **6.7%** under 200 GB/s SRAM bandwidth with the STARK prover continuously active. This means 6.7% of all Poseidon2 cycles advance ZK proofs; the remaining 93.3% provide PoW security only. In a conventional PoW network, U\_avg = 0%: all mining energy produces security and nothing else. ZK-SPoW's 6.7% represents computation that would otherwise have no useful output beyond block validation. U\_avg scales with memory bandwidth: ~13% at 400 GB/s, ~40% at 1.2 TB/s (§A.5.4). When the STARK prover is inactive (no ZK demand), U\_avg = 0% and the ASIC operates as a conventional PoW miner.

**f is not waste — it is a throughput allocation metric.** It describes how Poseidon2 cycles are allocated between STARK (memory-bandwidth-limited) and PoW (compute-limited). U ≈ 67% because 8 of 24 input state elements carry header\_digest rather than ZK data. The PoW cycles provide additional network security as a low-marginal-cost byproduct. The two workloads are complementary: PoW is compute-bound, STARK is memory-bound. They share Poseidon2 cores but bottleneck on different resources, achieving near-perfect utilization.

**SRAM bandwidth assumption.** The 200 GB/s figure assumes ideal conditions. Routing overhead, bank conflicts, and NTT/Poseidon2 arbitration may reduce effective bandwidth by 10–30%, proportionally lowering f\_sym and U\_avg. The qualitative conclusions (complementary bottleneck, PoW-dominated allocation) are robust to this range.

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

## Appendix B: Trace Grinding Analysis

We prove that trace selection in Symbiotic mode provides zero advantage for PoW mining under the PRP assumption on Poseidon2.

### B.1 Assumptions

1. **PRP.** The Poseidon2 permutation π: F\_p^24 → F\_p^24 is a pseudorandom permutation. For any input x, the output π(x) is indistinguishable from uniform over F\_p^24.
2. **Fixed tree sizes.** Each Merkle commitment phase i (i = 0 for the initial trace commitment, i = 1, ..., m for FRI rounds) has 2Nᵢ − 1 internal nodes, where Nᵢ is determined by the protocol (trace length, blowup factor, FRI folding rate). The values Nᵢ are independent of trace content.
3. **Fixed header digest.** The header digest h\_H ∈ F\_p^8 is fixed at the start of each Merkle commitment phase and constant throughout that phase.

### B.2 Ticket Count Invariance

Each Poseidon2 permutation in the Merkle tree produces three PoW tickets. The total number of permutations across all STARK phases is:

```
P = Σᵢ (2Nᵢ − 1)
```

Since each Nᵢ is a protocol parameter independent of the trace t:

```
∀ t₁, t₂:  P(t₁) = P(t₂) = P
```

The miner cannot increase the number of PoW tickets by selecting a different trace.

### B.3 Distribution Invariance

Under PRP, for any distinct inputs x₁, ..., x\_P to the permutation, the outputs π(x₁), ..., π(x\_P) are jointly pseudorandom. In a Merkle tree, all permutation inputs are distinct with overwhelming probability: two nodes sharing the same input (n\_L, n\_R, h\_H) requires a collision in the 248-bit child outputs, which occurs with probability at most C(P,2) · 2^{−248} — negligible for P ≤ 10⁷.

Each permutation produces three PoW tickets. Under PRP, the success event for each permutation (at least one ticket below target T) is a Bernoulli trial with parameter:

```
q = 1 − (1−p)³ ≈ 3p,    p = T / 2^248
```

This parameter depends only on the target T, not on the permutation input. Since inter-permutation independence holds (distinct inputs under PRP), the number of successful permutations follows:

```
V ~ Binomial(P, q)
```

Both parameters (P, q) are trace-independent. Therefore, not only the expectation E[V] = Pq but the *entire distribution* of valid tickets is invariant under trace selection. There is no trace for which the variance is smaller (guaranteeing a hit) or larger.

**Fiat-Shamir cascade.** Trace selection affects FRI round Merkle trees via the Fiat-Shamir challenge dependency on the initial Merkle root. Changing the trace changes all subsequent challenges, folding points, and FRI Merkle trees. However, each FRI tree size Nᵢ is protocol-determined, and PRP ensures each resulting ticket has identical success probability. The argument above extends to all commitment phases.

### B.4 Header Digest Grinding

The miner can produce different header digests h\_H by choosing different parent blocks or transaction sets. With the same trace but a new h\_H, the entire Merkle tree must be rebuilt (cost: P permutations), producing P new permutation triples. Under PRP, this is functionally equivalent to P pure PoW nonce hashes — identical cost, identical ticket distribution. Moreover, only one h\_H can be used for the STARK proof; the remaining attempts produce PoW tickets but discard the STARK computation. Header digest grinding offers no advantage beyond standard nonce grinding.

### B.5 Multi-Trial Grinding

A miner who computes k distinct traces and selects the one with the most valid PoW tickets:

**Cost:** k × (C\_NTT + C\_trace + P) permutations, where C\_NTT and C\_trace are the NTT and trace generation costs (counted conservatively as zero in the comparison below).

**Benefit:** max(V₁, ..., V\_k) where each Vᵢ ~ Binomial(P, q) independently.

For the same k × P Poseidon2 permutations in pure PoW mode, all tickets are valid (none discarded), yielding V\_PoW ~ Binomial(kP, q).

By linearity, E[V\_PoW] = kPq. The best-of-k selection gives E[max(V₁, ..., V\_k)] ≤ Pq + O(√(log k · Pq(1−q))). For any k ≥ 2:

```
E[max(V₁, ..., V_k)] < kPq = E[V_PoW]
```

Multi-trial grinding is strictly dominated by pure PoW. Including the NTT and trace generation overhead (omitted above) makes the comparison strictly worse for grinding.

### B.6 Merkle Tree Feedback Structure

In Width-24 compression, the Merkle parent output S[0..7] becomes an input to the next tree level, and h\_H occupies S[16..23] at every level. This creates structured, non-i.i.d. inputs to successive permutations. Under PRP, the permutation's output distribution is uniform regardless of input structure. The full 30-round Poseidon2 (8 external + 22 internal) provides complete diffusion across all 24 state elements. Any weakness in PRP for structured inputs would constitute a break of Poseidon2 itself — the same assumption underlying the PoW security analysis (§6.3). ∎

### B.7 Triple-Ticket Independence

The three PoW tickets pow\_ticket₀ = S[0..7], pow\_ticket₁ = S[8..15], and pow\_ticket₂ = S[16..23] are outputs of the same Poseidon2 permutation and therefore deterministically linked. We show that under PRP, this linkage carries no exploitable statistical correlation.

**Proposition.** Under the PRP assumption on Poseidon2, pow\_ticket₀, pow\_ticket₁, and pow\_ticket₂ are mutually independent.

**Proof.** Let π: F\_p^24 → F\_p^24 be a PRP. For any fixed input x ∈ F\_p^24, the output π(x) is computationally indistinguishable from a uniform sample over F\_p^24.

Partition F\_p^24 = F\_p^8 × F\_p^8 × F\_p^8 as (A, B, C) where A = S[0..7], B = S[8..15], C = S[16..23]. If (A, B, C) is uniform over F\_p^24, then A, B, C are mutually independent, each uniform over F\_p^8. This is a standard property of product probability spaces: the uniform distribution on a product space implies independence of coordinate projections.

Therefore:

```
P(A < T) = p = T/2^248
P(B < T) = p
P(C < T) = p
P(A < T ∧ B < T ∧ C < T) = p³
P(A < T ∨ B < T ∨ C < T) = 1 − (1−p)³ = 3p − 3p² + p³
```

The quantity q = 1 − (1−p)³ used in §6.4 and Appendix B.3 is exact under PRP, not an approximation.

**Implication for mining.** A miner who observes any one ticket gains no information about whether the other two are below target. The only way to evaluate all tickets is to compute the full Poseidon2 permutation — which already produces all three simultaneously. There is no early-termination optimization. In Symbiotic mode, pow\_ticket₀ = merkle\_parent: reading it for PoW comparison does not modify the value used by the STARK Merkle tree.

**Distinguisher reduction.** Any statistical test T that detects correlation among S[0..7], S[8..15], and S[16..23] across multiple Poseidon2 evaluations can be converted into a PRP distinguisher: run T on π vs a truly random permutation ρ, and distinguish based on whether the test detects correlation. The advantage of T as a correlator equals its advantage as a PRP distinguisher. Under the PRP assumption, no such efficient T exists. ∎

---

## 9. References

[1] Ball, M., Rosen, A., Sabin, M., & Vasudevan, P. N. (2017). "Proofs of Useful Work." *IACR Cryptology ePrint Archive*, 2017/203. https://eprint.iacr.org/2017/203

[2] Grassi, L., Khovratovich, D., Rechberger, C., Roy, A., & Schofnegger, M. (2021). "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems." *USENIX Security Symposium*. https://eprint.iacr.org/2019/458

[3] Grassi, L., Khovratovich, D., & Schofnegger, M. (2023). "Poseidon2: A Faster Version of the Poseidon Hash Function." *IACR Cryptology ePrint Archive*, 2023/323. https://eprint.iacr.org/2023/323

[4] StarkWare. "Stwo: A STARK Prover." https://github.com/starkware-libs/stwo

[5] Sompolinsky, Y., & Sutton, M. (2022). "The DAG KNIGHT Protocol: A Parameterless Generalization of Nakamoto Consensus." *IACR Cryptology ePrint Archive*, 2022/1494. https://eprint.iacr.org/2022/1494

[6] Kaspa. "kHeavyHash Specification." https://github.com/kaspanet/rusty-kaspa

[7] Fitzi, M., Kiayias, A., Panagiotakos, G., & Russell, A. (2022). "Ofelimos: Combinatorial Optimization via Proof-of-Useful-Work." *Crypto 2022*. https://eprint.iacr.org/2021/1379

[8] Komargodski, I. & Weinstein, O. (2025). "Proofs of Useful Work from Arbitrary Matrix Multiplication." *IACR Cryptology ePrint Archive*, 2025/685. https://eprint.iacr.org/2025/685

[9] Bar-On, Y., Komargodski, I., & Weinstein, O. (2025). "Proof of Work With External Utilities." arXiv:2505.21685. https://arxiv.org/abs/2505.21685

[10] Plonky3. "A Toolkit for Polynomial IOPs." `poseidon2/src/round_numbers.rs`, `mersenne-31/src/poseidon2.rs`. https://github.com/Plonky3/Plonky3 (accessed 2026-02-16).
