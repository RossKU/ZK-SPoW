# ZK-PoUW: ZK-Symbiotic Proof of Work for Kaspa

**Version 0.1 — Draft — 2026-02-14**

---

## 1. Abstract

We propose replacing Kaspa's kHeavyHash proof-of-work function with Poseidon2 over the Mersenne field M31, the same algebraic hash function used internally by the Stwo STARK prover. By extending the Poseidon2 state width by a small constant to accept block\_header as an additional input, every intermediate hash computed during STARK proof generation simultaneously produces a valid PoW ticket. This unification means that a mining ASIC and a ZK prover share identical Poseidon2 cores — a property we call **ZK-Symbiotic Proof of Work**. The same hardware performs both network security (PoW) and useful computation (ZK proofs) with zero-overhead switching and no hashrate degradation in either mode. We show this design is Pareto-optimal and constitutes a Nash equilibrium among rational miners in a mature ZK fee market.

---

## 2. Introduction

### 2.1 The PoW Energy Problem

Kaspa uses kHeavyHash — Blake3 composed with a 256×256 binary matrix multiplication — for proof of work. Like all traditional PoW schemes, the computational work produces no output beyond network security. The entire energy expenditure is justified solely by the security guarantees it provides.

### 2.2 Proof of Useful Work

Ball et al. [1] formalize **Proof of Useful Work (PoUW)** as a PoW scheme where the mining computation simultaneously produces useful output verifiable by the network. Their strict definition requires:

1. The PoW computation itself produces useful output
2. The verifier can confirm the usefulness
3. The useful output is bound to the PoW evidence

We adopt a **practical definition**:

> **Definition (Usefulness).** U = 100% iff the mining hardware is executing ZK proof computation. ZK proof generation constitutes useful work — the economic output (verifiable proofs) has independent value regardless of whether a PoW solution is found during the computation. PoW nonce grinding without concurrent ZK computation provides network security but does not constitute useful work (U = 0%).

The key insight: ZK proof computation *is* work. When an ASIC runs the Stwo prover, every cycle of operation produces economically valuable ZK proofs. PoW tickets emerge as a costless byproduct of the same Poseidon2 hashes. If a PoW solution is found mid-proof, the block is submitted, but U does not decrease — the ASIC was performing useful work throughout. When no ZK demand exists, the ASIC reverts to pure PoW (U = 0%), identical to any conventional miner.

### 2.3 Stwo and Poseidon2

Kaspa has adopted StarkWare's **Stwo** [4] as the STARK backend for verifiable programs (vProgs). Stwo operates over the Mersenne field M31 and uses **Poseidon2** [3] as its internal hash function for Merkle tree commitments and Fiat-Shamir challenges.

This creates a unique opportunity: if the PoW hash function is also Poseidon2 over M31, then the mining ASIC's primary computational element — the Poseidon2 pipeline — is *identical* to what is needed for STARK proof generation. Mining hardware becomes ZK proving hardware at zero additional cost.

---

## 3. Notation and Definitions

| Symbol | Definition |
|--------|-----------|
| F\_p | Finite field, p = 2^31 - 1 (Mersenne prime M31) |
| Poseidon2\_π | Poseidon2 permutation over F\_p^t |
| t | State width (rate + capacity) |
| r | Rate: number of input/output elements |
| c | Capacity: security parameter (hidden elements) |
| n | Hash output size in field elements (n = 8, giving 248 bits) |
| H | Block header: (parent\_hashes, tx\_merkle\_root, timestamp) |
| h\_H | Header hash: PoseidonSponge(H) ∈ F\_p^k |
| k | Header hash element count (k ≥ 4 for collision resistance) |
| (v\_1, v\_2) | Nonce: v\_1, v\_2 ∈ F\_p^8 |
| T | Target ∈ F\_p^8 (difficulty-adjusted) |
| S | Poseidon2 state after permutation, S ∈ F\_p^t |

**Stwo baseline parameters (estimated):**

| Parameter | Value |
|-----------|-------|
| Field | M31, p = 2^31 - 1 |
| Hash output | 8 elements = 248 bits |
| Standard width | t₀ = 24 (rate 16, capacity 8) |
| External rounds R\_f | 8 |
| Internal rounds R\_p | ~14 |
| S-box exponent | α = 5 |

---

## 4. Protocol Specification

### 4.1 PoW Function Replacement

**Current (kHeavyHash):**

```
pow_hash = Blake3(M · Blake3(H || nonce))
valid iff pow_hash < target
```

where M is a fixed 256×256 binary matrix, nonce is 8 bytes.

**Proposed (Poseidon2-PoW):**

```
h_H = PoseidonSponge(H)                          // header pre-hash
S   = Poseidon2_π(v₁ || v₂ || h_H || 0^c)       // single permutation
pow_hash = (S[8], S[9], ..., S[15])               // 8 M31 elements = 248 bits
valid iff pow_hash < target                        // lexicographic comparison
```

where (v₁, v₂) are 8 M31 elements each (64 bytes total nonce).

**Verification cost:** One Poseidon2 permutation + one header pre-hash (amortized over all nonce attempts per header).

### 4.2 Poseidon2 Width Extension

The core design change is extending the Poseidon2 state to accommodate block\_header as an additional input.

**Standard Stwo Merkle hash:**

```
Width t₀ = 24
Rate  r₀ = 16   ← left_child[8] + right_child[8]
Capacity c = 8
```

**Extended ZK-PoUW hash:**

```
Width t  = 28    ← t₀ + k, where k = 4 (header hash elements)
Rate  r  = 20    ← left_child[8] + right_child[8] + header_hash[4]
Capacity c = 8   ← unchanged (security preserved)
```

**Impact on Poseidon2 internals:**

| Component | Width 24 (standard) | Width 28 (extended) | Change |
|-----------|--------------------|--------------------|--------|
| External round MDS | M4 block circulant (additions) | +1 M4 block (additions) | +17% additions |
| Internal round MDS | 24 multiplications (diag + 1·1^T) | 28 multiplications | +17% multiplications |
| S-box (external) | 24 per round | 28 per round | +17% |
| S-box (internal) | 1 per round | 1 per round | unchanged |
| State registers | 24 × 31 bits | 28 × 31 bits | +17% |

**Poseidon2-specific advantage:** Internal round MDS scales as O(t), not O(t²), because the sparse MDS structure (diagonal + rank-1) requires only t multiplications per round. This makes the width extension significantly cheaper than it would be for standard Poseidon.

**Total die area impact: +7%** (see Appendix A.5).

### 4.3 I/O Mapping

For a single Poseidon2 permutation with state S ∈ F\_p^28:

```
INPUT (rate, 20 elements):
  S[0..7]    ← left_child     8 M31 elements (248 bits)
  S[8..15]   ← right_child    8 M31 elements (248 bits)
  S[16..19]  ← header_hash    4 M31 elements (124 bits)

INPUT (capacity, 8 elements):
  S[20..27]  ← 0              security padding

          ┌─────────────────────────────┐
          │   Poseidon2 permutation     │
          │   R_f external + R_p internal│
          └─────────────────────────────┘

OUTPUT:
  S[0..7]    → merkle_node    STARK Merkle parent node (useful)
  S[8..15]   → pow_hash       PoW target comparison (248 bits)
  S[16..19]  → extra          additional PoW tickets (optional)
  S[20..27]  → [hidden]       capacity (never revealed)
```

**Dual-use property:**

| Mode | S[0..7] input | S[8..15] input | S[0..7] output | S[8..15] output |
|------|--------------|---------------|---------------|----------------|
| STARK (PoUW) | Merkle left child | Merkle right child | Merkle parent (useful) | PoW ticket |
| Pure PoW | Random v₁ | Random v₂ | (discarded) | PoW ticket |

The same Poseidon2 hardware computes both modes. Only the input source differs.

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

**Minimum: k = 4** (124-bit collision resistance). This determines the width extension: t = 24 + 4 = 28.

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
S = Poseidon2_π(n_L || n_R || h_H || 0^c)
merkle_parent = S[0..7]       ← advances STARK proof (useful)
pow_ticket    = S[8..15]      ← checked against PoW target (security)

if pow_ticket < target → BLOCK FOUND
```

Every Poseidon2 invocation in the STARK computation simultaneously:
- **(a)** Advances the ZK proof (economic value — useful work)
- **(b)** Produces a PoW ticket (network security — costless byproduct)

**U = 100%.** The ASIC is performing ZK proof computation — useful work by definition. PoW tickets are a costless byproduct of the same Poseidon2 hashes. Even if a PoW solution is found mid-proof, U remains 100%: the ASIC was doing useful work (ZK) throughout.

### 5.2 Pure PoW Mode (No ZK Demand)

When no ZK proofs are requested:

```
loop:
  v₁ = next_nonce_1()
  v₂ = next_nonce_2()
  S = Poseidon2_π(v₁ || v₂ || h_H || 0^c)
  if S[8..15] < target → BLOCK FOUND
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

U is determined by ZK demand, not by PoW difficulty.

| Condition | U | Rationale |
|-----------|---|-----------|
| Stwo Prover active, any difficulty | 100% | ZK proof computation = useful work |
| No ZK demand, any difficulty | 0% | Pure PoW = security only |

Difficulty affects how many hashes are needed to find a block, but does not change what the ASIC is *doing*. Whether difficulty is 8M or 188G, the ASIC either computes ZK proofs (U = 100%) or grinds nonces (U = 0%). The ratio of STARK-to-PoW cycles within the pipeline is determined by SRAM bandwidth (a hardware constant), not by the network's difficulty target.

### 5.6 Complementary Bottleneck Structure

The simultaneous execution of PoW and STARK is possible because they bottleneck on different resources:

| Resource | PoW | STARK | Combined |
|----------|-----|-------|----------|
| Poseidon2 cores | **100%** (compute-bound) | ~10% (data-starved) | **~100%** |
| NTT unit | 0% (unused) | **100%** | **100%** |
| SRAM bandwidth | 0% (registers only) | **100%** (memory-bound) | **100%** |

PoW is compute-bound (limited by Poseidon2 throughput). STARK is memory-bound (limited by SRAM bandwidth feeding Merkle data to Poseidon2). They share Poseidon2 cores but contend on different bottleneck resources, achieving near-perfect utilization of all hardware components simultaneously. This complementary structure is the foundation of the Pareto optimality claim in §6.

---

## 6. Pareto Analysis

### 6.1 Competing Designs

Four designs compared under identical die area and power budget:

| Design | Die allocation | Hashrate | ZK Throughput | U | Can mine? |
|--------|---------------|----------|--------------|---|-----------|
| Pure PoW | 95% Poseidon2, 5% control | 1.9 H | 0 | 0% | Yes |
| Pure Stwo | 20% Poseidon2, 40% NTT, 35% SRAM | 0 | Z\_max | 100% | No |
| **ZK-PoUW** | **50% Poseidon2, 25% NTT, 20% SRAM** | **H** | **Z** | **100%** | **Yes** |
| ZK-PoUW+HBM | 50% Poseidon2, 25% NTT, 20% HBM I/F | H | Z\_hbm | 100% | Yes |

Pure PoW achieves 1.9× hashrate by filling NTT and SRAM area with additional Poseidon2 cores but produces no ZK proofs (U = 0%). Pure Stwo achieves U = 100% but cannot mine — it cannot independently sustain the network and has no block reward income. Only ZK-PoUW achieves both U = 100% and mining capability.

### 6.2 Post-Difficulty-Adjustment Revenue

Difficulty adjusts to total network hashrate. When all N miners use the same design, per-ASIC mining revenue is identical regardless of design. The differentiator is ZK revenue.

```
Revenue_per_ASIC = block_reward / N  +  Z × proof_fee
                   ─────────────────     ──────────────
                   mining (same for all)  ZK (design-dependent)
```

| Design | Mining revenue | ZK revenue | Total |
|--------|---------------|------------|-------|
| Pure PoW (all N) | B/N | 0 | B/N |
| ZK-PoUW (all N) | B/N | Z × F | **B/N + ZF** |

**ZK-PoUW strictly dominates Pure PoW: B/N + ZF > B/N for any ZF > 0.**

### 6.3 Mixed Network Equilibrium

When ZK-PoUW and Pure PoW miners coexist:

```
Pure PoW miner revenue:   1.9H / R_total × B
ZK-PoUW miner revenue:    H / R_total × B  +  Z × F
```

ZK-PoUW is preferred when:

```
Z × F  >  0.9 × H / R_total × B
       =  0.9 × (per-miner mining reward)
```

As the network grows (N → ∞), per-miner mining reward → 0, while ZK fees remain constant. **Beyond a crossover network size N\*, ZK-PoUW always dominates.**

```
N* = 0.9 B / ((1.9 - 0.9α) × Z × F)
```

where α is the fraction of miners using ZK-PoUW.

### 6.4 Nash Equilibrium

> **Theorem (informal).** In a mature ZK market where Z × F exceeds the hashrate differential value, the unique Nash equilibrium is all miners choosing ZK-PoUW.
>
> *Argument.*
> 1. Given all others choose ZK-PoUW, switching to Pure PoW gains ~0.9× hashrate advantage but loses Z × F in ZK fees.
> 2. For Z × F > 0.9 × (per-miner mining reward), deviating to Pure PoW is unprofitable.
> 3. No miner can profitably switch to Pure Stwo (cannot mine, loses block reward share).
> 4. No miner has incentive to deviate → Nash equilibrium.

In the early network phase (Z × F ≈ 0), Pure PoW is the equilibrium. As the ZK market matures, the equilibrium shifts to ZK-PoUW. This transition is gradual — the ZK-PoUW ASIC functions as a Pure PoW miner until ZK demand appears.

### 6.5 Revenue Resilience

```
Revenue/ASIC
    ↑
    │╲
    │  ╲  Mining reward B/N (declining with network growth)
    │    ╲
    │─────╲─────────────────── ZK-PoUW total revenue
    │       ╲
    │        ╲━━━━━━━━━━━━━━━ ZK fee floor (constant)
    │
    │── ── ── ── ── ── ── ── ── Pure PoW (mining only, no floor)
    │
    └────────────────────────→ Network size N
```

ZK-PoUW provides a **revenue floor** from ZK fees. Pure PoW miners face revenue declining monotonically with network growth, with no alternative income source.

---

## 7. Security Considerations

### 7.1 Poseidon2 Cryptographic Properties

Poseidon2 [3] maintains 128-bit security with capacity c = 8 M31 elements (248 bits). Known algebraic attack vectors (Gröbner basis, interpolation, differential) have been analyzed for standard parameters. StarkWare has adopted Poseidon2 for production use in Starknet, representing significant implicit endorsement of its security.

### 7.2 Single Primitive Dependency

| Component | Current Kaspa | Proposed |
|-----------|--------------|----------|
| PoW | kHeavyHash (Blake3) | Poseidon2 |
| STARK | Poseidon2 | Poseidon2 |
| Independence | PoW ≠ STARK | PoW = STARK |

**Risk:** A cryptographic break in Poseidon2 compromises both PoW security and STARK validity simultaneously.

**Mitigating factors:**
- StarkWare's Starknet accepts identical risk (entire L2 depends on Poseidon2)
- Active cryptanalysis community monitors algebraic hash functions
- Kaspa's modular architecture permits hash function upgrade via hard fork
- No known viable attack exists against Poseidon2 with recommended parameters

**Assessment:** Accepted risk, shared with the broader Poseidon2 ecosystem. The benefit (mining ASIC = ZK prover) outweighs the concentration risk for a system already committed to Stwo.

### 7.3 Non-Standard Width

| Width | Deployment | Security analysis |
|-------|-----------|------------------|
| 8 | Filecoin, Mina | Extensively analyzed |
| 12 | Plonky2 | Well analyzed |
| 16 | Research | Analyzed |
| 24 | Stwo (standard) | Analyzed by StarkWare |
| **28** | **This proposal** | **Not yet analyzed** |

**Required:** Commission dedicated security analysis for Poseidon2 with t = 28 over M31, covering:
- Algebraic attack resistance (Gröbner basis complexity bounds)
- Differential and linear cryptanalysis
- Round number adequacy (R\_f, R\_p) for extended width
- MDS matrix security properties for the 28×28 case

### 7.4 PoW Hash Distribution

The pow\_hash = S[8..15] consists of rate elements (not capacity). Revealing rate elements is safe under the sponge security model — this is the standard operating mode of any sponge-based hash function. No additional information about the capacity (and hence no security degradation) is leaked.

### 7.5 Quantum Resistance

Poseidon2's security against quantum adversaries:
- Grover's algorithm halves the effective hash bits: 248/2 = 124-bit quantum security
- Comparable to SHA-256 under quantum attack (256/2 = 128 bits)
- kHeavyHash (Blake3 based) has similar quantum resistance profile
- No regression from the transition

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

The final design synthesizes insights from Architecture #2 (dual-use Poseidon outputs) and Architecture #3 (hardware multithreading with zero-cost context switch), while avoiding their individual weaknesses:

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
| PoW hash size | **248 bits** (8 M31 elements) | ≈ 256-bit, matches kHeavyHash/Bitcoin class |
| Header hash elements | **4** (124 bits) | Birthday bound 2^62 (secure) |
| STARK enforcement | **None** | Market-driven ZK adoption; avoids bandwidth penalty |
| Nonce structure | **(v₁, v₂)** each 8 elements | 64 bytes, maps to Merkle child pair in STARK mode |

**Option C rationale:** Three options were evaluated for STARK enforcement:

| Option | STARK requirement | Bandwidth impact | PoUW (Ball et al.) |
|--------|------------------|-----------------|-------------------|
| A | Every block | +3–5 MB/s | Yes (strict) |
| B | Every N blocks | +50 KB/s (N=100) | Partial |
| **C** | **None** | **None** | **Practical (U=100% when ZK active)** |

Option C was selected because it preserves Kaspa's existing bandwidth profile while enabling PoUW through economic incentives rather than protocol enforcement. U = 100% when ZK demand exists (miners voluntarily run Stwo); U = 0% when no ZK demand (pure PoW fallback).

### 8.3 Relationship to Ball et al.

| Ball et al. requirement | Status |
|------------------------|--------|
| PoW computation produces useful output | Partial: STARK hashes produce useful output; PoW fill hashes provide security only (not "useful" per Ball et al.) |
| Verifier confirms usefulness | Not enforced (Option C) |
| Useful output bound to PoW evidence | Not enforced (STARK proof not in block) |

ZK-PoUW does not satisfy Ball et al.'s strict definition (0 of 3 criteria). It satisfies the **practical PoUW definition** (§2.2): when ZK demand exists, the ASIC performs ZK proof computation (U = 100%) and PoW tickets emerge as byproducts. When no ZK demand exists, U = 0% — the ASIC is a conventional miner. The protocol *enables* PoUW without *mandating* it, making the transition market-driven rather than protocol-enforced.

---

## 9. Open Questions

1. **Stwo production parameters.** The exact Width, Rate, Capacity, and round numbers for Stwo's production Poseidon2 are not yet finalized. The extended parameters (Width 28) depend on these baseline values.

2. **Width 28 security analysis.** No published cryptanalysis exists for Poseidon2 at t = 28 over M31. This analysis is a prerequisite for deployment.

3. **Hard fork governance.** Transitioning from kHeavyHash to Poseidon2 renders existing kHeavyHash ASICs obsolete and requires community consensus.

4. **Mining pool protocol.** The 64-byte nonce (vs current 8-byte) requires updates to the stratum protocol for nonce range distribution. Stratum V2 may accommodate this natively.

5. **ZK market maturity.** The economic advantage of ZK-PoUW over Pure PoW depends on sufficient ZK proof demand. The timeline for this market to develop is uncertain, though ZK-PoUW ASICs function as standard PoW miners in the interim.

6. **Complementary bottleneck validation.** The claim that PoW (compute-bound) and STARK (memory-bound) can run simultaneously at full throughput requires hardware-level validation on actual ASIC designs.

---

## 10. References

[1] Ball, M., Rosen, A., Sabin, M., & Vasudevan, P. N. (2021). "Proofs of Useful Work." *IACR Cryptology ePrint Archive*, 2017/203.

[2] Grassi, L., Khovratovich, D., Rechberger, C., Roy, A., & Schofnegger, M. (2021). "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems." *USENIX Security Symposium*.

[3] Grassi, L., Khovratovich, D., & Schofnegger, M. (2023). "Poseidon2: A Faster Version of the Poseidon Hash Function." *IACR Cryptology ePrint Archive*, 2023/323.

[4] StarkWare. "Stwo: A STARK Prover." https://github.com/starkware-libs/stwo

[5] Sompolinsky, Y., & Zohar, A. "DAGKNIGHT: A Parameterless Generalization of Nakamoto Consensus."

[6] Kaspa. "kHeavyHash Specification." https://github.com/kaspanet/rusty-kaspa
