# ZK-SPoW: ZK-Symbiotic Proof of Work

**Anonymous** (Preprint. Work in progress.)

February 2026 — Version 2.0 (Draft)

---

## Abstract

Proof-of-work (PoW) blockchains expend energy solely for network security. Proof of Useful Work (PoUW) aims to reclaim this cost. Recent constructions achieve provable security for specific problem classes—matrix multiplication [8] and combinatorial optimization [7]—building on the foundational study by Ball et al. [1]. Bar-On et al. [9] analyze the game-theoretic equilibrium. These operate in the direction PoW → useful output, prioritizing protocol-enforced usefulness with domain-specific verification. ZK-based approaches [14]—using STARK proof generation as useful work—are a natural candidate, but face two issues: stateful, multi-phase STARK proving yields trial intervals of tens of milliseconds to seconds (non-memoryless); and losing miners' proofs are discarded, so effective usefulness under difficulty adjustment is no better than pure PoW.

**ZK-SPoW** (ZK-Symbiotic Proof of Work) inverts the PoUW relationship: instead of making PoW computation useful, useful ZK computation (STARK Merkle hashing) naturally produces PoW tickets as a computational byproduct. This inversion resolves the non-memoryless problem—under the pseudorandom permutation (PRP) assumption, each Poseidon2 permutation within the STARK is computationally indistinguishable from an independent Bernoulli trial at nanosecond granularity, rather than proof-level intervals of tens of milliseconds to seconds. It also eliminates proof waste: losing miners' ZK computation remains useful regardless of PoW outcome. Header staleness is bounded by one Merkle commitment phase (~3 ms on GPU (measured)).

We do not claim protocol-level enforcement of useful work. Each Poseidon2 evaluation in STARK Merkle hashing simultaneously produces a Merkle parent (ZK output) and PoW tickets (checked against the difficulty target)—a mathematical byproduct, not a protocol mandate. Useful computation occurs when external proof demand exists. We instantiate with Width-24 Poseidon2 over M31 ($p = 2^{31}-1$): three PoW tickets per permutation, per-permutation usefulness $U = 100\%$ during STARK proving. However, the system-level time-averaged usefulness $U_{sys} = f_{sym} \times U$ depends on SRAM bandwidth: $f_{sym} \approx 10\%$ (SRAM) to $\sim 98\%$ (HBM3E, compute-saturated), with remaining cycles filling Pure PoW at $U = 0\%$ (§5.4). When ZK proof demand is absent, $U_{sys} = 0\%$ and the ASIC operates as a conventional PoW miner (§7). Projected near-zero throughput switching overhead between modes. **Scope of claims:** All security claims (PRP assumption, progress-freedom, collision resistance) assume final Poseidon2 production round constants. GPU throughput measurements (Appendix C) are valid under any constants; they should not be interpreted as security validation. The current Stwo implementation uses placeholder values.

---

## 1. Introduction

PoW blockchains waste energy on security-only computation (§1.1), yet replacing it with useful work requires preserving memoryless block discovery (§1.2)—a property fundamentally at odds with stateful computation. Prior PoUW approaches make different design tradeoffs in this direction (§1.3). ZK-SPoW inverts the relationship, extracting memoryless PoW from useful STARK computation at the permutation level (§1.4).

### 1.1 The PoW Energy Problem

Traditional PoW schemes produce no output beyond network security. The entire energy expenditure is justified solely by the security guarantees it provides.

We develop ZK-SPoW as a general framework and instantiate it for Kaspa [6], a PoW blockchain targeting 100 BPS via the DAGKnight protocol [5]. Kaspa currently uses kHeavyHash for PoW and is evaluating StarkWare's Stwo [4]—a high-performance STARK prover native to Poseidon2 over M31—as a backend for verifiable programs. This combination makes Kaspa a natural first candidate: if the PoW hash is also Poseidon2, the same ASIC can serve both mining and STARK proving. Since Kaspa mining is already ASIC-dominated (kHeavyHash ASICs), §5 and Appendix A examine the implications of replacing the kHeavyHash pipeline with Poseidon2 at the ASIC level—sharing a single permutation unit between PoW and ZK workloads. At 100 BPS the block interval is 10 ms—the most demanding condition for header staleness and progress-freedom—making it a useful upper bound for design analysis.

### 1.2 The Memoryless Requirement in Proof of Work

A PoW scheme is *progress-free* (or *memoryless*) if the probability of finding a valid block on any trial is independent of all previous trials. This property is fundamental:

1. **Fairness.** Progress-based mining gives miners with more accumulated work a higher instantaneous success probability, undermining proportional-hashrate fairness.
2. **Poisson block arrival.** Independent Bernoulli trials yield a Poisson block arrival process—a prerequisite for Nakamoto consensus security proofs.
3. **Difficulty adjustment.** Memoryless trials give expected block time $1/(N \cdot H_{rate} \cdot q)$. Progress introduces state-dependent variance that breaks difficulty estimation.

SHA-256 (Bitcoin) and kHeavyHash (Kaspa) are memoryless by construction: each hash evaluation is independent. The challenge: useful computation (STARK proving, optimization) is inherently stateful and progressive.

### 1.3 Prior PoUW Approaches and ZK-SPoW's Direction

Komargodski & Weinstein [8] provide the first non-trivial formalization of **Proof of Useful Work (PoUW)** and construct a protocol for arbitrary matrix multiplication with $1+o(1)$ multiplicative overhead, where miners select their own inputs (e.g., AI training workloads), building on the foundational study by Ball et al. [1]. Ofelimos [7] achieves provably secure PoUW for combinatorial optimization using SNARGs. Bar-On et al. [9] analyze equilibrium dynamics when miners receive external economic rewards for useful computation.

These works operate in the direction **PoW → useful output**: the mining computation is designed so that its result is simultaneously useful. [8] achieves near-optimal efficiency ($1+o(1)$ overhead) with protocol-enforced usefulness and domain-specific verification. The design prioritizes guaranteed useful output over memorylessness—matrix multiplication is inherently progressive (see §8.3 for detailed comparison).

**ZK-SPoW takes the reverse direction** and prioritizes different properties. Instead of making PoW results useful, we start from useful computation (STARK proof generation) and observe that PoW tickets emerge as a mathematical byproduct. By operating at the granularity of a single Poseidon2 permutation (nanoseconds), each trial is independent with no sunk cost—memorylessness is preserved. Verification remains a standard `hash < target` check. The tradeoff: usefulness is market-driven rather than protocol-enforced:

> **Conventional PoUW:** PoW computation → make results useful → protocol-enforced usefulness, domain-specific verification [1, 7, 8, 9]
>
> **ZK-SPoW:** Useful ZK computation → PoW tickets as mathematical byproduct → memorylessness, standard verification, market-driven usefulness (§8.3)

More precisely: each Poseidon2 Merkle hash in the STARK prover simultaneously computes a Merkle parent (advancing the ZK proof) and produces PoW tickets (checked against the difficulty target). This dual output is a mathematical consequence of reading the same permutation output for two purposes—not a hardware trick or a protocol mandate. The STARK does not *require* PoW, and the PoW does not *require* the STARK; they coexist because the permutation output serves both roles.

This differs from the PoUW constructions in [1, 7, 8]: **blocks contain only the PoW hash, not the STARK proof** (§4.4). The protocol does not enforce useful computation—usefulness depends on external ZK proof demand (§7). In [8], miners supply their own inputs; in ZK-SPoW, proof demand must come from the market (§8.3). However, the operating mode *is* distinguishable—via nonce format conventions and mempool STARK proof correlation (§4.4)—if the protocol chooses to verify it.

**Definition (ZK-SPoW).** A PoW scheme where the hash function is a width-extended Poseidon2 compression function operating on STARK Merkle data, such that every permutation simultaneously advances a ZK proof and produces PoW tickets.

The mechanism: STARK proof generation requires millions of Merkle hashes. Each Width-24 Poseidon2 Merkle hash takes (left child, right child, header digest) as input and produces three PoW tickets as output. Ticket 0 equals the Merkle parent and simultaneously advances the ZK proof. All three output regions are checked against the difficulty target. The miner cannot choose the Merkle inputs—they are determined by the STARK computation.

**Definition (Usefulness).**

$$U = \frac{\text{ZK-contributing trials}}{\text{total mining trials}}$$

- **Continuous proving**: Every Poseidon2 permutation advances a ZK proof → $U = 100\%$. ZK work remains useful regardless of PoW outcome.
- **With idle gaps**: Pure PoW fallback trials (no pending ZK work) are wasted → $U$ decreases proportionally.

The per-permutation data overhead is 8/24 ≈ 33% (header digest occupying 8 of 24 state elements); this is the cost of PoW integration, not a usefulness loss.

**System-level usefulness.** $U$ is a per-permutation metric. On ASIC, the Poseidon2 pipeline is compute-bound while STARK Merkle hashing is SRAM-bandwidth-limited, so only a fraction $f_{sym}$ of Poseidon2 cycles execute in Symbiotic mode. The system-level time-averaged usefulness is $U_{sys} = f_{sym} \times U$, projected to range from ~10% (SRAM) to ~98% (HBM3E, compute-saturated) depending on memory configuration (§5.4, Appendix A). With pipelined proof generation, NTT and Merkle phases of different proofs overlap, and the constraint is memory bandwidth contention rather than phase sequencing.

### 1.4 Our Solution: Permutation-Level PoW Extraction

The central insight of ZK-SPoW is that **non-memoryless computation can produce memoryless PoW**, provided the PoW granularity is finer than the computation's internal state.

A STARK proof is non-memoryless: it consists of trace generation, polynomial commitment (Merkle trees), Fiat-Shamir challenges, and FRI folding—each phase depends on the previous one. A miner who has completed 80% of a proof has "progressed" relative to one who has completed 20%.

However, ZK-SPoW does not use proof completion as the PoW event. Instead, **each individual Poseidon2 permutation within the STARK is an independent PoW trial.** Under the PRP assumption on Poseidon2:

- Each permutation output is pseudorandom, regardless of input structure
- The Merkle tree's feedback structure (parent outputs become child inputs at the next level) does not create exploitable correlation
- Each PoW trial takes nanoseconds (one permutation), not tens of milliseconds to seconds (one proof)

**Result:** ZK-SPoW is computationally progress-free (§2.1) despite embedding PoW within a stateful computation. Block discovery is computationally indistinguishable from a Poisson process under the PRP assumption—satisfying the prerequisite that Nakamoto consensus security proofs require [5, 16].

This addresses the challenge of combining useful computation with memoryless PoW—not by making the computation memoryless, but by extracting PoW at a granularity where the PRP assumption guarantees independence.

### 1.5 Contributions

1. **Memoryless PoW from non-memoryless computation.** We formalize *computational* progress-freedom for PoW schemes embedded in stateful computations and prove that ZK-SPoW achieves it under the PRP assumption on Poseidon2 (Theorem 1, §2). Header staleness is bounded to one Merkle commitment phase (~3 ms GPU (measured)).

2. **Width-24 Poseidon2 in compression function mode.** We specify Width-24 Poseidon2 over M31 (1 permutation per Merkle hash, vs 2 in sponge mode) with $R_p = 22$ internal rounds for 128-bit security (§6.2). The 24-element output naturally partitions into a Merkle parent (advancing the ZK proof) and two additional PoW tickets—the dual-purpose output that makes ZK-SPoW possible (§4.2–§4.3).

3. **GPU empirical validation.** We implement a complete Width-24 Poseidon2 Circle STARK prover on GPU and validate: (a) STARK Merkle hashes produce PoW tickets as a byproduct (Appendix C.1); (b) no meaningful throughput difference between random nonce and structured Merkle inputs ($99.3\% \pm 0.3\%$; Appendix C.2).

---

## 2. Progress-Freedom and Header Staleness

### 2.1 Computational Progress-Freedom

A PoW scheme is *progress-free* if each trial's success probability is independent of all prior outcomes—i.e., the success events form an i.i.d. Bernoulli($q$) sequence, yielding Poisson block arrivals [16]. SHA-256 and kHeavyHash achieve this information-theoretically under the random oracle model.

Poseidon2 is not a random oracle, so we use a computational relaxation: a PoW scheme is *computationally progress-free* if no PPT adversary can distinguish the trial outcomes from i.i.d. Bernoulli($q$) with non-negligible advantage. This suffices for Nakamoto consensus security: the adversary in these proofs is PPT [16, 5], so computational indistinguishability from i.i.d. preserves all security bounds.

**Why proof-level PoW fails.** If proof completion is the PoW event, the scheme is not progress-free at any level of assumption: a miner at 80% completion has observably higher conditional success probability than one at 10%.

### 2.2 Permutation-Level Independence

ZK-SPoW defines the PoW trial at the individual Poseidon2 permutation, not the proof. Each evaluation $\pi(x_j)$ on a Merkle node input produces three PoW tickets from the 24-element output (§4.3). The core argument is straightforward:

**Theorem 1 (Computational Progress-Freedom).** *Under the PRP assumption on Width-24 Poseidon2, the PoW success events $E_1, \ldots, E_P$ across all Merkle tree evaluations are computationally indistinguishable from i.i.d. Bernoulli($q$), with $q = 1 - (1-T/p^8)^3$.*

*Proof sketch.* Merkle tree inputs are pairwise distinct except with negligible probability ($< 10^{-60}$ for $P \leq 10^7$, by union bound on PRP output collisions). For distinct inputs, the PRP assumption guarantees that $(\pi(x_1), \ldots, \pi(x_P))$ is computationally indistinguishable from independent uniform outputs. Each ticket comparison $\text{ticket} < T$ is a deterministic function of the output, so the derived events inherit independence. The PRP–PRF switching cost is $P^2/(2p^{24}) < 10^{-210}$, negligible. ∎

The full proof (real–ideal reduction) and details on input distinctness, structured inputs, side information, and malicious input injection are in Appendix D.

**Remark (Malicious inputs).** A miner who feeds repeated or arbitrary inputs to Poseidon2 receives the same deterministic output for each repeated input—strictly reducing their ticket count. Input distinctness is a rational behavior property, not a protocol requirement.

### 2.3 Header Staleness

In Symbiotic mode, the header digest $h_H$ is fixed for one Merkle commitment phase. Traditional PoW has zero staleness (each hash references the latest header); ZK-SPoW introduces staleness $\Delta_{stale} \leq N_{max}/R_{perm}$, where $N_{max}$ is the largest Merkle tree size.

**Concrete bound (GPU, measured):** $\Delta_{stale} \approx 3.4$ ms at $\ell = 20$, 305 Mperm/s. This is the worst case—a new header arriving at the start of a Merkle phase forces the entire phase to complete with the stale header. On average, headers change mid-phase, so typical staleness is ~$\Delta_{stale}/2$.

**Staleness relative to propagation delay.** At 100 BPS, the block interval is 10 ms, but global P2P networks exhibit propagation delays of 50–200 ms under normal conditions (and significantly longer during network congestion or partitions). In a BlockDAG [5], the relevant baseline is propagation delay, not block interval—blocks created in parallel are all included in the DAG regardless. Worst-case GPU staleness of 3.4 ms is **1.7–6.8%** of the best-case propagation delay (3.4/200 – 3.4/50); under realistic network conditions the ratio is smaller still.

**Selfish mining.** An attacker announcing blocks at phase boundaries can waste victims' partial STARK computation, but gains no PoW advantage—stale tickets remain valid. The cost is at most one phase of ZK throughput (~3.4 ms × $f_{sym}$), not PoW security. Precision timing is impractical over 50–200 ms network jitter.

**Trace size tradeoff.** Larger traces increase $\Delta_{stale}$: at $\ell = 22$, $\Delta_{stale} \approx 13.6$ ms (GPU), exceeding one block interval. The protocol should constrain $\ell$ such that $N_{max}/R_{perm}$ remains below the block interval.

---

## 3. Notation and Definitions

| Symbol | Definition |
|---|---|
| $\mathbb{F}_p$ | Finite field, $p = 2^{31}-1$ (Mersenne prime M31) |
| $\text{Poseidon2}_\pi$ | Poseidon2 permutation over $\mathbb{F}_p^t$ |
| $t$ | State width (number of field elements in permutation) |
| $H$ | Block header (all consensus fields; see §4.4) |
| $h_H$ | Header digest: $\text{PoseidonSponge}(H \text{ excluding nonce}) \in \mathbb{F}_p^k$ |
| $k$ | Header digest element count ($k = 8$ for symmetric I/O and three PoW tickets) |
| $(v_1, v_2)$ | Nonce: $v_1, v_2 \in \mathbb{F}_p^8$ |
| $T$ | Difficulty target (integer; 8 M31 elements interpreted as 248-bit big-endian value) |
| $p_t$ | Single-ticket success probability: $p_t = T/p^8$ (see Remark below) |
| $q$ | Per-permutation success probability: $q = 1 - (1-p_t)^3$ |
| $S$ | Poseidon2 state after permutation, $S \in \mathbb{F}_p^t$ |
| $N$ | Number of miners in the network |
| $N_i$ | Number of leaves in Merkle commitment phase $i$ |
| $N_{spow}$ | Number of ZK-SPoW miners (subset of $N$) |
| $m$ | Number of FRI folding rounds |
| $P$ | Total Poseidon2 permutations across all STARK phases: $P = \sum_{i=0}^{m}(N_i - 1)$ |
| $\ell$ | Trace size exponent (trace has $2^\ell$ rows) |
| $U$ | Per-permutation usefulness: ZK-contributing trials / total mining trials (§1.3) |
| $U_{sys}$ | System-level time-averaged usefulness: $U_{sys} = f_{sym} \times U$ |
| $d$ | S-box exponent ($x \mapsto x^d$; $d = 5$ for Poseidon2) |
| $f_{sym}$ | Fraction of Poseidon2 cycles executing STARK Merkle hashes |
| $\mathcal{H}$ | Total PoW hashrate (network-wide) |
| $R_{perm}$ | Permutation throughput (permutations per second) |
| $K$ | Number of distinct traces in multi-trial grinding analysis (Appendix B.5) |
| $B$ | Block reward |
| $Z$ | ZK proof throughput (proofs per second per miner) |
| $F$ | Fee per ZK proof |
| $D$ | Network-wide ZK proof demand (proofs per second) |
| $\alpha$ | Pure PoW efficiency advantage over ZK-SPoW ASIC ($\alpha \approx 1.1$–$1.2$) |

**Stwo baseline parameters** (confirmed from source code [4]):

| Parameter | Value |
|---|---|
| Field | M31, $p = 2^{31}-1$ |
| Hash output | 8 elements = 248 bits |
| Standard width | $t_0 = 16$ (sponge mode: rate 8, capacity 8) |
| External rounds $R_f$ | 8 (4 + 4) |
| Internal rounds $R_p$ | 14 |
| S-box exponent | $d = 5$ |
| Merkle hash | 2 permutations per node (sponge: absorb left[8], absorb right[8]) |
| Commitment hash | Blake2s (base layer), Poseidon over Felt252 (recursive proofs) |

**Note:** All security claims in this paper assume final production round constants. As of writing, production round constants for Poseidon2 over M31 have not been finalized; the Stwo implementation uses development placeholders (`EXTERNAL_ROUND_CONSTS` and `INTERNAL_ROUND_CONSTS` uniformly set to `1234`).

**Remark (Probability space).** Each ticket is an element of $\mathbb{F}_p^8$, which has $p^8$ elements—not $2^{248}$. Ticket comparison and the difficulty target $T$ operate in the natural ordering of $\mathbb{F}_p^8$ (lexicographic, with each element in $\{0, \ldots, p-1\}$). The success probability is therefore $p_t = T/p^8$. The approximation $p^8 \approx 2^{248}$ introduces a relative error of $1 - (p/2^{31})^8 = 1 - ((2^{31}-1)/2^{31})^8 \approx 8 \cdot 2^{-31} \approx 3.7 \times 10^{-9}$, which is negligible for all practical difficulty targets. Throughout this paper we use the exact definition $p_t = T/p^8$; the notation "$248$-bit" refers to the bit-length of the representation, not the exact cardinality of the probability space.

---

## 4. Protocol Specification

### 4.1 PoW Function Replacement

**Current (kHeavyHash [6]):**
```
pre_pow_hash = Blake2b(H with nonce=0, timestamp=0)
inner        = cSHAKE256_PoW(pre_pow_hash || timestamp || [0u8;32] || nonce)
pow_hash     = cSHAKE256_Heavy(M * inner XOR inner)
valid iff pow_hash < target
```
where $M$ is a 64×64 full-rank matrix over 4-bit nibbles (generated from `pre_pow_hash` via XoShiRo256++), nonce is 8 bytes (`u64`).

**Proposed (Poseidon2-PoW):**
```
h_H = PoseidonSponge(H excluding nonce)       // amortized pre-hash (8 M31 elements)
S   = Poseidon2_pi(v1 || v2 || h_H)           // single permutation, width 24
pow_hash0 = (S[0], S[1], ..., S[7])            // 8 M31 elements = 248 bits
pow_hash1 = (S[8], S[9], ..., S[15])           // 8 M31 elements = 248 bits
pow_hash2 = (S[16], S[17], ..., S[23])         // 8 M31 elements = 248 bits
valid iff pow_hash0 < target OR pow_hash1 < target OR pow_hash2 < target
```
where $(v_1, v_2)$ are 8 M31 elements each (64 bytes total nonce). The permutation operates in **compression function mode**—all 24 input elements are visible (no hidden capacity). Each permutation produces **three PoW tickets**. The comparison interprets both hash and target as 248-bit unsigned integers via big-endian concatenation of 8 M31 elements, each zero-padded to 31 bits. In Symbiotic mode, pow_hash0 equals the Merkle parent: the same output advances the ZK proof and serves as a PoW ticket.

**Verification cost:** One Poseidon2 permutation (width 24) + three target comparisons + one header pre-hash (amortized).

**Standard PoW structure.** ZK-SPoW is conventional hash-based PoW: miners explore a nonce space by computing Poseidon2 permutations and comparing outputs against a difficulty target, identically to Nakamoto-style PoW. The ZK component (Symbiotic mode, §5.1) is an optional revenue source sharing the same hardware—it does not modify the PoW function, security model, or difficulty adjustment.

### 4.2 Poseidon2 Width Extension

The core design change is extending the Poseidon2 permutation width to accommodate header digest as an additional input, and switching from sponge mode to compression function mode.

**Stwo Baseline.** Stwo's Poseidon2 operates in sponge mode with width 16:
```
Width t0 = 16
Rate  r  = 8    <- absorbs 8 elements per permutation
Capacity c = 8  <- hidden elements (security)
```
For Merkle tree commitments, each node hash requires two sponge absorptions (2 permutations per Merkle hash).

**Design Alternatives.** Three approaches to integrate header digest into the Merkle hash:

| Design | Width | Mode | Perm/Merkle | Perm/PoW |
|---|---|---|---|---|
| A: Header re-hash | 16 | Sponge | 2 | 4 |
| B: Width 20 | 20 | Compression | 1 | 1 |
| **C: Width 24** | **24** | **Compression** | **1** | **1** |

**Design A** requires no Stwo modification but costs 4 permutations per PoW draw (2 Merkle + 2 header binding). **Design B** provides only 4 M31 elements (124 bits) for header digest—birthday bound $2^{62}$, insufficient for 128-bit security. **Design C (selected)** achieves symmetric 8+8+8 I/O with 1 permutation per Merkle hash and per PoW draw, 248-bit header digest (birthday bound $2^{124}$), and three PoW tickets per permutation. Width 24 is within the Poseidon2 paper's analyzed parameter range [3].

**Proposed Extension (Design C):**
```
Standard Stwo:    Width t0 = 16,  Sponge (rate 8, capacity 8)
Proposed ZK-SPoW: Width t  = 24,  Compression function (all 24 visible)
```

| Component | Width 16 (standard) | Width 24 (proposed) | Change |
|---|---|---|---|
| External MDS | 4 M4 blocks | 6 M4 blocks | +50% additions |
| Internal MDS | 16 multiplications | 24 multiplications | +50% mult. |
| S-box (external) | 16 per round | 24 per round | +50% |
| S-box (internal) | 1 per round | 1 per round | unchanged |
| Internal rounds $R_p$ | 14 | 22 | +57% |
| Total rounds | 22 (8+14) | 30 (8+22) | +36% |
| S-box operations | 142 | 214 | +51% |
| State registers | 16 × 31 = 496 bits | 24 × 31 = 744 bits | +50% |

Width-24 requires $R_p = 22$ internal rounds vs Width-16's $R_p = 14$ (§6.2), adding +36% pipeline depth. Per-permutation data overhead: 8/24 ≈ 33% of state carries header digest rather than ZK data. ASIC area implications are analyzed in Appendix A.

**Compression Function vs Sponge.**

| Property | Sponge (Stwo standard) | Compression (ZK-SPoW) |
|---|---|---|
| Hidden state | 8 capacity elements | None (all visible) |
| Security model | Indifferentiability | Collision/preimage of π |
| Perm. per Merkle hash | 2 | **1** |
| Width | 16 | 24 |
| PoW tickets per hash | 0 | **3** |

Both modes are established Poseidon2 usage modes [3]. The Poseidon2 paper recommends compression function mode for Merkle trees.

### 4.3 I/O Mapping

For a single Poseidon2 permutation with state $S \in \mathbb{F}_p^{24}$ (compression function mode):

```
INPUT (24 = 8+8+8, all visible):
  S[0..7]    <- left_child      8 M31 elements (248 bits)
  S[8..15]   <- right_child     8 M31 elements (248 bits)
  S[16..23]  <- header_digest   8 M31 elements (248 bits)

          [ Poseidon2 permutation (t = 24) ]
          [ R_f = 8 external + R_p = 22 int ]

OUTPUT (24 = 8+8+8, all visible):
  S[0..7]    -> pow_ticket0     merkle_parent (STARK) AND PoW ticket 0 (248 bits)
  S[8..15]   -> pow_ticket1     PoW ticket 1 (248 bits)
  S[16..23]  -> pow_ticket2     PoW ticket 2 (248 bits)
```

| Mode | S[0..7] in | S[8..15] in | S[16..23] in | S[0..7] out | S[8..15] out | S[16..23] out |
|---|---|---|---|---|---|---|
| Symbiotic | left child | right child | $h_H$ | merkle parent + ticket₀ | PoW ticket₁ | PoW ticket₂ |
| Pure PoW | $v_1$ | $v_2$ | $h_H$ | PoW ticket₀ | PoW ticket₁ | PoW ticket₂ |

The same Poseidon2 hardware computes both modes. Only the input source for S[0..15] differs. **Note on full diffusion:** The Poseidon2 permutation mixes all 24 state elements through its MDS matrix every round. All output elements are functions of all input elements—the 8+8+8 labeling is a convention for reading the output, not a property of the permutation.

**Note on ticket granularity: 3 tickets vs 1 ticket.** The 3 × 8-element partition is a protocol convention, not a cryptographic necessity. An alternative design reads all 24 output elements as a single value in $\mathbb{F}_p^{24}$ and adjusts the difficulty target accordingly ($q_{single} = T_{24}/p^{24}$ vs $q_{triple} = 1-(1-T_8/p^8)^3 \approx 3 T_8/p^8$). Both achieve equivalent expected block times via difficulty adjustment. We choose 3 × 248-bit tickets because: (1) **Stwo compatibility:** 8-element output matches the Stwo hash convention, enabling direct reuse of Merkle commitment infrastructure; (2) **Symbiotic mode constraint:** in Symbiotic mode, S[0..7] *must* be the Merkle parent (used by the STARK), so it has a designated role—the 3-ticket design naturally partitions the output into "STARK-functional" and "PoW-only" regions. A single-ticket design with 744-bit comparison achieves equivalent expected block times via difficulty adjustment; Proposition 3 (§6.4) confirms that the three tickets are computationally independent under PRP, validating the $q = 1-(1-p_t)^3$ formula.

### 4.4 Block Structure

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

The only structural change is the nonce expansion from `u64` (8 bytes) to `[F_p; 16]` (64 bytes, +56 bytes per block). At 100 BPS this adds 5.6 KB/s to network bandwidth—negligible relative to the ~12.5 MB/s block-body baseline (125 KB max body per KIP-13 × 100 BPS; total network traffic is higher due to headers and DAG parent references). The +56 bytes per header add <1 μs to block propagation delay at 1 Gbps, with negligible expected impact on orphan rates.

**Block hash vs PoW hash.** Block identity and PoW use separate hash functions. The block hash (e.g., Blake2b-256 for DAG references) is unchanged; only the PoW function is replaced.

**STARK proofs are NOT included in the block.** They are submitted as independent transactions in the mempool, providing economic value to the ZK ecosystem (vProgs fees). This eliminates the +3–5 MB/s bandwidth overhead that would result from mandatory per-block STARK proofs at 100 BPS.

**Mode distinguishability (optional, not consensus-critical).** Although blocks do not contain STARK proofs, the operating mode (Symbiotic vs Pure PoW) is distinguishable through two independent mechanisms if the protocol chooses to use them. Neither is required for consensus validity—PoW security holds regardless of mode:

1. **Nonce format convention.** The protocol can define that Pure PoW nonces must zero-pad a designated region (e.g., `v2[4..7] = 0`). In Symbiotic mode, the corresponding nonce positions contain Merkle child data—Poseidon2 outputs that are zero with probability $1/p^4 \approx 2^{-124}$. A verifier inspects the nonce field in the block header: zeros in the designated region indicate Pure PoW; non-zeros indicate a Symbiotic claim. The cost is a reduction in Pure PoW nonce space from 16 to 12 elements (372 bits)—no practical impact on hashrate.

2. **Mempool proof correlation.** Each STARK proof submitted to the mempool commits to Merkle trees constructed with a specific header digest $h_H$ as a salt in every node (§4.2). The verifier of the STARK proof requires $h_H$ to recompute Merkle opening paths. By matching a proof's $h_H$ against a block header, any node can verify *after the fact* that a given block was mined during Symbiotic computation for that specific header. This requires no protocol changes—the binding evidence is inherent in the STARK proof structure.

Neither mechanism is mandatory for consensus. The current design treats mode distinguishability as optional: the PoW function is valid regardless of mode, and ZK revenue provides the economic incentive for Symbiotic mining. However, if future protocol evolution requires rewarding or mandating useful work, these mechanisms provide a path to verifiable mode identification without modifying the PoW function itself.

**Header digest collision resistance.** The 248-bit header digest ($k = 8$ M31 elements) provides a birthday bound of $2^{124}$—a factor of $2^4$ below the $2^{128}$ standard for 256-bit hashes. If this 4-bit gap is deemed insufficient, the ticket readout can be changed to full-state comparison (744 bits, birthday bound $2^{372}$) without modifying the permutation.

---

## 5. Operating Modes

The following analysis assumes a hypothetical Poseidon2 ASIC; architecture estimates are in Appendix A.

### 5.1 Symbiotic Mode (Stwo Prover Active)

When ZK proof demand exists, the ASIC runs the Stwo prover. The STARK proof generation pipeline:
```
1. Trace generation       -> circuit evaluation
2. NTT (LDE)              -> polynomial domain extension
3. Merkle tree            -> Poseidon2 hashing (PoW tickets appear here)
4. Fiat-Shamir challenge  -> derived from Merkle root
5. FRI rounds             -> folding + commitment
```

At step 3, every Merkle hash is:
```
S = Poseidon2_pi(n_L || n_R || h_H)   // width 24, compression function
merkle_parent = S[0..7]     <- advances STARK proof AND checked as PoW ticket 0
pow_ticket1   = S[8..15]    <- checked against PoW target
pow_ticket2   = S[16..23]   <- checked against PoW target

if S[0..7] < target OR S[8..15] < target OR S[16..23] < target -> BLOCK FOUND
```

Every permutation simultaneously advances the ZK proof and produces PoW tickets ($U = 100\%$). GPU validation confirms up to 136.39M PoW tickets/s as a byproduct of STARK proof generation (Appendix C.1).

**Header freshness.** A STARK proof spans multiple Merkle commitment phases—typically $O(10)$. The header digest is fixed per phase; between phases it updates to the current chain tip. Maximum staleness is one Merkle phase (§2.3).

### 5.2 Pure PoW Mode (No ZK Demand)

When no ZK proofs are requested:
```
loop:
  v1 = next_nonce_1()
  v2 = next_nonce_2()
  S = Poseidon2_pi(v1 || v2 || h_H)   // width 24, compression function
  if S[0..7] < target OR S[8..15] < target OR S[16..23] < target -> BLOCK FOUND
```

Identical Poseidon2 pipeline, identical throughput. $U = 0\%$—no ZK proof is being computed. The ASIC provides network security only.

### 5.3 Linear Mode Transition

```
+-- Poseidon2 Pipeline (always 100% utilized) ---------+
|                                                       |
|  Input MUX (per-cycle decision):                      |
|    SRAM data ready  -> STARK Merkle hash (Symbiotic)  |
|    SRAM not ready   -> PoW nonce hash    (PoW)        |
|                                                       |
|  Switching cost: near-zero throughput overhead (combinational MUX) |
|  Hashrate: invariant across all modes                 |
+-------------------------------------------------------+
```

The transition is **per-cycle and linear**, not a discrete mode switch. The pipeline is always full—the ratio is determined by SRAM bandwidth. Total PoW hashrate $\mathcal{H} = N_{\text{cores}} \times R_{\text{core}}$ is therefore invariant across operating modes (GPU validation: Merkle/Random throughput ratio $99.3\% \pm 0.3\%$; Appendix C.2).

### 5.4 Complementary Bottleneck Structure

**STARK data path.** The STARK pipeline alternates between NTT-bound and Poseidon2-bound phases:

```
Phase:    Trace → NTT(LDE) → Merkle → Fiat-Shamir → [FRI fold(NTT) → FRI Merkle] × m
HW:       CPU/NTT   NTT unit   Poseidon2   CPU         NTT unit        Poseidon2
SRAM:     write     read/write read/write   —           read/write      read/write
Poseidon2: Pure PoW  Pure PoW  Symbiotic+PoW Pure PoW   Pure PoW       Symbiotic+PoW
```

**Single-proof model.** Within a single proof, phases are sequential: during Merkle phases, Poseidon2 cores are fed Merkle data from memory (Symbiotic) up to the bandwidth limit; during NTT/FRI folding phases, the NTT unit uses memory and Poseidon2 runs 100% Pure PoW.

**Pipelined proof generation.** With continuous proof demand, multiple proofs are in-flight simultaneously. While proof $n$ is in its NTT phase, proof $n-1$'s Merkle data is already in memory and can feed Poseidon2 cores. The NTT and Merkle phases of different proofs overlap:

```
Proof n-1:        [NTT] → [Merkle] → [NTT] → ...
Proof n  :                 [NTT] → [Merkle] → ...
Poseidon2:         PoW      Sym       Sym      Sym
```

In this model, Poseidon2 is nearly always fed Merkle data. The constraint is no longer phase sequencing but **memory bandwidth contention**: NTT and Merkle I/O share the memory bus simultaneously.

**$f_{sym}$ derivation (pipelined).** Each Merkle hash requires 96 bytes of memory I/O (32 read left + 32 read right + 32 write parent). NTT butterflies require ~20 bytes each (read 2 + twiddle, write 2 M31 elements) at ~1 Gbfly/s, consuming $BW_{ntt} \approx 20$ GB/s. The effective bandwidth available for Merkle I/O is $BW - BW_{ntt}$:

$$f_{sym} = \min\!\left(1 - \epsilon,\; \frac{(BW - BW_{ntt})/96}{R_{perm}}\right)$$

where $\epsilon \approx 0.02$ accounts for proof setup, FRI round transitions, and pipeline drain between proofs. See Appendix A.3 for concrete values.

**Width-24 efficiency.** Compression function mode halves the permutation count per Merkle hash (2→1 vs sponge mode, §4.2), but each Width-24 permutation is larger (30 rounds vs 22). Net cycle reduction is ~32% (30 vs 44 rounds), not 50%.

---

## 6. Security Analysis

### 6.1 Poseidon2 Cryptographic Properties

**Sponge mode vs compression function mode.** Poseidon2 [3] provides 128-bit security in sponge mode via an indifferentiability argument: with capacity $c = 8$ M31 elements (248 bits), the sponge construction is indifferentiable from a random oracle up to $2^{124}$ queries. ZK-SPoW operates in compression function mode (§4.2), where all 24 input elements are visible—there is no hidden capacity. Security in this mode relies on different, more direct properties of the permutation $\pi$:

- **Collision resistance:** Finding $x \neq x'$ with $\pi(x)[0:7] = \pi(x')[0:7]$ requires $\Omega(2^{124})$ work (birthday bound on 248-bit output). This secures STARK Merkle binding.
- **Preimage resistance:** Finding $x$ with $\pi(x)[0:7] < T$ requires $\Omega(1/p_t)$ work. This secures PoW.
- **PRP (pseudorandom permutation):** Outputs are computationally indistinguishable from random, regardless of input structure. This secures progress-freedom (Theorem 1).

These are strictly weaker assumptions than sponge-mode indifferentiability—collision/preimage resistance of a permutation is implied by, but does not require, random oracle behavior. The Poseidon2 literature [2, 3] primarily analyzes the sponge construction, raising a natural question: does the round count $R_p = 22$ assume the existence of hidden capacity elements?

**Round count is capacity-independent.** We verify from Plonky3's source [10] that the round number calculation depends on exactly four parameters: field prime $p$, state width $t$, S-box degree $d$, and security level $M$. Capacity does not appear. The six security constraints (statistical, interpolation, three Gröbner basis variants, and the algebraic bound from [15]) target the *permutation itself*—they bound the number of rounds needed to resist distinguishing attacks against $\pi$, not against a sponge construction built from $\pi$. Capacity $c$ is a separate, orthogonal parameter that determines sponge-mode security as $\min(\text{permutation security}, 2^{c/2})$. For compression function mode, only permutation security applies, and $R_p = 22$ (with margin $R_f += 2$, $R_p \times 1.075$) targets 128-bit permutation security directly.

**Compression function mode risks.** While the round count is sound, operating without capacity has two consequences: (1) the full 744-bit state is visible, giving an attacker more information for algebraic cryptanalysis (e.g., CICO attacks [12] become formulated over the full state rather than a reduced output space); (2) there is no "safety net"—in sponge mode, even a partial permutation break may be absorbed by the capacity, whereas in compression mode any permutation weakness directly impacts security. We believe this risk is manageable because: (a) the 30-round configuration's algebraic properties (S-box degree growth, MDS diffusion) are mode-independent; (b) the Poseidon2 paper [3] explicitly recommends compression function mode for Merkle trees; (c) any attack on the sponge implies an attack on the compression function with equal or lower complexity, since the compression-mode attacker has strictly more information [3]. Conversely, compression-mode-specific attacks (exploiting visible capacity elements) do not necessarily transfer to sponge mode—making compression mode the more conservative security setting.

**Single primitive dependency.**

| Component | Traditional (e.g., Kaspa [6]) | ZK-SPoW |
|---|---|---|
| PoW | kHeavyHash (Blake2b + cSHAKE256) | Poseidon2 |
| STARK | Poseidon2 | Poseidon2 |
| Independence | PoW ≠ STARK | PoW = STARK |

**Correlated failure mode.** A cryptographic break in Poseidon2 compromises both PoW security and STARK validity simultaneously—a qualitatively different risk profile from traditional designs where PoW and STARK use independent primitives. We assess this risk quantitatively:

- **Current margin.** The 30-round configuration ($R_f = 8$, $R_p = 22$) includes a $+2$ external round margin and $\times 1.075$ internal round margin over the minimum required by known attacks. Merz and Rodríguez García [12] improve algebraic collision attacks by $2^{106}$ for one parameter set but conclude the full-round primitive "does not fall short of its claimed security level." Resultant-based attacks [13] solve instances with $\leq 10$ total rounds—far below the 30-round configuration.
- **Margin erosion trajectory.** The attack improvement of [12] targets the sparse structure of $M_I$, a design feature inherent to Poseidon2. Further advances along this line are plausible. If a future attack reduces security below $2^{128}$ for Width-24/30-round, both PoW and STARK would require parameter updates simultaneously.
- **Fallback strategy.** A Poseidon2 break does not require abandoning ZK-SPoW. The framework is parametric: increasing $R_p$ (more internal rounds) restores security at the cost of throughput and die area. In the extreme case, the chain can revert to Pure PoW mode with a different hash function while maintaining block production—Symbiotic mode is an overlay, not a prerequisite for consensus. A hard fork would be required in either case (as it would for any PoW hash function break).

### 6.2 Width-24 Security

**Collision resistance (STARK binding).** With 248-bit output (8 M31 elements), the birthday bound is $2^{124}$, providing 124-bit collision security.

**STARK soundness.** The header digest acts as a fixed salt in each Merkle hash—determined before proof generation, cannot be chosen adaptively. STARK soundness reduces to collision resistance with a fixed third input—a strictly easier assumption.

**PoW security.** Under the PRP assumption, finding an input whose output satisfies ticket $< T$ requires expected $1/p_t$ evaluations—no better strategy than random trials exists. No known algebraic attack on the 30-round permutation (8 external + 22 internal) provides a shortcut.

**$R_p$ for Width-24.** $R_p = 22$ for 128-bit security at $d = 5$ over M31, computed via Plonky3's round number formula [10]. The formula applies six security constraints from [2, 3] (statistical, interpolation, three Gröbner basis variants) plus the algebraic attack bound from [15], then adds margin: $R_f += 2$, $R_p \times 1.075$. Pre-margin optimum: $(R_f, R_p) = (6, 20)$; post-margin: $(8, 22)$. The binding constraints are statistical ($R_{f,1} = 6$, since $M = 128 \leq \lfloor \log_2 p - 2 \rfloor \times (t+1) = 700$) and Gröbner-2 ($R_{f,4} = 6$), tied. **Crucially, the formula depends only on $(p, t, d, M)$—capacity is not a parameter** (see §6.1). Total rounds: 30 (8 + 22). S-box operations: $t \times R_f + R_p = 24 \times 8 + 22 = 214$ per permutation; in compression function mode (§4.2), this yields 25% fewer S-boxes per Merkle hash than Width-16 sponge (214 vs $2 \times 142 = 284$). Supplementary: $M_I^k$ is invertible for $k = 1..48$ (subspace trail resistance). Algebraic degree after 30 rounds exceeds $2^{69}$ ($5^{30} \approx 2^{69.7}$), far above the interpolation threshold of $2^{\min(M,\, n)} = 2^{31}$ for M31; the interpolation constraint is non-binding.

**Recent cryptanalysis.** See §6.1 (Current margin) for detailed analysis of [12] and [13].

### 6.3 Trace Grinding Resistance

Trace selection in Symbiotic mode provides zero advantage for PoW mining under the PRP assumption. We summarize the results here; full proofs are in Appendix B.

Under PRP, three invariants hold:

1. **Ticket count invariance.** Total permutations $P = \sum_{i=0}^{m}(N_i - 1)$ is protocol-determined and trace-independent. The miner cannot increase ticket count by selecting a different trace (Appendix B.2).

2. **Distribution invariance.** The number of valid tickets follows $V \sim \text{Binomial}(P, q)$, where both $P$ and $q = 1 - (1-p_t)^3$ are trace-independent. The *entire distribution* is invariant under trace selection (Appendix B.3).

3. **Grinding is dominated.** Header digest grinding is equivalent to nonce grinding at equal cost (Appendix B.4). Multi-trial grinding ($K$ traces, best-of-$K$) is strictly dominated by $K \times P$ pure PoW permutations (Appendix B.5). Merkle tree feedback does not violate PRP (Appendix B.6).

### 6.4 Triple-Ticket Independence

The three PoW tickets from a single permutation are deterministically linked but computationally independent under PRP.

**Proposition 3.** *Under the PRP assumption, the joint distribution of $\text{ticket}_0 = S[0..7]$, $\text{ticket}_1 = S[8..15]$, and $\text{ticket}_2 = S[16..23]$ is computationally indistinguishable from three mutually independent uniform samples over $\mathbb{F}_p^8$, each with success probability $p_t = T/p^8$. The per-permutation success probability is $q = 1 - (1-p_t)^3$.*

No early-termination optimization exists: evaluating any ticket requires the full permutation, which produces all three simultaneously. Any statistical test detecting inter-ticket correlation can be converted into a PRP distinguisher with equal advantage. Full proof in Appendix B.7.

### 6.5 Quantum Resistance

- Grover's algorithm halves effective hash bits: 248/2 = 124-bit quantum security
- Comparable to SHA-256 under quantum attack (256/2 = 128 bits)
- kHeavyHash: 256/2 = 128-bit quantum security
- **Delta:** −8 bits classical (248 vs 256) / −4 bits quantum (124 vs 128). Both exceed the current NIST minimum of 112 bits (SP 800-131A Rev. 2) and BSI's 120-bit floor (TR-02102-1). The 4-bit shortfall relative to SHA-256 and the proposed NIST 128-bit floor (SP 800-131A Rev. 3 draft, 2030) is a known tradeoff of using the 248-bit M31 Poseidon2 output; if needed, full-state comparison (744-bit) provides ample margin.

---

## 7. Economic Considerations

A ZK-SPoW ASIC includes NTT units and memory infrastructure absent from a Pure PoW design. This die area overhead reduces hash/watt by a factor $\alpha > 1$ (the exact value depends on implementation; see Appendix A for one illustrative design). The economic question is whether ZK revenue compensates for this gap.

**Revenue comparison.** Per-watt revenue:

$$\text{ZK-SPoW} = \frac{\mathcal{H}_{spow}}{\mathcal{H}_{total}} \cdot B + Z \cdot F \quad \text{vs} \quad \text{Pure PoW} = \frac{\mathcal{H}_{pow}}{\mathcal{H}_{total}} \cdot B$$

ZK-SPoW dominates when $Z \cdot F$ exceeds the mining revenue gap from the $\alpha$ disadvantage.

**Equilibrium condition.** Let $\alpha$ be the Pure PoW hash/watt advantage. ZK-SPoW ASICs survive in equilibrium when:

$$Z \cdot F \geq (\alpha - 1) \cdot \frac{\mathcal{H}_{spow}}{\mathcal{H}_{total}} \cdot B$$

Let $D$ be the network-wide ZK proof demand (proofs/s) and $N_{spow}$ the number of ZK-SPoW miners. In steady state, per-miner ZK revenue is $F \cdot D / N_{spow}$. The minimum viable demand is:

$$D_{min} = \frac{N_{spow} \cdot (\alpha - 1) \cdot B}{N \cdot F}$$

**PoW security is unconditional.** Regardless of ZK demand, ZK-SPoW ASICs function as conventional PoW miners with a hash/watt disadvantage. Difficulty adjustment absorbs this. **Network security is not structurally compromised by insufficient ZK demand**; only the economic advantage of ZK-SPoW hardware is affected.

**Demand scarcity risk.** If $D < D_{min}$, Pure PoW ASICs dominate economically. If $D \ll N_{spow} \cdot Z$, most ZK-SPoW cycles execute in Pure PoW mode ($U_{sys} \approx 0\%$), and rational miners may prefer cheaper Pure PoW hardware—an economic (not security) collapse scenario. Whether sustained ZK proof demand at the required scale will materialize is an open question (§9).

---

## 8. Comparison with Prior Work

### 8.1 Proof-Level PoUW

Several proposals use ZK proof generation as useful work within PoW:

**Proof-level ZK PoW [14].** This proposal treats proof completion as a lottery ticket—completing a STARK proof yields one PoW attempt. Two limitations arise. First, proofs take tens of milliseconds to seconds, making the scheme non-memoryless: miners closer to completion have a higher conditional probability of finding a block, violating the progress-free property (§2.1). Second, losing miners must discard their proofs—only the winner's proofs enter the chain. Under difficulty adjustment with $N$ competing miners, useful output is $\sim 1/N$ of total computation, converging toward pure PoW.

Granularity comparison:
- **Proof-level**: 1 trial per tens of ms–seconds → non-memoryless, losing proofs wasted
- **Permutation-level (ZK-SPoW)**: 1 trial per nanoseconds → PRP-memoryless, ZK work survives PoW failure

**Ofelimos [7].** Performs combinatorial optimization (Doubly Parallel Local Search) as useful work in a provably secure PoUW framework, using SNARGs for efficient verification. The useful computation is protocol-mandated, not market-driven. Like ZK-SPoW, Ofelimos addresses the PoUW challenge formally, but operates in the "PoW → useful" direction with domain-specific verification.

**Komargodski & Weinstein [8].** Construct a PoUW protocol for arbitrary matrix multiplication with near-optimal overhead, enabling miners to repurpose GPU computation (e.g., AI training) for consensus. **Bar-On, Komargodski & Weinstein [9].** Analyze equilibrium dynamics when miners receive external economic rewards for useful computation alongside mining revenue. ZK-SPoW differs from [8] in that PoW tickets emerge as a byproduct of STARK hashing rather than through a separate verification protocol, and from [9] in that ZK revenue is a design-level property rather than an external market assumption.

### 8.2 Sequential ZK-then-Hash (Nockchain)

Nockchain [11] is a Layer-1 blockchain using zkPoW, launched May 2025. Miners compute a ZK proof of a deterministic puzzle, then hash the proof; the hash must meet a difficulty target.

| | Nockchain (zkPoW) | ZK-SPoW |
|---|---|---|
| PoW hash | hash(ZK proof) < T | Poseidon2 output < T |
| PoW generation | Two-step: ZK proof → external hash | **Single-step**: STARK Merkle hash = PoW hash |
| Field / output | Goldilocks (64-bit), 320-bit Tip5 hash | M31 (31-bit), 248-bit output |
| Useful work | ZK proof of deterministic puzzle | STARK Merkle hashing (tx verification) |
| STARK enforcement | Mandatory (proof = block) | Market-driven |

The key architectural difference: Nockchain computes the ZK proof first, then hashes it for PoW—two disjoint steps. ZK-SPoW's internal STARK Merkle hash *directly* produces the PoW output from a single permutation.

**Memorylessness nuance.** Nockchain's final hash-after-proof step is itself memoryless: the hash evaluation is independent of previous hashes. However, the proof computation phase preceding it is progressive (a miner closer to proof completion has higher conditional probability of reaching the hash step sooner). The overall block arrival process is therefore a *renewal process* where inter-arrival times are the sum of a progressive proof phase and a memoryless hash phase—not a pure Poisson process. The practical impact depends on proof duration relative to block interval: if proofs take seconds and blocks arrive every few seconds, the progressive component is significant. ZK-SPoW avoids this entirely by operating at permutation granularity (nanoseconds).

### 8.3 Relationship to PoUW [1, 8]

Komargodski & Weinstein [8] construct a PoUW protocol for arbitrary matrix multiplication with $1+o(1)$ overhead, where miners select their own inputs. Their formalization requires efficiency ($1+o(1)$ prover overhead), completeness, and hardness—building on the foundational study by Ball et al. [1]. Both operate in the direction **PoW → useful output**. ZK-SPoW takes the reverse direction: **useful computation → PoW output**. Rather than redesigning PoW to produce useful results, useful ZK computation naturally produces PoW-valid outputs as a byproduct of the same permutation.

**Different design priorities.** The two directions prioritize different properties. [8] achieves protocol-enforced usefulness with $1+o(1)$ efficiency and domain-specific verification—the verifier confirms the matrix result, guaranteeing useful output. The design accepts that matrix multiplication is inherently progressive (stateful), prioritizing guaranteed usefulness over memorylessness.

ZK-SPoW prioritizes memorylessness and verification simplicity. Each Poseidon2 permutation is an independent, nanosecond-scale trial with no sunk cost, and verification is a standard `hash < target` check with no domain-specific logic. The tradeoff: usefulness is market-driven rather than protocol-enforced—miners can fall back to Pure PoW mode if ZK proof demand is absent (§7).

| Criterion | PoW → useful [8] | Useful → PoW (ZK-SPoW) |
|---|---|---|
| Useful output | **Yes**: each PoW trial produces a matrix multiplication result | **Yes**: every Symbiotic permutation advances a STARK proof |
| Efficiency overhead | $1+o(1)$ multiplicative | Near-zero: same permutation serves both roles |
| Usefulness enforced | **Protocol-enforced**: verifier checks matrix result | **Not enforced**: consensus checks only PoW hash; mode distinguishable but optional (§4.4) |
| Verification | Domain-specific (matrix result check) | Standard PoW (`hash < target`) |
| Input selection | Miner-selected (arbitrary matrices) | STARK-determined (miner cannot choose Merkle inputs) |
| Memorylessness | Not addressed (matrix multiplication is stateful) | **Yes**: permutation-level, nanosecond granularity (§5.1) |
| Security basis | Novel conjectures on matrix multiplication hardness | PRP assumption on Poseidon2 (§6) |

**Shared limitation: demand dependency.** Both approaches depend on external demand for the useful computation to materialize. [8] requires demand for matrix multiplication; ZK-SPoW requires demand for STARK proofs (§7). Without such demand, miners in [8] compute arbitrary matrices with no economic value, while ZK-SPoW miners fall back to Pure PoW mode. Neither protocol guarantees useful computation in the absence of a market.

**Mode distinguishability without enforcement.** Blocks do not contain STARK proofs, but the operating mode is *not* indistinguishable. Two mechanisms allow mode identification (§4.4): (1) a nonce format convention that makes Pure PoW blocks identifiable from the header alone, and (2) mempool STARK proofs whose Merkle commitments contain the block's `h_H` as a salt, providing after-the-fact cryptographic evidence of Symbiotic mining. The current design treats these as optional—consensus validity depends only on the PoW hash, and economic incentives (ZK proof revenue) drive Symbiotic adoption. However, these mechanisms mean the protocol *could* enforce or reward useful work in future iterations without modifying the PoW function.

### 8.4 Memorylessness Comparison

| Property | SHA-256 / kHeavyHash | Proof-Level [14] | Nockchain [11] | **ZK-SPoW** |
|---|---|---|---|---|
| PoW granularity | 1 hash = 1 trial | 1 proof = 1 ticket | Proof → hash | **1 perm = 1 trial** |
| Trial duration | Nanoseconds | Tens of ms–s | Seconds + μs | **Nanoseconds** |
| Progress-free? | **Yes** (info-theoretic) | **No** — sunk cost | Final hash: yes; proof: no | **Yes** (computational, PRP) |
| Max staleness | 0 (stateless) | Proof duration | Proof duration | **~3 ms (GPU, measured)** |
| Losing miners' work | Security only | Proofs discarded | Proofs discarded | **ZK work preserved** |
| $U$ (per-perm) | 0% | $\sim 1/N$ | $\sim 1/N$ | **100%** ($U_{sys}$: $f_{sym}$-limited, §5.4) |

ZK-SPoW operates at the finest practical granularity—individual permutations (nanoseconds)—achieving computational progress-freedom analogous to the information-theoretic progress-freedom of traditional hash-based PoW (Theorem 1, §2.2). The cost: the PRP assumption on Poseidon2 replaces the random oracle assumption on traditional hash functions.

---

## 9. Open Questions

1. **Compression function mode: dedicated cryptanalysis for Width-24.** We verify that Plonky3's $R_p = 22$ formula targets permutation-level security independent of capacity (§6.1, §6.2). The cryptanalysis literature covers compression mode ([12] analyzes both sponge and compression modes; [15] targets the permutation generically; [3] defines and recommends compression function mode for Merkle trees). However, no published analysis specifically targets the Width-24 permutation over M31. A dedicated evaluation—CICO attacks over the full 744-bit state and Gröbner basis complexity at this width—would strengthen the 128-bit security claim. Until such analysis, the claim should be considered well-founded but not independently verified for this specific configuration.

2. **Memoryless validation.** Empirical verification of Poisson block inter-arrival times in a test network running ZK-SPoW would complement the theoretical analysis of §2.

3. **Output quality under production constants.** The GPU experiments (Appendix C) use placeholder round constants. Statistical testing of output pseudorandomness (e.g., NIST SP 800-22) under production Poseidon2 constants would validate that the PRP assumption holds in practice, not only in theory.

4. **ZK demand viability.** §7 derives the equilibrium condition and demand scarcity risk. The critical open question is whether sustained ZK proof demand at sufficient scale will materialize. A dynamic simulation of ZK-SPoW vs Pure PoW ASIC competition under stochastic ZK demand—including miner entry/exit dynamics and difficulty adjustment feedback—would quantify the collapse threshold more precisely. The source of demand (who buys proofs, and why ZK-SPoW miners are preferred over dedicated proving services) remains unaddressed.

5. **Consensus-level staleness analysis.** §2.3 bounds staleness as 1.7–6.8% of propagation delay. A DAGKnight consensus simulation—modeling adaptive k-cluster ordering, confirmation times, and throughput under heterogeneous staleness—would provide stronger quantitative guarantees. The selfish mining variant at phase boundaries (§2.3) warrants game-theoretic analysis, though its practical feasibility is limited by network propagation jitter (50–200 ms).

**Conditionally resolved.** Triple-ticket independence (§6.4) and trace grinding resistance (§6.3) are addressed under the PRP assumption.

---

## 10. Conclusion

ZK-SPoW extracts memoryless PoW at the individual Poseidon2 permutation level within STARK proof generation—each evaluation is computationally indistinguishable from an independent Bernoulli trial under the PRP assumption (Theorem 1), satisfying the Poisson block arrival prerequisite of Nakamoto consensus [5, 16]. Each Poseidon2 Merkle hash simultaneously advances a ZK proof and produces PoW tickets—a dual-purpose output from the same permutation, with near-zero throughput switching overhead between modes.

Where prior PoUW constructions [1, 8] operate in the direction PoW → useful output—achieving protocol-enforced usefulness at the cost of stateful computation and domain-specific verification—ZK-SPoW reverses the direction: useful computation → PoW output. Operating at permutation granularity (nanoseconds) rather than proof granularity preserves memorylessness and standard PoW verification, but shifts usefulness from protocol-enforced to market-driven.

The approach has clear limitations. Useful computation is not enforced at the consensus layer—mode distinguishability is achievable (§4.4) but not mandated. System-level usefulness ($U_{sys} \approx 10\text{–}98\%$) is constrained by memory bandwidth (§5.4), and depends entirely on external ZK proof demand—a limitation shared with prior PoUW approaches (§7, §8.3). The single-primitive dependency on Poseidon2 and the absence of Width-24-specific cryptanalysis over M31 remain open risks (§9).

---

## Appendix A: ASIC Architecture Estimates

*All parameters are reference estimates for a hypothetical 7 nm design. No chip has been fabricated.*

### A.1 Die Allocation

| Block | Die share | Function |
|---|---|---|
| Poseidon2 core array | 50% | Width-24 pipelines (M31), per-cycle input MUX (SRAM data / nonce) |
| NTT butterfly unit | 25% | M31 multiply-accumulate: LDE + FRI folding |
| SRAM | 20% | 32 MB on-chip, ~200 GB/s bandwidth |
| Control & I/O | 5% | PoW/STARK schedulers, network interface |

### A.2 Core Sizing

A fully pipelined Width-24 Poseidon2 core (30 rounds: 8 external + 22 internal) requires ~1.4M gates. On a 60M-gate die with 50% allocated to Poseidon2: **21 cores at ~1 Gperm/s each → 63G effective PoW hash/s** (3 tickets per permutation).

M31 multipliers (~1K gates, 31×31 bit) are 3.5× smaller than Goldilocks (~3.5K gates, 64×64 bit), suggesting up to 3.5× more cores per die at equivalent area, before accounting for routing and control overhead.

### A.3 SRAM Bandwidth and $f_{sym}$

Each Merkle hash requires 96 bytes of SRAM I/O (32 read left + 32 read right + 32 write parent). The header digest $h_H$ is fixed per phase and held in registers (no SRAM access). PoW ticket comparison is local (comparator, no SRAM write).

**Within Merkle phases**, the SRAM-limited Merkle throughput is:

$$R_{merkle} = \min\!\left(R_{perm},\; \frac{BW}{96}\right)$$

For 200 GB/s SRAM: $R_{merkle} = 200\text{G}/96 \approx 2.08\text{G hash/s}$, well below $R_{perm} = 21\text{G}$. The remaining $21 - 2.08 = 18.92\text{G}$ perm/s fill with Pure PoW.

**Pipelined proof generation** (§5.4). With continuous proof demand, NTT and Merkle phases of different proofs overlap. The NTT unit consumes $BW_{ntt} \approx 20$ GB/s (~20 bytes/butterfly × 1 Gbfly/s), leaving the remainder for Merkle I/O:

$$f_{sym} = \min\!\left(1 - \epsilon,\; \frac{(BW - BW_{ntt})/96}{R_{perm}}\right)$$

where $\epsilon \approx 0.02$ accounts for proof setup, FRI round transitions, and pipeline drain.

| Memory | Bandwidth | $BW - BW_{ntt}$ | $R_{merkle}$ | $f_{sym}$ |
|---|---|---|---|---|
| SRAM 32 MB | 200 GB/s | 180 GB/s | 1.88G | **~9%** |
| SRAM 64 MB | 400 GB/s | 380 GB/s | 3.96G | **~19%** |
| HBM3 8 GB | 1.2 TB/s | 1.18 TB/s | 12.3G | **~59%** |
| HBM3E 16 GB | 2.4 TB/s | 2.38 TB/s | 21G† | **~98%** |

†Compute-saturated: $R_{merkle}$ capped at $R_{perm} = 21\text{G}$. The ~2% gap from 100% reflects proof transition overhead ($\epsilon$).

**SRAM pipelining constraint.** SRAM configurations (32–64 MB) may lack sufficient capacity to hold two proofs' data simultaneously (each proof requires ~4–8 MB for trace + extended evaluation). In this case, proof pipelining is limited, and the single-proof sequential model applies: $f_{sym}$ depends on both $R_{merkle}/R_{perm}$ and phase sequencing. The values above assume pipelining is feasible; for SRAM-only designs, $f_{sym}$ may be slightly lower (~8–17%). HBM configurations (8–16 GB) can hold hundreds of proofs and pipeline without constraint.

**NoC area and power.** Delivering 200 GB/s of SRAM bandwidth to 21 Poseidon2 cores requires a Network-on-Chip (NoC) or crossbar interconnect. At 7 nm, a 200 GB/s mesh NoC (e.g., ring or 2D mesh topology) consumes approximately 1–3% of die area and 0.5–1.5 W, depending on topology and wire length. The 5% Control & I/O budget (§A.1) is tight: it must accommodate the NoC, PoW/STARK schedulers, difficulty comparison logic, and network interface. A more realistic breakdown may require 7–10% for Control, I/O, and interconnect, reducing Poseidon2 core allocation from 50% to ~47% (20 cores instead of 21). This is a ~5% hashrate reduction—non-negligible but does not qualitatively change the economic analysis (§7). The die allocation in §A.1 should be considered approximate; detailed physical design is beyond this paper's scope.

---

## Appendix B: Trace Grinding and Triple-Ticket Proofs

The following provides complete formal proofs for the claims in §6.3–6.4.

### B.1 Assumptions

1. **PRP.** The Poseidon2 permutation $\pi: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$ is a pseudorandom permutation. For any input $x$, the output $\pi(x)$ is indistinguishable from uniform over $\mathbb{F}_p^{24}$.
2. **Fixed tree sizes.** Each Merkle commitment phase $i$ ($i = 0$ for the initial trace commitment, $i = 1, \ldots, m$ for FRI rounds) has $N_i$ leaves, yielding $N_i - 1$ internal hash evaluations, where $N_i$ is determined by the protocol (trace length, blowup factor, FRI folding rate). The values $N_i$ are independent of trace content.
3. **Fixed header digest.** The header digest $h_H \in \mathbb{F}_p^8$ is fixed at the start of each Merkle commitment phase and constant throughout that phase.

### B.2 Ticket Count Invariance

Each Poseidon2 permutation in the Merkle tree produces three PoW tickets. The total number of permutations across all STARK phases is:

$$P = \sum_{i=0}^{m} (N_i - 1)$$

Since each $N_i$ is a protocol parameter independent of the trace $t$:

$$\forall\, t_1, t_2: \quad P(t_1) = P(t_2) = P$$

The miner cannot increase the number of PoW tickets by selecting a different trace.

### B.3 Distribution Invariance

Under PRP, for any distinct inputs $x_1, \ldots, x_P$ to the permutation, the outputs are jointly pseudorandom. Input distinctness holds with overwhelming probability (Appendix D).

Each permutation produces three PoW tickets. Under PRP, the success event for each permutation (at least one ticket below target $T$) is a Bernoulli trial with parameter:

$$q = 1 - (1-p_t)^3 \approx 3p_t, \quad p_t = T / p^8$$

This parameter depends only on the target $T$, not on the permutation input. Since inter-permutation independence holds (distinct inputs under PRP), the number of successful permutations follows:

$$V \sim \mathrm{Binomial}(P,\, q)$$

Both parameters $(P, q)$ are trace-independent. Therefore, not only the expectation $E[V] = Pq$ but the *entire distribution* of valid tickets is invariant under trace selection. There is no trace for which the variance is smaller (guaranteeing a hit) or larger.

**Fiat-Shamir cascade.** Trace selection affects FRI round Merkle trees via the Fiat-Shamir challenge dependency on the initial Merkle root. Changing the trace changes all subsequent challenges, folding points, and FRI Merkle trees. However, each FRI tree size $N_i$ is protocol-determined, and PRP ensures each resulting ticket has identical success probability. The argument above extends to all commitment phases.

### B.4 Header Digest Grinding

The miner can produce different header digests $h_H$ by choosing different parent blocks or transaction sets. With the same trace but a new $h_H$, the entire Merkle tree must be rebuilt (cost: $P$ permutations), producing $P$ new permutation triples. Under PRP, this is functionally equivalent to $P$ pure PoW nonce hashes—identical cost, identical ticket distribution. Moreover, only one $h_H$ can be used for the STARK proof; the remaining attempts produce PoW tickets but discard the STARK computation. Header digest grinding offers no advantage beyond standard nonce grinding.

### B.5 Multi-Trial Grinding

A miner who computes $K$ distinct traces and selects the one with the most valid PoW tickets:

**Cost:** $K \times (C_{\mathrm{NTT}} + C_{\mathrm{trace}} + P)$ permutations, where $C_{\mathrm{NTT}}$ and $C_{\mathrm{trace}}$ are the NTT and trace generation costs (counted conservatively as zero in the comparison below).

**Benefit:** $\max(V_1, \ldots, V_K)$ where each $V_i \sim \mathrm{Binomial}(P, q)$ independently.

For the same $K \times P$ Poseidon2 permutations in pure PoW mode, all tickets are valid (none discarded), yielding $V_{\mathrm{PoW}} \sim \mathrm{Binomial}(KP, q)$.

By linearity, $E[V_{\mathrm{PoW}}] = KPq$. The best-of-$K$ selection gives $E[\max(V_1, \ldots, V_K)] \leq Pq + O(\sqrt{\log K \cdot Pq(1-q)})$. For any $K \geq 2$:

$$E[\max(V_1, \ldots, V_K)] < KPq = E[V_{\mathrm{PoW}}]$$

Multi-trial grinding is strictly dominated by pure PoW in terms of PoW ticket yield. Including the NTT and trace generation overhead (omitted above) makes the comparison strictly worse for grinding.

**Interaction with ZK proof revenue.** The analysis above considers only PoW utility. A grinder who computes $K$ traces also produces $K$ STARK proofs as a byproduct, each with market value $F$. The effective grinding cost becomes:

$$C_{\mathrm{eff}} = K \times C_{\mathrm{trace}} - (K-1) \times F$$

If $F$ is sufficiently large, the grinder recovers most of the grinding cost from proof sales. However, this does not improve the grinder's *PoW advantage*—the ticket distribution remains $\max(V_1, \ldots, V_K) < V_{\mathrm{PoW}}$. The economic benefit is orthogonal: any miner producing proofs for revenue would produce them sequentially regardless, obtaining the same $K$ proofs and $K \times P$ total tickets without the best-of-$K$ selection loss. Grinding remains dominated even when proof revenue is considered.

### B.6 Merkle Tree Feedback Structure

In Width-24 compression, the Merkle parent output $S[0..7]$ becomes an input to the next tree level, and $h_H$ occupies $S[16..23]$ at every level. This creates structured, non-i.i.d. inputs to successive permutations. Under PRP, the permutation's output distribution is uniform regardless of input structure. The full 30-round Poseidon2 (8 external + 22 internal) provides complete diffusion across all 24 state elements. Any weakness in PRP for structured inputs would constitute a break of Poseidon2 itself—the same assumption underlying the PoW security analysis (§6.2). ∎

### B.7 Triple-Ticket Independence

The three PoW tickets $\text{ticket}_0 = S[0..7]$, $\text{ticket}_1 = S[8..15]$, and $\text{ticket}_2 = S[16..23]$ are outputs of the same Poseidon2 permutation and therefore deterministically linked. We show that under PRP, this linkage carries no exploitable statistical correlation.

**Proposition.** *Under the PRP assumption on Poseidon2, the joint distribution of $\text{ticket}_0$, $\text{ticket}_1$, and $\text{ticket}_2$ is computationally indistinguishable from three mutually independent uniform samples over $\mathbb{F}_p^8$.*

*Proof.* Let $\pi: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$ be a PRP. For any fixed input $x \in \mathbb{F}_p^{24}$, the output $\pi(x)$ is computationally indistinguishable from a uniform sample over $\mathbb{F}_p^{24}$.

**Ideal world.** Partition $\mathbb{F}_p^{24} = \mathbb{F}_p^8 \times \mathbb{F}_p^8 \times \mathbb{F}_p^8$ as $(A, B, C)$ where $A = S[0..7]$, $B = S[8..15]$, $C = S[16..23]$. If $(A, B, C)$ is uniform over $\mathbb{F}_p^{24}$, then $A$, $B$, $C$ are mutually independent, each uniform over $\mathbb{F}_p^8$ (standard property of product probability spaces).

**Indistinguishability.** Any PPT test detecting inter-ticket correlation can be composed with the PRP output to yield a PRP distinguisher, contradicting the assumption.

Therefore:

$$P(A < T) = p_t = T/p^8$$
$$P(B < T) = p_t$$
$$P(C < T) = p_t$$
$$P(A < T \wedge B < T \wedge C < T) = p_t^3$$
$$P(A < T \vee B < T \vee C < T) = 1 - (1-p_t)^3 = 3p_t - 3p_t^2 + p_t^3$$

The quantity $q = 1 - (1-p_t)^3$ used in §6.4 and Appendix B.2–B.5 is exact under PRP, not an approximation. ∎

---

## Appendix C: GPU Validation

**Scope and limitations.** GPUs can freely allocate compute between ZK and PoW in software—the ZK:PoW ratio is a scheduling parameter. The ASIC-specific claim of simultaneous pipeline execution (§5.4) cannot be validated on GPU. What GPU validates: (1) STARK computation produces PoW tickets as a byproduct, and (2) no measurable throughput difference between input sources.

**Placeholder round constants.** All GPU experiments use the current Stwo implementation, which sets `EXTERNAL_ROUND_CONSTS` and `INTERNAL_ROUND_CONSTS` uniformly to `1234` (placeholder values). This does not affect throughput measurements—Poseidon2's computational cost (field multiplications, additions, MDS matrix application) is identical regardless of round constant values. However, output quality (pseudorandomness, collision resistance) under placeholder constants may differ from production parameters. The throughput results in §C.1 and §C.2 are valid; they should not be interpreted as security validation of the Poseidon2 instantiation.

We implement a complete Width-24 Poseidon2 Circle STARK prover in CUDA—iCFFT/CFFT, Merkle tree commitment, constraint quotient, Fiat-Shamir, and FRI fold—as a single `stark.cu` file (~1600 lines). All Poseidon2 parameters match the paper specification: W=24, Rf=8, Rp=22, M31, compression function mode.

**Test environment:** NVIDIA GeForce RTX 4070 (46 SMs, 5483 CUDA cores, 1920 MHz base / 2475 MHz boost, 12 GB GDDR6X, 504 GB/s).

### C.1 PoW Tickets as a Byproduct of ZK

**Claim (§5.1).** Every Poseidon2 Merkle hash in the STARK simultaneously advances the ZK proof and produces 3 PoW tickets. A standard Width-16 prover produces 0 tickets.

**Method.** Run the GPU STARK prover for trace sizes $2^8$ through $2^{22}$. Count Merkle permutations per proof. Each produces 3 tickets.

| $\ell$ | Trace | STARK (ms) | Merkle perms | Proofs/s | PoW tickets/s |
|---|---|---|---|---|---|
| 8 | 256 | 2.30 | 374 | 434.8 | 487.8K |
| 10 | 1,024 | 3.19 | 1.5K | 313.7 | 1.43M |
| 12 | 4,096 | 4.20 | 6.1K | 238.2 | 4.38M |
| 14 | 16,384 | 5.75 | 24.6K | 173.8 | 12.80M |
| 16 | 65,536 | 8.41 | 98.3K | 118.9 | 35.06M |
| 18 | 262,144 | 13.65 | 393.2K | 73.3 | 86.41M |
| 20 | 1,048,576 | 34.60 | 1.57M | 28.9 | 136.39M |
| 22 | 4,194,304 | 163.48 | 6.29M | 6.1 | 115.46M |

Peak throughput: 136.39M PoW tickets/s at $\ell = 20$. At $\ell = 22$, STARK overhead (NTT, constraint evaluation) dominates.

### C.2 Input Independence

**Claim (§5.3).** Poseidon2 permutation throughput is independent of input source—random nonces (Pure PoW) vs structured Merkle data (Symbiotic mode).

**Method.** Two batched GPU kernels, identical Poseidon2, identical batch size ($2^{20}$):
- **k_pow_batch**: Input from registers (nonce + header_digest from constant memory). No global memory access.
- **k_merkle_batch**: Input from global memory (16 words) + header_digest from constant memory. Writes 8 words back.

3-second warmup per kernel (discarded). 10 measurement rounds with alternating execution order.

| Run | Random (Mperm/s) | Merkle (Mperm/s) | Ratio | Order |
|---|---|---|---|---|
| 1 | 307.09 | 303.29 | 98.8% | R→M |
| 2 | 305.32 | 303.11 | 99.3% | M→R |
| 3 | 304.98 | 302.78 | 99.3% | R→M |
| 4 | 304.95 | 302.85 | 99.3% | M→R |
| 5 | 305.48 | 302.66 | 99.1% | R→M |
| 6 | 305.45 | 303.04 | 99.2% | M→R |
| 7 | 304.89 | 302.82 | 99.3% | R→M |
| 8 | 304.52 | 303.10 | 99.5% | M→R |
| 9 | 305.31 | 303.18 | 99.3% | R→M |
| 10 | 303.47 | 303.06 | 99.9% | M→R |
| **Mean** | **305.15 ± 0.91** | **302.99 ± 0.20** | **99.3% ± 0.3%** | |

95% CI for ratio: [99.1%, 99.5%]. Paired $t$-test: $t = -7.893$, $p < 0.001$ ($p \approx 2.5 \times 10^{-5}$, $df = 9$).

**Interpretation.** The difference is statistically significant ($p < 0.001$) but practically negligible (0.7%). The gap is attributable to GPU global memory I/O overhead in the Merkle kernel (16-word read + 8-word write per permutation), not to input-dependent Poseidon2 computation. Poseidon2's 30-round arithmetic dominates execution time regardless of input source. Execution order has no measurable effect. On ASIC (SRAM latency ~1 cycle vs GPU global memory ~hundreds of cycles), this I/O gap is expected to be substantially reduced.

---

## Appendix D: Theorem 1 — Full Proof and Remarks

This appendix contains the complete proof of Theorem 1 (§2.2) and supporting remarks deferred from the main text.

**Setup.** Let $\pi: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$ be the Width-24 Poseidon2 permutation. In a STARK Merkle commitment phase, the prover evaluates $\pi$ on inputs $x_1, \ldots, x_P$ where $x_j = (n_{L,j} \| n_{R,j} \| h_H)$, with $n_{L,j}, n_{R,j} \in \mathbb{F}_p^8$ (child hashes) and $h_H \in \mathbb{F}_p^8$ (fixed header digest). $P = \sum_{i=0}^{m}(N_i - 1)$ is the total internal Merkle node count.

**Lemma (Input Distinctness).** In Circle STARK, leaf-level inputs are distinct by construction (evaluations at $2^\ell$ distinct domain points). At higher levels, a collision requires $\pi(x_j)[0:7] = \pi(x_{j'})[0:7]$ for distinct $x_j, x_{j'}$. Under PRP, this occurs with probability $\leq p^{-8}$ per pair. Union bound: $\Pr[\text{any collision}] < P^2/(2p^8) < 10^{-60}$ for $P \leq 10^7$. We condition on the "good event" $G$ (all inputs distinct).

**Full proof of Theorem 1.**

*Real world.* The prover evaluates $\pi(x_1), \ldots, \pi(x_P)$ on distinct inputs (conditioned on $G$) and derives $E_j = \{\exists\, k : \text{ticket}_{j,k} < T\}$.

*Ideal world.* Replace $\pi$ with a truly random function $f$. For distinct inputs, $f(x_1), \ldots, f(x_P)$ are independent and uniform over $\mathbb{F}_p^{24}$. Each output decomposes as $\mathbb{F}_p^8 \times \mathbb{F}_p^8 \times \mathbb{F}_p^8$ (product space), so the three ticket comparisons are independent with $\Pr[\text{ticket}_{j,k} < T] = T/p^8$. Hence $\{E_j\}$ are i.i.d. Bernoulli($q$) with $q = 1-(1-T/p^8)^3$.

*Indistinguishability.* Any PPT distinguisher for the event sequences composes with the event-extraction function to yield a PRP distinguisher—contradicting the assumption. ∎

**PRP–PRF switching.** The statistical distance between a random permutation and a random function on $P$ queries is $\leq P^2/(2p^{24}) < 10^{-210}$ for $P = 10^7$. Absorbed into $\mathrm{negl}(\kappa)$.

**Structured inputs.** Merkle feedback (parent outputs → child inputs) and Fiat-Shamir cascades create non-i.i.d. inputs. The PRP guarantee holds for *any* distinct inputs regardless of generation method; any weakness for structured inputs would constitute a Poseidon2 break.

**Side information.** A miner observes STARK execution state (tree level, NTT progress) in addition to PoW outcomes. Under PRP, this does not help predict future PoW outcomes: $\pi(x_j)$ is pseudorandom regardless of how $x_j$ was generated or what the miner knows about computation progress.

**Consensus sufficiency.** Nakamoto consensus security proofs—chain-based [16] and DAG-based [5]—rely on Poisson block arrivals for selfish-mining bounds and difficulty adjustment convergence. The adversary in both frameworks is PPT, so computational indistinguishability from i.i.d. preserves all security bounds.

---

## References

[1] M. Ball, A. Rosen, M. Sabin, and P. N. Vasudevan, "Proofs of Useful Work," IACR ePrint 2017/203, 2017. https://eprint.iacr.org/2017/203

[2] L. Grassi, D. Khovratovich, C. Rechberger, A. Roy, and M. Schofnegger, "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems," USENIX Security 2021. https://eprint.iacr.org/2019/458

[3] L. Grassi, D. Khovratovich, and M. Schofnegger, "Poseidon2: A Faster Version of the Poseidon Hash Function," IACR ePrint 2023/323, 2023. https://eprint.iacr.org/2023/323

[4] StarkWare, "Stwo: A STARK Prover." https://github.com/starkware-libs/stwo

[5] Y. Sompolinsky and M. Sutton, "The DAG KNIGHT Protocol: A Parameterless Generalization of Nakamoto Consensus," IACR ePrint 2022/1494, 2022. https://eprint.iacr.org/2022/1494

[6] Kaspa, "kHeavyHash Specification." https://github.com/kaspanet/rusty-kaspa

[7] M. Fitzi, A. Kiayias, G. Panagiotakos, and A. Russell, "Ofelimos: Combinatorial Optimization via Proof-of-Useful-Work," Crypto 2022. https://eprint.iacr.org/2021/1379

[8] I. Komargodski, I. Schen, and O. Weinstein, "Proofs of Useful Work from Arbitrary Matrix Multiplication," IACR ePrint 2025/685, 2025. https://eprint.iacr.org/2025/685

[9] Y. Bar-On, I. Komargodski, and O. Weinstein, "Proof of Work With External Utilities," arXiv:2505.21685, 2025. https://arxiv.org/abs/2505.21685

[10] Plonky3, "A Toolkit for Polynomial IOPs." https://github.com/Plonky3/Plonky3 (accessed 2026-02-16).

[11] Nockchain, "The zkPoW L1." https://www.nockchain.org/ (accessed 2026-02-18).

[12] S.-P. Merz and À. Rodríguez García, "Skipping Class: Algebraic Attacks exploiting weak matrices and operation modes of Poseidon2(b)," IACR ePrint 2026/306, 2026. https://eprint.iacr.org/2026/306

[13] A. Bak, A. Bariant, A. Boeuf, M. Hostettler, and G. Jazeron, "Claiming bounties on small scale Poseidon and Poseidon2 instances using resultant-based algebraic attacks," IACR ePrint 2026/150, 2026. https://eprint.iacr.org/2026/150

[14] S. Oleksak, R. Gazdik, M. Peresini, and I. Homoliak, "Zk-SNARK Marketplace with Proof of Useful Work," arXiv:2510.09729, 2025. https://arxiv.org/abs/2510.09729

[15] T. Ashur, T. Buschman, and M. Mahzoun, "Algebraic Cryptanalysis of HADES Design Strategy: Application to POSEIDON and Poseidon2," IACR ePrint 2023/537, 2023. https://eprint.iacr.org/2023/537

[16] J. Garay, A. Kiayias, and N. Leonardos, "The Bitcoin Backbone Protocol: Analysis and Applications," Eurocrypt 2015, LNCS 9057, pp. 281–310, 2015.
