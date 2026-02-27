# ZK-SPoW: ZK-Symbiotic Proof of Work

**Anonymous** (Preprint. Work in progress.)

February 2026 — Version 2.0 (Draft)

---

## Abstract

Proof-of-work (PoW) blockchains expend energy solely for network security. Proof of Useful Work (PoUW) attempts to reclaim this cost—a direction Ball et al. [1] show faces fundamental deployment constraints. ZK-based approaches [14]—using STARK proof generation as useful work—are a natural candidate, but face two issues: stateful, multi-phase STARK proving yields trial intervals of tens of milliseconds to seconds (non-memoryless); and losing miners' proofs are discarded, so effective usefulness under difficulty adjustment is no better than pure PoW.

**ZK-SPoW** (ZK-Symbiotic Proof of Work) addresses the non-memoryless problem by extracting PoW at the *permutation level*. Each Poseidon2 permutation within the STARK Merkle tree simultaneously advances the ZK proof and produces PoW tickets. Under the pseudorandom permutation (PRP) assumption, each output is independent regardless of the structured STARK context—yielding Bernoulli trials and memoryless block discovery. Header staleness is bounded by one Merkle commitment phase (~3 ms on GPU (measured), <1 ms on ASIC (projected)).

We instantiate for Kaspa with Width-24 Poseidon2 over M31 ($p = 2^{31}-1$): three PoW tickets per permutation, $U = 100\%$ during continuous proving (proof-level: $\sim 1/N$; pure PoW: $0\%$), zero switching overhead between compute-bound PoW and memory-bound STARK proving. Security claims assume final Poseidon2 production round constants; the current Stwo implementation uses placeholder values.

---

## 1. Introduction

We develop ZK-SPoW as a general framework and instantiate it for **Kaspa**, a PoW blockchain achieving real-time decentralization (RTD) at 100 blocks per second via the DAGKnight protocol [5]. Kaspa is evaluating StarkWare's Stwo—a high-performance STARK prover native to Poseidon2 over M31—making it a natural first candidate for PoW/STARK symbiosis.

### 1.1 The PoW Energy Problem

Traditional PoW schemes produce no output beyond network security. The entire energy expenditure is justified solely by the security guarantees it provides.

### 1.2 The Memoryless Requirement in Proof of Work

A PoW scheme is *progress-free* (or *memoryless*) if the probability of finding a valid block on any trial is independent of all previous trials. This property is fundamental:

1. **Fairness.** Progress-based mining gives miners with more accumulated work a higher instantaneous success probability, undermining proportional-hashrate fairness.
2. **Poisson block arrival.** Independent Bernoulli trials yield a Poisson block arrival process—a prerequisite for Nakamoto-style and DAG-based consensus security proofs.
3. **Difficulty adjustment.** Memoryless trials give expected block time $1/(N \cdot H \cdot q)$. Progress introduces state-dependent variance that breaks difficulty estimation.

SHA-256 (Bitcoin) and kHeavyHash (Kaspa) are memoryless by construction: each hash evaluation is independent. The challenge: useful computation (STARK proving, optimization) is inherently stateful and progressive.

### 1.3 The PoUW Paradox and Its Inversion

Ball et al. [1] formalize **Proof of Useful Work (PoUW)** as a PoW scheme where the mining computation simultaneously produces useful output. Their strict definition requires: (1) the PoW computation itself produces useful output, (2) the verifier can confirm the usefulness, and (3) the useful output is bound to the PoW evidence.

The fundamental tension: PoW requires random exploration (nonce grinding), while useful computation requires specific, deterministic work. Prior PoUW constructions [1, 7, 8, 9] achieve provable security for specific problem classes, but require pre-hashing, SNARGs, or domain-specific verification that limits practical deployment in high-throughput blockchains.

**ZK-SPoW inverts this relationship.** Instead of making PoW results useful, we start from useful computation (STARK proof generation) and observe that PoW tickets emerge as a natural mathematical byproduct:

> **Conventional PoUW:** PoW computation → try to make results useful → fundamental constraints [1]
>
> **ZK-SPoW:** Useful ZK computation → PoW tickets as mathematical byproduct → no contradiction

**Definition (ZK-SPoW).** A PoW scheme where the hash function is a width-extended Poseidon2 compression function operating on STARK Merkle data, such that every permutation simultaneously advances a ZK proof and produces PoW tickets.

The mechanism: STARK proof generation requires millions of Merkle hashes. Each Width-24 Poseidon2 Merkle hash takes $(left\_child, right\_child, header\_digest)$ as input and produces $(pow\_ticket_0, pow\_ticket_1, pow\_ticket_2)$ as output, where $pow\_ticket_0 = merkle\_parent$ simultaneously advances the ZK proof. All three output regions are checked against the difficulty target. The miner cannot choose the Merkle inputs—they are determined by the STARK computation.

**Definition (Usefulness).**

$$U = \frac{\text{ZK-contributing trials}}{\text{total mining trials}}$$

- **Continuous proving**: Every Poseidon2 permutation advances a ZK proof → $U = 100\%$. Multiple proofs do not reduce $U$; ZK work is preserved regardless of PoW outcome.
- **With idle gaps**: Pure PoW fallback trials (no pending ZK work) are wasted → $U$ decreases proportionally.

The per-permutation data overhead is 8/24 ≈ 33% (header digest occupying 8 of 24 state elements); this is the cost of PoW integration, not a usefulness loss.

### 1.4 Our Solution: Permutation-Level PoW Extraction

The central insight of ZK-SPoW is that **non-memoryless computation can produce memoryless PoW**, provided the PoW granularity is finer than the computation's internal state.

A STARK proof is non-memoryless: it consists of trace generation, polynomial commitment (Merkle trees), Fiat-Shamir challenges, and FRI folding—each phase depends on the previous one. A miner who has completed 80% of a proof has "progressed" relative to one who has completed 20%.

However, ZK-SPoW does not use proof completion as the PoW event. Instead, **each individual Poseidon2 permutation within the STARK is an independent PoW trial.** Under the PRP assumption on Poseidon2:

- Each permutation output is pseudorandom, regardless of input structure
- The Merkle tree's feedback structure (parent outputs become child inputs at the next level) does not create exploitable correlation
- Each PoW trial takes nanoseconds (one permutation), not tens of milliseconds to seconds (one proof)

**Result:** ZK-SPoW is progress-free despite embedding PoW within a stateful computation. Block discovery follows a Poisson process. DAGKnight's security proofs apply without modification.

This resolves the fundamental tension between useful computation and memoryless PoW—not by making the computation memoryless, but by extracting PoW at a granularity where the PRP assumption guarantees independence.

### 1.5 Contributions

1. **Memoryless PoW from non-memoryless computation.** We formalize the progress-free property for PoW schemes embedded in stateful computations, prove that ZK-SPoW achieves it under the PRP assumption on Poseidon2, and bound header staleness to one Merkle commitment phase (§2).

2. **PoUW paradox inversion.** We formalize ZK-Symbiotic Proof of Work, where useful ZK computation naturally produces PoW tickets as a cryptographically bound byproduct—inverting the traditional PoUW direction explored by Ball et al. [1], Ofelimos [7], and Komargodski et al. [8, 9] (§1.3).

3. **Width-24 Poseidon2 parameterization.** We specify Width-24 Poseidon2 over M31 in compression function mode (1 permutation per Merkle hash, vs 2 in sponge mode) and verify its security parameters: $R_p = 22$ internal rounds for 128-bit security with $d = 5$ (§6.2).

4. **Complementary bottleneck architecture.** We demonstrate that PoW mining (compute-bound) and STARK proof generation (memory-bound) share Poseidon2 hardware with zero-cycle switching overhead, and provide ASIC architecture analysis for a 7 nm implementation (§5, Appendix A).

5. **GPU empirical validation.** We implement a complete Width-24 Poseidon2 Circle STARK prover on GPU and validate: (a) STARK Merkle hashes produce PoW tickets as a computational byproduct (Appendix C.1); (b) no meaningful throughput difference between random nonce and structured Merkle input sources ($99.3\% \pm 0.3\%$, 10 runs, alternating order; Appendix C.2).

**Generality.** The ZK-SPoW construction is hardware-agnostic and not specific to Kaspa. Any PoW blockchain adopting Poseidon2-based STARKs can apply the same approach. We present Kaspa as the concrete instantiation: its planned Stwo integration, existing ASIC mining ecosystem, and 100 BPS throughput make it a compelling first target.

---

## 2. Memoryless PoW from Non-Memoryless Computation

This section formalizes the core theoretical contribution: how ZK-SPoW extracts progress-free PoW from the inherently stateful STARK proving process.

### 2.1 Progress-Free Property

**Definition 1 (Progress-Free PoW).** A PoW scheme with trial sequence $\tau_1, \tau_2, \ldots$ is *progress-free* if for all $n \geq 1$ and all outcomes $o_1, \ldots, o_n$ of the first $n$ trials:

$$P(\tau_{n+1} \text{ succeeds} \mid o_1, \ldots, o_n) = P(\tau_{n+1} \text{ succeeds}) = q$$

where $q$ is a fixed parameter determined by the difficulty target $T$. Equivalently, the success events form an i.i.d. Bernoulli($q$) sequence. Progress-free trials at rate $R = N \cdot H$ yield Poisson block arrivals with rate $\lambda = Rq$—a prerequisite for Nakamoto-style and DAG-based consensus security proofs.

SHA-256 and kHeavyHash achieve this trivially under the random oracle model. STARK proof generation does not: it is stateful (trace → NTT → Merkle → Fiat-Shamir → FRI), and if proof completion is the PoW event, a miner at 80% completion has higher conditional success probability than one at 10%.

### 2.2 Permutation-Level Independence

ZK-SPoW resolves this by defining the PoW trial at the individual permutation level, not the proof level.

**Setup.** Let $\pi: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$ be the Width-24 Poseidon2 permutation. In a STARK Merkle tree commitment phase, the prover evaluates $\pi$ on inputs $x_1, \ldots, x_P$ where:
- $x_j = (n_{L,j} \| n_{R,j} \| h_H)$ for Merkle node $j$
- $n_{L,j}, n_{R,j} \in \mathbb{F}_p^8$ are the left and right child hashes
- $h_H \in \mathbb{F}_p^8$ is the (fixed) header digest
- $P = \sum_{i=0}^{m}(N_i - 1)$ is the total number of internal Merkle nodes across all commitment phases

Each evaluation produces three PoW tickets: $\text{ticket}_{j,k} = \pi(x_j)[8k:8k+7]$ for $k \in \{0, 1, 2\}$.

**Theorem 1 (Permutation-Level Independence).** *Under the PRP assumption on Poseidon2, the PoW success events*

$$E_j = \{\exists\, k \in \{0,1,2\} : \text{ticket}_{j,k} < T\}, \quad j = 1, \ldots, P$$

*are mutually independent Bernoulli($q$) trials with $q = 1 - (1-p_t)^3$, $p_t = T/2^{248}$.*

*Proof.* We establish two properties:

**(a) Input distinctness.** In a Merkle tree, two internal nodes $j \neq j'$ have $x_j \neq x_{j'}$ unless $n_{L,j} = n_{L,j'}$ and $n_{R,j} = n_{R,j'}$. Since child hashes are outputs of $\pi$ at the previous level (or leaf values), a collision requires finding distinct inputs that produce identical 248-bit outputs—probability at most $\binom{P}{2} \cdot 2^{-248}$, negligible for $P \leq 10^7$.

**(b) PRP implies joint pseudorandomness.** For distinct inputs $x_1, \ldots, x_P$, the PRP property guarantees that $(\pi(x_1), \ldots, \pi(x_P))$ is computationally indistinguishable from $(y_1, \ldots, y_P)$ where each $y_j$ is drawn uniformly from $\mathbb{F}_p^{24}$.

**(c) Independence from uniformity.** If each $y_j$ is uniform over $\mathbb{F}_p^{24} = \mathbb{F}_p^8 \times \mathbb{F}_p^8 \times \mathbb{F}_p^8$, then the three 8-element projections are mutually independent (standard property of product probability spaces), each uniform over $\mathbb{F}_p^8$. The event $\{y_j[0:7] < T\}$ has probability $p_t = T/2^{248}$, and the three tickets within permutation $j$ are independent. Hence $P(E_j) = 1-(1-p_t)^3 = q$.

**(d) Inter-permutation independence.** Since the $y_j$ are independent across distinct $j$ (they are independent uniform samples), the events $E_j$ are independent.

Combining (a)–(d): the events $\{E_j\}_{j=1}^P$ are i.i.d. Bernoulli($q$). ∎

**Corollary 1.** *ZK-SPoW is progress-free. Block discovery follows a Poisson process.* A miner's "progress" through the STARK carries no information about future PoW success—each permutation is an independent trial regardless of its position in the proof.

**Remark on structured inputs.** Merkle tree feedback (parent outputs become child inputs) and Fiat-Shamir cascades (trace selection affects FRI trees) create structured, non-i.i.d. permutation inputs. This does not violate Theorem 1: PRP guarantees pseudorandom outputs regardless of input structure—any weakness would constitute a Poseidon2 break. The total ticket count $P$ is trace-independent (§6.3).

### 2.3 Header Staleness Bound

In ZK-SPoW, the header digest $h_H$ is fixed for the duration of a Merkle commitment phase. A PoW ticket found at the end of a phase references a header that may be slightly behind the current DAG tip. We bound this staleness.

**Theorem 2 (Header Staleness Bound).** *The maximum header staleness of a ZK-SPoW PoW ticket is 1 Merkle commitment phase, bounded by:*

$$\Delta_{stale} \leq \frac{N_{max}}{R_{perm}}$$

*where $N_{max} = \max_i N_i$ is the largest Merkle tree across all commitment phases, and $R_{perm}$ is the permutation throughput.*

*Proof.* A STARK proof consists of $O(10)$ Merkle commitment phases (initial trace + FRI rounds). Each phase $i$ has $N_i - 1$ internal hash evaluations. The header digest register $h_H$ updates between phases to the current block. Within phase $i$, all permutations use the same $h_H$.

At throughput $R_{perm}$, phase $i$ completes in $t_i = (N_i - 1)/R_{perm}$ seconds. The maximum staleness is the duration of the longest phase. Between phases, $h_H$ is refreshed to the current DAG tip. ∎

**Concrete bounds:**
- **GPU** (measured): At $\ell = 20$ (trace size $2^{20}$), the largest Merkle tree has $\sim 2^{20}$ permutations. At 305 Mperm/s (Appendix C.2), $\Delta_{stale} \approx 3.4$ ms.
- **ASIC** (projected): At $\sim 1$ Gperm/s per core with 21 cores, $R_{perm} \approx 21$ Gperm/s. $\Delta_{stale} \approx 0.05$ ms.

**DAGKnight tolerance.** At 100 BPS, the block interval is 10 ms. DAGKnight accepts blocks with parent sets up to its anticone parameter (typically $\geq 10$) blocks deep, tolerating $\sim 100$ ms of latency. A 1-block staleness (10 ms worst case on GPU, <1 ms on ASIC) is well within tolerance and indistinguishable from normal parallel block production in the DAG.

### 2.4 Comparison with Proof-Level Approaches

The key differentiator of ZK-SPoW is the *granularity* at which PoW operates within the ZK computation.

| Property | SHA-256 / kHeavyHash | Proof-Level [14] | Nockchain [11] | **ZK-SPoW** |
|---|---|---|---|---|
| PoW granularity | 1 hash = 1 trial | 1 proof = 1 ticket | Proof → hash | **1 perm = 1 trial** |
| Trial duration | Nanoseconds | Tens of ms–s | Seconds + μs | **Nanoseconds** |
| Progress-free? | **Yes** | **No** — sunk cost | Partially | **Yes** (PRP) |
| Max staleness | 0 (stateless) | Proof duration | Proof duration | **~3 ms (measured)** |
| Losing miners' work | Security only | Proofs discarded | Proofs discarded | **ZK work preserved** |
| Useful work | None | Proof = PoW | Proof then hash | **Perm = PoW + ZK** |
| $U$ | 0% | $\sim 1/N$ | $\sim 1/N$ | **100%** |

ZK-SPoW operates at the finest possible granularity—individual permutations (nanoseconds)—achieving the same progress-free property as traditional hash-based PoW. The STARK proof continues regardless of PoW outcomes: losing miners' ZK computation is preserved, not discarded.

---

## 3. Notation and Definitions

| Symbol | Definition |
|---|---|
| $\mathbb{F}_p$ | Finite field, $p = 2^{31}-1$ (Mersenne prime M31) |
| $\text{Poseidon2}_\pi$ | Poseidon2 permutation over $\mathbb{F}_p^t$ |
| $t$ | State width (number of field elements in permutation) |
| $r$ | Rate: number of input/output elements (sponge mode) |
| $c$ | Capacity: security parameter (sponge mode; hidden elements) |
| $n$ | Hash output size in field elements ($n = 8$, giving 248 bits) |
| $H$ | Block header (all consensus fields; see §4.4) |
| $h_H$ | Header digest: $\text{PoseidonSponge}(H \text{ excluding nonce}) \in \mathbb{F}_p^k$ |
| $k$ | Header digest element count ($k = 8$ for symmetric I/O and three PoW tickets) |
| $(v_1, v_2)$ | Nonce: $v_1, v_2 \in \mathbb{F}_p^8$ |
| $T$ | Target $\in \mathbb{F}_p^8$ (difficulty-adjusted) |
| $p_t$ | Single-ticket success probability: $p_t = T/2^{248}$ |
| $q$ | Per-permutation success probability: $q = 1 - (1-p_t)^3$ |
| $S$ | Poseidon2 state after permutation, $S \in \mathbb{F}_p^t$ |
| $N$ | Number of miners in the network |
| $N_i$ | Number of leaves in Merkle commitment phase $i$ |
| $m$ | Number of FRI folding rounds |
| $P$ | Total Poseidon2 permutations across all STARK phases: $P = \sum_{i=0}^{m}(N_i - 1)$ |
| $\ell$ | Trace size exponent (trace has $2^\ell$ rows) |
| $U$ | Usefulness: ZK-contributing trials / total mining trials (§1.3) |
| $d$ | S-box exponent ($x \mapsto x^d$; $d = 5$ for Poseidon2) |
| $f_{sym}$ | Fraction of Poseidon2 cycles executing STARK Merkle hashes |
| $\mathcal{H}$ | Total PoW hashrate (network-wide) |
| $R_{perm}$ | Permutation throughput (permutations per second) |
| $K$ | Number of distinct traces in multi-trial grinding analysis |
| $B$ | Block reward |
| $Z$ | ZK proof throughput (proofs per second per miner) |
| $F$ | Fee per ZK proof |

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
| Commitment hash | Blake2s (base layer), Poseidon2 (recursive proofs) |

**Note:** All security claims in this paper assume final production round constants. The current Stwo implementation uses placeholder values (`EXTERNAL_ROUND_CONSTS` and `INTERNAL_ROUND_CONSTS` uniformly set to `1234`).

---

## 4. Protocol Specification

### 4.1 PoW Function Replacement

**Current (kHeavyHash [6]):**
```
pre_pow_hash = Blake2b(H excluding nonce and timestamp)
inner        = cSHAKE256_PoW(pre_pow_hash || timestamp || nonce)
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
where $(v_1, v_2)$ are 8 M31 elements each (64 bytes total nonce). The permutation operates in **compression function mode**—all 24 input elements are visible (no hidden capacity). Each permutation produces **three PoW tickets**. The comparison $pow\_hash < target$ interprets both as 248-bit unsigned integers via big-endian concatenation of 8 M31 elements, each zero-padded to 31 bits. In Symbiotic mode, $pow\_hash_0 = merkle\_parent$: the same output advances the ZK proof and serves as a PoW ticket.

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

| Design | Width | Mode | Perm/Merkle | Perm/PoW | Core Δ | Die Δ |
|---|---|---|---|---|---|---|
| A: Header re-hash | 16 | Sponge | 2 | 4 | 0% | 0% |
| B: Width 20 | 20 | Compression | 1 | 1 | +22% | ~+11% |
| **C: Width 24** | **24** | **Compression** | **1** | **1** | **+44–105%** | **~+22–50%** |

**Design A** requires no Stwo modification but costs 4 permutations per PoW draw (2 Merkle + 2 header binding). **Design B** provides only 4 M31 elements (124 bits) for header digest—below the 128-bit birthday bound target. **Design C (selected)** achieves symmetric 8+8+8 I/O with 1 permutation per Merkle hash and per PoW draw, 248-bit header digest (birthday bound $2^{124}$), and three PoW tickets per permutation. Width 24 is within the Poseidon2 paper's analyzed parameter range [3].

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

**Core area overhead: +44% (datapath width) to ~+105% (fully pipelined).** Width-24 requires $R_p = 22$ internal rounds vs Width-16's $R_p = 14$ (§6.3), adding +36% pipeline depth. Die area impact: ~+22% to ~+50% (Poseidon2 is 50% of die; Appendix A). Per-permutation data overhead: 8/24 ≈ 33% of state carries header digest rather than ZK data.

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

**Note on ticket granularity.** The 3 × 8-element partition is a protocol convention, not a cryptographic constraint. Under PRP, any non-overlapping subset of the 24 output elements is pseudorandom. The default 8-element reading (248 bits) matches Stwo's hash output convention. If future hashrate growth demands finer granularity, the comparison window can be widened without modifying the hash function.

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

The only structural change is the nonce expansion from `u64` (8 bytes) to `[F_p; 16]` (64 bytes, +56 bytes per block, ~0.04% of 125 KB).

**Block hash vs PoW hash.** Kaspa computes block identity and PoW with separate hash functions. The block hash (Blake2b-256) provides DAG references—unchanged. Only the PoW function is replaced.

**STARK proofs are NOT included in the block.** They are submitted as independent transactions in the mempool, providing economic value to the ZK ecosystem (vProgs fees). This eliminates the +3–5 MB/s bandwidth overhead that would result from mandatory per-block STARK proofs at 100 BPS.

### 4.5 Header Digest Collision Resistance

The header digest $h_H$ compresses the block header into $k$ field elements. If $k$ is too small, an attacker can find two headers with identical $h_H$, allowing PoW solutions to be transplanted.

| $k$ (elements) | Bits | Birthday bound | Security |
|---|---|---|---|
| 1 | 31 | $2^{15}$ ≈ 32K | INSECURE |
| 2 | 62 | $2^{31}$ ≈ 2×10⁹ | INSECURE |
| 4 | 124 | $2^{62}$ ≈ 4.6×10¹⁸ | SECURE |
| 8 | 248 | $2^{124}$ | Conservative but justified |

**Minimum: $k = 4$** (124-bit collision resistance). We choose **$k = 8$** (248-bit, matching PoW hash size) to enable symmetric 8+8+8 I/O and 3 PoW tickets per permutation.

---

## 5. Operating Modes

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

**Header freshness.** A STARK proof spans multiple Merkle commitment phases—typically $O(10)$. The header digest is fixed per phase; each phase completes within one block interval (<10 ms at ~2G hash/sec). Between phases, the header digest updates to the current block. Maximum staleness is 1 block—see Theorem 2 (§2.3) for formal bound and DAGKnight tolerance analysis.

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
|  Switching cost: 0 cycles (combinational MUX, ~300 gates) |
|  Hashrate: invariant across all modes                 |
+-------------------------------------------------------+
```

The transition is **per-cycle and linear**, not a discrete mode switch. The pipeline is always full—the ratio is determined by SRAM bandwidth.

### 5.4 Hashrate Invariance

**Proposition 2.** *Total PoW hashrate $\mathcal{H}$ is independent of the operating mode.*

*Argument.* $\mathcal{H} = N_{cores} \times \text{throughput\_per\_core}$. Each core's throughput is 1 hash per pipeline depth cycles (fully pipelined), regardless of input source. Input MUX adds zero latency (combinational logic). Therefore $\mathcal{H}$ is constant across all modes.

**GPU validation.** Mean Merkle/Random throughput ratio: $99.3\% \pm 0.3\%$ across 10 runs with alternating execution order (95% CI: [99.1%, 99.5%]; Appendix C.2). The 0.7% gap is statistically significant ($p = 0.006$) but attributable to GPU global memory I/O overhead in the Merkle kernel, not input-dependent Poseidon2 computation.

### 5.5 Complementary Bottleneck Structure

The simultaneous execution of PoW and STARK is possible because they bottleneck on different resources:

| Resource | PoW | STARK | Combined |
|---|---|---|---|
| Poseidon2 cores | **100%** (compute-bound) | Low (SRAM-starved) | ~**100%** |
| NTT unit | 0% | **100%** | **100%** |
| SRAM bandwidth | 0% (registers only) | **100%** (memory-bound) | **100%** |

Under 200 GB/s SRAM bandwidth, $f_{sym} \approx 10\%$ of Poseidon2 cycles execute STARK Merkle hashes ($U = 100\%$); the remaining ~90% fill with Pure PoW ($U = 0\%$). See Appendix A for derivation.

**Width-24 efficiency.** Compression function mode halves STARK's Poseidon2 cycle consumption versus sponge mode (§4.2), freeing more cycles for PoW.

---

## 6. Security Analysis

### 6.1 Poseidon2 Cryptographic Properties

Poseidon2 [3] provides 128-bit security in sponge mode with capacity $c = 8$ M31 elements (248 bits). In compression function mode (ZK-SPoW), security relies on collision resistance and preimage resistance of the permutation—well-established properties within the Poseidon2 framework (see §4.2 for mode comparison).

**Single Primitive Dependency.**

| Component | Current Kaspa | Proposed |
|---|---|---|
| PoW | kHeavyHash (Blake2b + cSHAKE256) | Poseidon2 |
| STARK | Poseidon2 | Poseidon2 |
| Independence | PoW ≠ STARK | PoW = STARK |

**Risk:** A cryptographic break in Poseidon2 compromises both PoW security and STARK validity simultaneously.

### 6.2 Width-24 Security

**Collision resistance (STARK binding).** With 248-bit output (8 M31 elements), the birthday bound is $2^{124}$, providing 124-bit collision security.

**STARK soundness.** The header digest acts as a fixed salt in each Merkle hash—determined before proof generation, cannot be chosen adaptively. STARK soundness reduces to collision resistance with a fixed third input—a strictly easier assumption.

**Preimage resistance (PoW security).** The 30-round Poseidon2 permutation (8 external + 22 internal) satisfies this: no known attack reduces preimage search below the generic $2^{248}$ bound.

**$R_p$ for Width-24.** $R_p = 22$ for 128-bit security at $d = 5$ over M31, computed via Plonky3's round number formula [10], which applies constraints from [2, 3] plus the algebraic attack bound from Khovratovich [15], with margin $R_f += 2$, $R_p \times 1.075$. Total rounds: 30 (8 + 22). S-box operations: 214 per permutation; in compression function mode (§4.2), this yields 25% fewer S-boxes per Merkle hash than Width-16 sponge (214 vs 2 × 142 = 284). Supplementary: $M_I^k$ is invertible for $k = 1..48$ (subspace trail resistance). Algebraic degree after 30 rounds exceeds $2^{69}$, above the $2^{64}$ threshold for 128-bit interpolation security.

**Recent cryptanalysis.** Merz and Rodríguez García [12] improve algebraic CICO attacks by exploiting $M_I$'s sparse structure (round-skipping). For one recommended 128-bit parameter set, the improvement is $2^{106}$ over prior art; however, they note the full-round primitive "does not fall short of its claimed security level." Resultant-based attacks [13] solve small-scale instances ($R_f \leq 6$, $R_p \leq 4$; ≤10 total rounds). These do not affect the 30-round configuration.

### 6.3 Trace Grinding Resistance

Trace selection in Symbiotic mode provides zero advantage for PoW mining under the PRP assumption. We summarize the results here; full proofs are in Appendix B.

Under PRP, three invariants hold:

1. **Ticket count invariance.** Total permutations $P = \sum_{i=0}^{m}(N_i - 1)$ is protocol-determined and trace-independent. The miner cannot increase ticket count by selecting a different trace (Appendix B.2).

2. **Distribution invariance.** The number of valid tickets follows $V \sim \text{Binomial}(P, q)$, where both $P$ and $q = 1 - (1-p_t)^3$ are trace-independent. The *entire distribution* is invariant under trace selection (Appendix B.3).

3. **Grinding is dominated.** Header digest grinding is equivalent to nonce grinding at equal cost (Appendix B.4). Multi-trial grinding ($K$ traces, best-of-$K$) is strictly dominated by $K \times P$ pure PoW permutations (Appendix B.5). Merkle tree feedback does not violate PRP (Appendix B.6).

### 6.4 Triple-Ticket Independence

The three PoW tickets from a single permutation are deterministically linked but computationally independent under PRP.

**Proposition 3.** *Under the PRP assumption, $\text{ticket}_0 = S[0..7]$, $\text{ticket}_1 = S[8..15]$, and $\text{ticket}_2 = S[16..23]$ are mutually independent, each with success probability $p_t = T/2^{248}$. The per-permutation success probability is $q = 1 - (1-p_t)^3$.*

No early-termination optimization exists: evaluating any ticket requires the full permutation, which produces all three simultaneously. Any statistical test detecting inter-ticket correlation can be converted into a PRP distinguisher with equal advantage. Full proof in Appendix B.7.

### 6.5 Quantum Resistance

- Grover's algorithm halves effective hash bits: 248/2 = 124-bit quantum security
- Comparable to SHA-256 under quantum attack (256/2 = 128 bits)
- kHeavyHash: 256/2 = 128-bit quantum security
- **Delta:** −8 bits classical (248 vs 256) / −4 bits quantum (124 vs 128). Both values remain above the 100-bit security floor considered acceptable for PoW.

---

## 7. Pareto Analysis

### 7.1 Competing Designs

Four designs compared under identical die area and power budget:

| Design | Die allocation | Hashrate | ZK | $U$ | Mine? |
|---|---|---|---|---|---|
| Pure PoW | 95% Pos2, 5% ctrl | ~1.9 $\mathcal{H}$ | 0 | 0% | Yes |
| Pure Stwo | 20% Pos2, 40% NTT, 35% SRAM | 0 | $Z_{max}$ | 100% | No |
| **ZK-SPoW** | **50% Pos2, 25% NTT, 20% SRAM** | $\mathcal{H}$ | $Z$ | **100%** ($f_{sym}$~10%) | **Yes** |
| ZK-SPoW+HBM | 50% Pos2, 25% NTT, 20% HBM | $\mathcal{H}$ | $Z_{hbm}$ | 100% ($f_{sym}$~60%) | Yes |

Pure PoW achieves ~1.9× hashrate on a die-area basis but produces no ZK proofs ($U = 0\%$). On hashes-per-watt, the gap narrows to ~1.1–1.2× (idle NTT and SRAM contribute static leakage). Pure Stwo cannot mine. ZK-SPoW achieves $U = 100\%$ during STARK proving; the fraction of cycles in proving ($f_{sym}$) is SRAM-bandwidth limited (~10% for 32 MB SRAM, ~60% for HBM3).

### 7.2 Economic Dominance

In a homogeneous network (all miners use the same ASIC type), difficulty adjusts so that per-miner mining revenue is $B/N$ regardless of absolute hashrate. In a heterogeneous network, mining revenue is proportional to hashrate share: a Pure PoW ASIC with ~1.1–1.2× hashes-per-watt advantage earns ~10–20% more mining revenue per watt than a ZK-SPoW ASIC.

The differentiator is ZK revenue. Per-watt revenue comparison:

$$\text{ZK-SPoW} = \frac{\mathcal{H}_{spow}}{\mathcal{H}_{total}} \cdot B + Z \cdot F \quad \text{vs} \quad \text{Pure PoW} = \frac{\mathcal{H}_{pow}}{\mathcal{H}_{total}} \cdot B$$

where $\mathcal{H}_{pow}/\mathcal{H}_{spow} \approx 1.1\text{–}1.2$ per watt. ZK-SPoW dominates when $Z \cdot F$ exceeds the ~10–20% mining revenue gap.

**PoW security without ZK demand.** When $Z \cdot F = 0$, the ASIC operates as a conventional PoW miner with a hash/watt disadvantage due to die area overhead. Difficulty adjustment absorbs this. The overhead is the cost of optionality—it purchases the ability to capture ZK revenue when the market emerges.

---

## 8. Comparison with Prior Work

### 8.1 Proof-Level PoUW

Several proposals use ZK proof generation as useful work within PoW:

**Proof-level ZK PoW [14].** This proposal treats proof completion as a lottery ticket—completing a STARK proof yields one PoW attempt. Two limitations arise. First, proofs take tens of milliseconds to seconds, making the scheme non-memoryless: miners closer to completion have a higher conditional probability of finding a block, violating the progress-free property (§2.1). Second, losing miners must discard their proofs—only the winner's proofs enter the chain. Under difficulty adjustment with $N$ competing miners, useful output is $\sim 1/N$ of total computation, converging toward pure PoW.

Granularity comparison:
- **Proof-level**: 1 trial per tens of ms–seconds → non-memoryless, losing proofs wasted
- **Permutation-level (ZK-SPoW)**: 1 trial per nanoseconds → PRP-memoryless, ZK work survives PoW failure

**Ofelimos [7].** Uses SNARK proofs as useful work in a provably secure PoUW framework. The useful computation is protocol-mandated (combinatorial optimization), not market-driven. Like ZK-SPoW, Ofelimos addresses the PoUW challenge formally, but operates in the "PoW → useful" direction with domain-specific verification.

**Komargodski et al. [8, 9].** Explore PoUW via matrix multiplication and external utility functions. ZK-SPoW differs: (1) useful computation is market-driven rather than protocol-mandated, and (2) PoW tickets emerge as a byproduct of STARK hashing rather than through separate verification.

### 8.2 Sequential ZK-then-Hash (Nockchain)

Nockchain [11] is a Layer-1 blockchain using zkPoW, launched May 2025. Miners compute a ZK proof of a deterministic puzzle, then hash the proof; the hash must meet a difficulty target.

| | Nockchain (zkPoW) | ZK-SPoW |
|---|---|---|
| PoW hash | hash(ZK proof) < T | Poseidon2 output < T |
| PoW generation | Two-step: ZK proof → external hash | **Single-step**: STARK Merkle hash = PoW hash |
| Field / output | Goldilocks (64-bit), 256-bit hash | M31 (31-bit), 248-bit output |
| Hardware target | GPU | ASIC-optimized |
| Useful work | ZK proof of deterministic puzzle | STARK Merkle hashing (tx verification) |
| STARK enforcement | Mandatory (proof = block) | Market-driven |

The key architectural difference: Nockchain computes the ZK proof first, then hashes it for PoW—two disjoint steps. ZK-SPoW's internal STARK Merkle hash *directly* produces the PoW output from a single permutation.

### 8.3 Relationship to Ball et al.

Ball et al. [1] formalize PoUW in the direction **PoW → useful output**. ZK-SPoW operates inversely: **useful computation → PoW output**.

| Ball et al. criterion | PoW → useful (their framework) | Useful → PoW (ZK-SPoW) |
|---|---|---|
| PoW produces useful output | Partial | **Yes**: every Symbiotic permutation advances a STARK proof |
| Verifier confirms usefulness | Not enforced | Verifiable: STARK proofs are publicly checkable |
| Useful output bound to PoW | Not enforced | **Inherent**: same permutation produces both |

Ball et al.'s hardness results constrain the PoW → useful direction. ZK-SPoW sidesteps these constraints by never attempting to make PoW useful—useful computation happens to produce PoW-valid outputs.

### 8.4 Memorylessness Comparison

See §2.4 (Table 1) for the full comparison. ZK-SPoW is the only ZK-based PoW scheme that achieves the same progress-free property as traditional hash-based PoW, while simultaneously producing useful computation. The cost: the PRP assumption on Poseidon2 replaces the random oracle assumption on SHA-256/kHeavyHash.

---

## 9. Open Questions

1. **$R_p$ for Width-24 (resolved; pending independent verification).** $R_p = 22$ computed via Plonky3's formula [10]; independent cryptanalytic verification pending (§6.2).

2. **Memoryless validation.** Empirical verification of Poisson block inter-arrival times in a test network running ZK-SPoW would complement the theoretical analysis of §2.

**Resolved.** Triple-ticket independence (§6.4) and trace grinding resistance (§6.3) are resolved under the PRP assumption.

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

M31 multipliers (~1K gates, 31×31 bit) are 3.5× smaller than Goldilocks (~3.5K gates, 64×64 bit), giving 3.5× more cores per die at equivalent area.

### A.3 SRAM Bandwidth and $f_{sym}$

Each Merkle hash requires 96 bytes of SRAM I/O (32 read left + 32 read right + 32 write parent).

$$f_{sym} = \frac{\text{SRAM bandwidth} / 96}{\text{total Poseidon2 throughput}} = \frac{200\text{G}/96}{21\text{G}} \approx 9.9\%$$

| Metric | Value |
|---|---|
| STARK hash throughput | ~2.08G hash/s (SRAM-limited) |
| PoW hashrate | ~63G effective (fills idle cycles) |
| STARK proofs/sec | ~260 |

$f_{sym}$ scales with memory bandwidth:

| Memory | Bandwidth | $f_{sym}$ | Proofs/sec |
|---|---|---|---|
| SRAM 32 MB | 200 GB/s | ~10% | ~260 |
| SRAM 64 MB | 400 GB/s | ~20% | ~520 |
| HBM3 8 GB | 1.2 TB/s | ~60% | ~1,560 |
| HBM3E 16 GB | 2.4 TB/s | ~100%† | ~2,625† |

†Compute-saturated: STARK fraction caps at 100%.

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

Under PRP, for any distinct inputs $x_1, \ldots, x_P$ to the permutation, the outputs are jointly pseudorandom. Input distinctness holds with overwhelming probability (Theorem 1, step (a)).

Each permutation produces three PoW tickets. Under PRP, the success event for each permutation (at least one ticket below target $T$) is a Bernoulli trial with parameter:

$$q = 1 - (1-p_t)^3 \approx 3p_t, \quad p_t = T / 2^{248}$$

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

Multi-trial grinding is strictly dominated by pure PoW. Including the NTT and trace generation overhead (omitted above) makes the comparison strictly worse for grinding.

### B.6 Merkle Tree Feedback Structure

In Width-24 compression, the Merkle parent output $S[0..7]$ becomes an input to the next tree level, and $h_H$ occupies $S[16..23]$ at every level. This creates structured, non-i.i.d. inputs to successive permutations. Under PRP, the permutation's output distribution is uniform regardless of input structure. The full 30-round Poseidon2 (8 external + 22 internal) provides complete diffusion across all 24 state elements. Any weakness in PRP for structured inputs would constitute a break of Poseidon2 itself—the same assumption underlying the PoW security analysis (§6.2). ∎

### B.7 Triple-Ticket Independence

The three PoW tickets $\text{ticket}_0 = S[0..7]$, $\text{ticket}_1 = S[8..15]$, and $\text{ticket}_2 = S[16..23]$ are outputs of the same Poseidon2 permutation and therefore deterministically linked. We show that under PRP, this linkage carries no exploitable statistical correlation.

**Proposition.** *Under the PRP assumption on Poseidon2, $\text{ticket}_0$, $\text{ticket}_1$, and $\text{ticket}_2$ are mutually independent.*

*Proof.* Let $\pi: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$ be a PRP. For any fixed input $x \in \mathbb{F}_p^{24}$, the output $\pi(x)$ is computationally indistinguishable from a uniform sample over $\mathbb{F}_p^{24}$.

Partition $\mathbb{F}_p^{24} = \mathbb{F}_p^8 \times \mathbb{F}_p^8 \times \mathbb{F}_p^8$ as $(A, B, C)$ where $A = S[0..7]$, $B = S[8..15]$, $C = S[16..23]$. If $(A, B, C)$ is uniform over $\mathbb{F}_p^{24}$, then $A$, $B$, $C$ are mutually independent, each uniform over $\mathbb{F}_p^8$. This is a standard property of product probability spaces: the uniform distribution on a product space implies independence of coordinate projections.

Therefore:

$$P(A < T) = p_t = T/2^{248}$$
$$P(B < T) = p_t$$
$$P(C < T) = p_t$$
$$P(A < T \wedge B < T \wedge C < T) = p_t^3$$
$$P(A < T \vee B < T \vee C < T) = 1 - (1-p_t)^3 = 3p_t - 3p_t^2 + p_t^3$$

The quantity $q = 1 - (1-p_t)^3$ used in §6.5 and Appendix B.2–B.5 is exact under PRP, not an approximation. ∎

---

## Appendix C: GPU Validation

GPUs can freely allocate compute between ZK and PoW in software—the ZK:PoW ratio is a scheduling parameter. The ASIC-specific claim of simultaneous pipeline execution (§5.6) cannot be validated on GPU. What GPU validates: (1) STARK computation produces PoW tickets as a byproduct, and (2) no measurable throughput difference between input sources.

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

**Claim (§5.4).** Poseidon2 permutation throughput is independent of input source—random nonces (Pure PoW) vs structured Merkle data (Symbiotic mode).

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

95% CI for ratio: [99.1%, 99.5%]. Paired $t$-test: $t = -7.893$, $p = 0.006$.

**Interpretation.** The difference is statistically significant ($p = 0.006$) but practically negligible (0.7%). The gap is attributable to GPU global memory I/O overhead in the Merkle kernel (16-word read + 8-word write per permutation), not to input-dependent Poseidon2 computation. Poseidon2's 30-round arithmetic dominates execution time regardless of input source. Execution order has no measurable effect. On ASIC (SRAM latency ~1 cycle vs GPU global memory ~hundreds of cycles), this I/O gap is expected to vanish.

---

## References

[1] M. Ball, A. Rosen, M. Sabin, and P. N. Vasudevan, "Proofs of Useful Work," IACR ePrint 2017/203, 2017.

[2] L. Grassi, D. Khovratovich, C. Rechberger, A. Roy, and M. Schofnegger, "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems," USENIX Security 2021.

[3] L. Grassi, D. Khovratovich, and M. Schofnegger, "Poseidon2: A Faster Version of the Poseidon Hash Function," IACR ePrint 2023/323, 2023.

[4] StarkWare, "Stwo: A STARK Prover." https://github.com/starkware-libs/stwo

[5] Y. Sompolinsky and M. Sutton, "The DAG KNIGHT Protocol," IACR ePrint 2022/1494, 2022.

[6] Kaspa, "kHeavyHash Specification." https://github.com/kaspanet/rusty-kaspa

[7] M. Fitzi, A. Kiayias, G. Panagiotakos, and A. Russell, "Ofelimos: Combinatorial Optimization via Proof-of-Useful-Work," Crypto 2022.

[8] I. Komargodski and O. Weinstein, "Proofs of Useful Work from Arbitrary Matrix Multiplication," IACR ePrint 2025/685, 2025.

[9] Y. Bar-On, I. Komargodski, and O. Weinstein, "Proof of Work With External Utilities," arXiv:2505.21685, 2025.

[10] Plonky3, "A Toolkit for Polynomial IOPs." https://github.com/Plonky3/Plonky3 (accessed 2026-02-16).

[11] Nockchain, "The zkPoW L1." https://www.nockchain.org/ (accessed 2026-02-18).

[12] S.-P. Merz and À. Rodríguez García, "Skipping Class: Algebraic Attacks exploiting weak matrices and operation modes of Poseidon2(b)," IACR ePrint 2026/306, 2026.

[13] A. Bak, A. Bariant, A. Boeuf, M. Hostettler, and G. Jazeron, "Claiming bounties on small scale Poseidon and Poseidon2 instances using resultant-based algebraic attacks," IACR ePrint 2026/150, 2026.

[14] S. Oleksak, R. Gazdik, M. Peresini, and I. Homoliak, "Zk-SNARK Marketplace with Proof of Useful Work," arXiv:2510.09729, 2025.

[15] D. Khovratovich, "Algebraic attacks on hash functions over prime fields," IACR ePrint 2023/537, 2023.
