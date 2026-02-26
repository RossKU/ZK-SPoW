# ZK-SPoW: ZK-Symbiotic Proof of Work

**Anonymous** (Preprint. Work in progress.)

February 2026 — Version 2.0 (Draft)

---

## Abstract

Proof-of-work (PoW) blockchains expend energy solely for network security. Proof of Useful Work (PoUW) attempts to reclaim this cost—a direction Ball et al. [1] show faces fundamental deployment constraints. ZK-based approaches [14]—using STARK proof generation as useful work—are a natural candidate, but face two issues: stateful, multi-phase STARK proving yields trial intervals of tens of milliseconds to seconds (non-memoryless); and losing miners' proofs are discarded, so effective usefulness under difficulty adjustment is no better than pure PoW.

**ZK-SPoW** (ZK-Symbiotic Proof of Work) addresses the non-memoryless problem by extracting PoW at the *permutation level*. Each Poseidon2 permutation within the STARK Merkle tree simultaneously advances the ZK proof and produces PoW tickets. Under the pseudorandom permutation (PRP) assumption, each output is independent regardless of the structured STARK context—yielding Bernoulli trials and memoryless block discovery. Header staleness is bounded by one Merkle commitment phase (~3 ms on GPU (measured), <1 ms on ASIC (projected)).

We instantiate for Kaspa with Width-24 Poseidon2 over M31 ($p = 2^{31}-1$): three PoW tickets per permutation, $U = 100\%$ during continuous proving (proof-level: $\sim 1/N$; pure PoW: $0\%$), zero switching overhead between compute-bound PoW and memory-bound STARK proving. Security claims assume final Poseidon2 production parameters (§9).

---

## 1. Introduction

We develop ZK-SPoW as a general framework and instantiate it for **Kaspa**, a PoW blockchain achieving real-time decentralization (RTD) at 100 blocks per second via the DAGKnight protocol [5]. Kaspa is evaluating StarkWare's Stwo—a high-performance STARK prover native to Poseidon2 over M31—making it a natural first candidate for PoW/STARK symbiosis.

### 1.1 The PoW Energy Problem

Kaspa uses kHeavyHash—cSHAKE256 (Keccak/SHA-3 family) composed with a 64×64 matrix multiplication over 4-bit nibbles—for proof of work. Like all traditional PoW schemes, the computational work produces no output beyond network security. The entire energy expenditure is justified solely by the security guarantees it provides.

### 1.2 The Memoryless Requirement in Proof of Work

A PoW scheme is *progress-free* (or *memoryless*) if the probability of finding a valid block on any trial is independent of all previous trials. This property is fundamental:

1. **Fairness.** Progress-based mining gives miners with more accumulated work a higher instantaneous success probability, undermining proportional-hashrate fairness.
2. **Poisson block arrival.** Independent Bernoulli trials yield a Poisson block arrival process—a prerequisite for DAGKnight's security proofs [5].
3. **Difficulty adjustment.** Memoryless trials give expected block time $1/(N \cdot H \cdot p)$. Progress introduces state-dependent variance that breaks difficulty estimation.

SHA-256 (Bitcoin) and kHeavyHash (Kaspa) are memoryless by construction: each hash evaluation is independent. The challenge: useful computation (STARK proving, optimization) is inherently stateful and progressive.

### 1.3 The PoUW Paradox and Its Inversion

Ball et al. [1] formalize **Proof of Useful Work (PoUW)** as a PoW scheme where the mining computation simultaneously produces useful output. Their strict definition requires: (1) the PoW computation itself produces useful output, (2) the verifier can confirm the usefulness, and (3) the useful output is bound to the PoW evidence.

The fundamental tension: PoW requires random exploration (nonce grinding), while useful computation requires specific, deterministic work. Prior PoUW constructions [1, 7, 8, 9] achieve provable security for specific problem classes, but require pre-hashing, SNARGs, or domain-specific verification that limits practical deployment in high-throughput blockchains (100 BPS).

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

3. **Width-24 Poseidon2 parameterization.** We specify Width-24 Poseidon2 over M31 in compression function mode (1 permutation per Merkle hash, vs 2 in sponge mode) and verify its security parameters: $R_p = 22$ internal rounds for 128-bit security with $D = 5$ (§6.3).

4. **Complementary bottleneck architecture.** We demonstrate that PoW mining (compute-bound) and STARK proof generation (memory-bound) share Poseidon2 hardware with zero-cycle switching overhead, and provide gate-level ASIC architecture analysis for a 7 nm implementation (§5, Appendix A).

5. **GPU empirical validation.** We implement a complete Width-24 Poseidon2 Circle STARK prover on GPU and validate: (a) STARK Merkle hashes produce PoW tickets as a computational byproduct (Appendix B.1); (b) no meaningful throughput difference between random nonce and structured Merkle input sources ($99.3\% \pm 0.3\%$, 10 runs, alternating order; Appendix B.2).

**Generality.** The ZK-SPoW construction is hardware-agnostic and not specific to Kaspa. Any PoW blockchain adopting Poseidon2-based STARKs can apply the same approach. We present Kaspa as the concrete instantiation: its planned Stwo integration, existing ASIC mining ecosystem, and 100 BPS throughput make it a compelling first target.

---

## 2. Memoryless PoW from Non-Memoryless Computation

This section formalizes the core theoretical contribution: how ZK-SPoW extracts progress-free PoW from the inherently stateful STARK proving process.

### 2.1 Progress-Free Property: Formal Definition

**Definition 1 (Progress-Free PoW).** A PoW scheme with trial sequence $\tau_1, \tau_2, \ldots$ is *progress-free* if for all $n \geq 1$ and all outcomes $o_1, \ldots, o_n$ of the first $n$ trials:

$$P(\tau_{n+1} \text{ succeeds} \mid o_1, \ldots, o_n) = P(\tau_{n+1} \text{ succeeds}) = q$$

where $q$ is a fixed parameter determined by the difficulty target $T$.

In other words, no history of previous trials—successful or not—affects the probability of the next trial succeeding. This is equivalent to requiring that the success events $\{E_i\}_{i \geq 1}$ form an i.i.d. Bernoulli($q$) sequence.

**Definition 2 (Memoryless Block Discovery).** Block discovery is *memoryless* if the number of blocks found in any time interval $[t, t+\Delta]$ depends only on the aggregate hashrate during that interval, not on computation performed before time $t$. Formally, the block arrival process is Poisson with rate $\lambda = N \cdot H \cdot q$, where $N$ is the number of miners, $H$ is per-miner trial rate, and $q$ is the per-trial success probability.

**Proposition 1.** *Progress-free PoW implies memoryless block discovery.*

*Proof sketch.* If each trial is an independent Bernoulli($q$) event and trials occur at rate $H$ per miner, the total network trial rate is $R = N \cdot H$. The number of successes in time $\Delta$ follows $\text{Binomial}(R\Delta, q)$. For large $R$ and small $q$ (the regime of PoW mining), this converges to $\text{Poisson}(\lambda\Delta)$ with $\lambda = Rq$. The Poisson process is memoryless: the conditional distribution of future arrivals given the past is independent of the past. ∎

**Standard PoW is progress-free.** In SHA-256 mining, each trial computes $H(\text{header} \| \text{nonce})$ and checks if the output is below the target. The hash function's preimage resistance ensures that no information from trial $n$ helps predict trial $n+1$. This is immediate from the random oracle model.

**The challenge for ZK-based PoW.** STARK proof generation is stateful: trace generation → NTT → Merkle commitment → Fiat-Shamir → FRI folding. Each phase depends on the previous one. If proof completion is the PoW event (as in proof-level approaches), the scheme is *not* progress-free: a miner at 80% completion has a higher conditional probability of finding a block in the next second than a miner at 10% completion.

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

*are mutually independent Bernoulli($q$) trials with $q = 1 - (1-p)^3$, $p = T/2^{248}$.*

*Proof.* We establish two properties:

**(a) Input distinctness.** In a Merkle tree, two internal nodes $j \neq j'$ have $x_j \neq x_{j'}$ unless $n_{L,j} = n_{L,j'}$ and $n_{R,j} = n_{R,j'}$. Since child hashes are outputs of $\pi$ at the previous level (or leaf values), a collision requires finding distinct inputs that produce identical 248-bit outputs—probability at most $\binom{P}{2} \cdot 2^{-248}$, negligible for $P \leq 10^7$.

**(b) PRP implies joint pseudorandomness.** For distinct inputs $x_1, \ldots, x_P$, the PRP property guarantees that $(\pi(x_1), \ldots, \pi(x_P))$ is computationally indistinguishable from $(y_1, \ldots, y_P)$ where each $y_j$ is drawn uniformly from $\mathbb{F}_p^{24}$.

**(c) Independence from uniformity.** If each $y_j$ is uniform over $\mathbb{F}_p^{24} = \mathbb{F}_p^8 \times \mathbb{F}_p^8 \times \mathbb{F}_p^8$, then the three 8-element projections are mutually independent (standard property of product probability spaces), each uniform over $\mathbb{F}_p^8$. The event $\{y_j[0:7] < T\}$ has probability $p = T/2^{248}$, and the three tickets within permutation $j$ are independent. Hence $P(E_j) = 1-(1-p)^3 = q$.

**(d) Inter-permutation independence.** Since the $y_j$ are independent across distinct $j$ (they are independent uniform samples), the events $E_j$ are independent.

Combining (a)–(d): the events $\{E_j\}_{j=1}^P$ are i.i.d. Bernoulli($q$). ∎

**Corollary 1.** *ZK-SPoW is progress-free. Block discovery follows a Poisson process.*

*Proof.* By Theorem 1, each Poseidon2 permutation in the STARK is an independent Bernoulli trial. The total number of valid tickets across all permutations follows $\text{Binomial}(P, q)$. Whether a permutation occurs at the beginning of a proof (phase 0) or the end (FRI round $m$) does not affect its success probability. A miner's "progress" through the STARK computation carries no information about future PoW success. By Proposition 1, block discovery is Poisson. ∎

**Remark on Merkle tree feedback.** The Merkle tree creates structured, non-i.i.d. inputs: parent outputs at level $\ell$ become child inputs at level $\ell+1$, and $h_H$ occupies $S[16..23]$ at every level. This feedback does not violate Theorem 1 because: (1) the PRP assumption guarantees pseudorandom outputs *regardless of input structure*; (2) any weakness for structured inputs would constitute a PRP break of Poseidon2 itself—the same assumption underlying all Poseidon2 security claims; (3) the full 30-round permutation (8 external + 22 internal) provides complete diffusion across all 24 state elements.

**Remark on Fiat-Shamir cascade.** Trace selection affects FRI Merkle trees via Fiat-Shamir challenge dependency on the initial Merkle root. Changing the trace changes all subsequent challenges, folding points, and FRI Merkle trees. However, each FRI tree size $N_i$ is protocol-determined, and PRP ensures each resulting ticket has identical success probability $q$. The total ticket count $P$ is trace-independent. See §6.3 for the full trace grinding analysis.

### 2.3 Header Staleness Bound

In ZK-SPoW, the header digest $h_H$ is fixed for the duration of a Merkle commitment phase. A PoW ticket found at the end of a phase references a header that may be slightly behind the current DAG tip. We bound this staleness.

**Theorem 2 (Header Staleness Bound).** *The maximum header staleness of a ZK-SPoW PoW ticket is 1 Merkle commitment phase, bounded by:*

$$\Delta_{stale} \leq \frac{N_{max}}{R_{perm}}$$

*where $N_{max} = \max_i N_i$ is the largest Merkle tree across all commitment phases, and $R_{perm}$ is the permutation throughput.*

*Proof.* A STARK proof consists of $O(10)$ Merkle commitment phases (initial trace + FRI rounds). Each phase $i$ has $N_i - 1$ internal hash evaluations. The header digest register $h_H$ updates between phases to the current block. Within phase $i$, all permutations use the same $h_H$.

At throughput $R_{perm}$, phase $i$ completes in $t_i = (N_i - 1)/R_{perm}$ seconds. The maximum staleness is the duration of the longest phase. Between phases, $h_H$ is refreshed to the current DAG tip. ∎

**Concrete bounds:**
- **GPU** (measured): At $k = 20$ (trace size $2^{20}$), the largest Merkle tree has $\sim 2^{20}$ permutations. At 305 Mperm/s (Table B.2), $\Delta_{stale} \approx 3.4$ ms.
- **ASIC** (projected): At $\sim 1$ Gperm/s per core with 21 cores, $R_{perm} \approx 21$ Gperm/s. $\Delta_{stale} \approx 0.05$ ms.

**DAGKnight tolerance.** At 100 BPS, the block interval is 10 ms. DAGKnight accepts blocks with parent sets up to $k$ blocks deep (anticone parameter typically $k \geq 10$), tolerating $\sim 100$ ms of latency. A 1-block staleness (10 ms worst case on GPU, <1 ms on ASIC) is well within tolerance and indistinguishable from normal parallel block production in the DAG.

### 2.4 Comparison with Proof-Level Approaches

The key differentiator of ZK-SPoW is the *granularity* at which PoW operates within the ZK computation.

| Property | Proof-Level [2510.09729] | Sequential (Nockchain [11]) | **Permutation-Level (ZK-SPoW)** |
|---|---|---|---|
| PoW granularity | 1 proof = 1 lottery ticket | Proof → hash → check | **1 permutation = 1 trial** |
| Trial duration | Tens of ms to seconds | Seconds (proof) + μs (hash) | **Nanoseconds** |
| Progress-free? | **No** — partial proof = sunk cost | Partially — proof phase is progressive | **Yes** — PRP guarantees independence |
| Max staleness | Proof duration (tens of ms–s) | Proof duration (seconds) | **1 Merkle phase (~3 ms measured)** |
| Losing miners' work | Proofs discarded | Proofs discarded | **ZK proof survives PoW failure** |
| Useful work coupling | Proof = PoW (tight) | Proof then hash (loose) | **Permutation = PoW + ZK (tight)** |
| Usefulness $U$ | $\sim 1/N$ ($N$ miners; losing proofs discarded) | $\sim 1/N$ | **100%** (ZK work preserved) |

**Proof-level approaches** (e.g., arXiv 2510.09729) treat proof completion as the lottery event. Two issues arise. First, a proof takes tens of milliseconds to seconds—during that window, a miner who is closer to completion has a higher conditional probability of finding a block, making the scheme non-memoryless. Second, only the winning miner's proofs enter the chain; losing miners must discard their proofs. Under difficulty adjustment, effective usefulness converges toward that of pure PoW as competition grows.

**Sequential approaches** (Nockchain [11]) compute a ZK proof first, then hash the proof for PoW. The hash step is memoryless, but the proof step is progressive. The overall scheme has mixed memoryless properties: between proofs, trials are independent; within a proof, progress accumulates.

**ZK-SPoW** operates at the finest possible granularity: individual permutations. Since each Poseidon2 evaluation takes nanoseconds and produces an independent output (Theorem 1), the scheme is fully progress-free. The useful work (advancing the STARK Merkle tree) happens as a side effect of the PoW trial, not as a prerequisite for it. Crucially, the STARK proof continues regardless of PoW outcomes—losing miners' ZK computation is not wasted.

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
| $S$ | Poseidon2 state after permutation, $S \in \mathbb{F}_p^t$ |
| $U$ | Per-permutation usefulness: $t_0/t$ ($t_0$ useful elements in width-$t$ state) |
| $f_{sym}$ | Fraction of Poseidon2 cycles executing STARK Merkle hashes |
| $U_{avg}$ | Time-averaged usefulness: $f_{sym} \times U$ |

**Stwo baseline parameters** (confirmed from source code [4]):

| Parameter | Value |
|---|---|
| Field | M31, $p = 2^{31}-1$ |
| Hash output | 8 elements = 248 bits |
| Standard width | $t_0 = 16$ (sponge mode: rate 8, capacity 8) |
| External rounds $R_f$ | 8 (4 + 4) |
| Internal rounds $R_p$ | 14 |
| S-box exponent | $\alpha = 5$ |
| Merkle hash | 2 permutations per node (sponge: absorb left[8], absorb right[8]) |
| Commitment hash | Blake2s (base layer), Poseidon2 (recursive proofs) |

**Note:** All security claims in this paper assume final production round constants. The current Stwo implementation uses placeholder values (`EXTERNAL_ROUND_CONSTS` and `INTERNAL_ROUND_CONSTS` uniformly set to `1234`; see §9).

---

## 4. Protocol Specification

### 4.1 PoW Function Replacement

**Current (kHeavyHash):**
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

**Core area overhead: +44% (datapath width) to ~+105% (fully pipelined).** Width-24 requires $R_p = 22$ internal rounds vs Width-16's $R_p = 14$ (§6.3), adding +36% pipeline depth. Die area impact: ~+22% to ~+50% (Poseidon2 is 50% of die; Appendix A). Usefulness cost: $U = t_0/t = 16/24 \approx 67\%$.

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

$U = t_0/t = 16/24 \approx 67\%$. GPU validation confirms up to 136.39M PoW tickets/s as a byproduct of STARK proof generation (Appendix B.1).

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

**GPU validation.** No meaningful throughput difference was observed: mean ratio $99.3\% \pm 0.3\%$ across 10 runs with alternating execution order (95% CI: [99.1%, 99.5%]; Appendix B.2). The 0.7% gap is attributable to GPU global memory I/O overhead, not input-dependent computation.

### 5.5 Difficulty Independence

$U$ is determined by ZK demand and width ratio, not by PoW difficulty.

| Condition | $U$ | Rationale |
|---|---|---|
| Stwo Prover active, any difficulty | ≈ 67% | ZK proof, minus width overhead |
| No ZK demand, any difficulty | 0% | Pure PoW = security only |

### 5.6 Complementary Bottleneck Structure

The simultaneous execution of PoW and STARK is possible because they bottleneck on different resources:

| Resource | PoW | STARK | Combined |
|---|---|---|---|
| Poseidon2 cores | **100%** (compute-bound) | Low (SRAM-starved) | ~**100%** |
| NTT unit | 0% | **100%** | **100%** |
| SRAM bandwidth | 0% (registers only) | **100%** (memory-bound) | **100%** |

Under 200 GB/s SRAM bandwidth, the STARK allocation is $f_{sym} \approx 10\%$ of Poseidon2 cycles, yielding $U_{avg} = f_{sym} \times U \approx 6.7\%$ time-averaged usefulness (see Appendix A for derivation).

**Width-24 efficiency.** Width-24 compression uses 1 permutation per Merkle hash versus Width-16 sponge's 2 permutations. This halves STARK's Poseidon2 cycle consumption, freeing more cycles for PoW.

---

## 6. Security Analysis

### 6.1 Poseidon2 Cryptographic Properties

Poseidon2 [3] provides 128-bit security in sponge mode with capacity $c = 8$ M31 elements (248 bits). In compression function mode (ZK-SPoW), security relies on collision resistance and preimage resistance of the permutation—well-established properties within the Poseidon2 framework. The Poseidon2 paper explicitly recommends compression function mode for Merkle trees (Section 4.2 of [3]).

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

**$R_p$ for Width-24.** $R_p = 22$ for 128-bit security at $D = 5$ over M31, computed via Plonky3's round number formula [10], which applies constraints from [2, 3] plus the algebraic attack bound from Khovratovich et al. (ePrint 2023/537), with margin $R_f += 2$, $R_p \times 1.075$. Total rounds: 30 (8 + 22). S-box operations: 214 per permutation; however, compression function mode requires only 1 permutation per Merkle hash (vs 2 in sponge mode), yielding 25% fewer S-boxes per hash (214 vs 2 × 142 = 284). Supplementary: $M_I^k$ is invertible for $k = 1..48$ (subspace trail resistance). Algebraic degree after 30 rounds exceeds $2^{69}$, above the $2^{64}$ threshold for 128-bit interpolation security.

**Recent cryptanalysis.** Merz and Rodríguez García [12] improve algebraic CICO attacks by exploiting $M_I$'s sparse structure (round-skipping). For one recommended 128-bit parameter set, the improvement is $2^{106}$ over prior art; however, they note the full-round primitive "does not fall short of its claimed security level." Resultant-based attacks [13] solve small-scale instances ($R_f \leq 6$, $R_p \leq 4$; ≤10 total rounds). These do not affect the 30-round configuration.

### 6.3 Trace Grinding Resistance

We prove that trace selection in Symbiotic mode provides zero advantage for PoW mining under the PRP assumption.

**Assumptions.** (1) PRP: Poseidon2 $\pi: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$ is a pseudorandom permutation. (2) Fixed tree sizes: Each Merkle phase $i$ has $N_i$ leaves, yielding $N_i - 1$ evaluations, independent of trace content. (3) Fixed header digest per phase.

**Ticket Count Invariance.** Total permutations: $P = \sum_{i=0}^{m}(N_i - 1)$. Since each $N_i$ is a protocol parameter independent of the trace: $\forall t_1, t_2: P(t_1) = P(t_2) = P$. The miner cannot increase ticket count by selecting a different trace.

**Distribution Invariance.** Under PRP, for distinct inputs $x_1, \ldots, x_P$, the outputs are jointly pseudorandom. All permutation inputs are distinct with overwhelming probability (248-bit collision). Each permutation produces three PoW tickets. The success event for each permutation is Bernoulli with parameter $q = 1 - (1-p)^3$, $p = T/2^{248}$. This depends only on $T$, not the input. The number of valid tickets follows $V \sim \text{Binomial}(P, q)$—both parameters are trace-independent. The *entire distribution* of valid tickets is invariant under trace selection.

**Header Digest Grinding.** The miner can produce different $h_H$ by choosing different parent blocks or transaction sets. With the same trace but a new $h_H$, the entire Merkle tree must be rebuilt (cost: $P$ permutations), producing $P$ new triples. Under PRP, this is functionally equivalent to $P$ pure PoW nonce hashes—identical cost, identical distribution. Header digest grinding offers no advantage beyond standard nonce grinding.

**Multi-Trial Grinding.** A miner computing $k$ distinct traces and selecting the best: cost $k \times P$ permutations; benefit $\max(V_1, \ldots, V_k)$ where each $V_i \sim \text{Binomial}(P, q)$. For the same budget in pure PoW: $V_{PoW} \sim \text{Binomial}(kP, q)$. Since $E[\max(V_1, \ldots, V_k)] < kPq = E[V_{PoW}]$ for all $k \geq 2$, multi-trial grinding is strictly dominated by pure PoW.

**Merkle Tree Feedback.** Parent outputs become inputs at the next level, and $h_H$ is constant at every level. This creates structured inputs. Under PRP, output distribution is uniform regardless of input structure—any weakness would constitute a Poseidon2 break. ∎

### 6.4 Triple-Ticket Independence

The three PoW tickets $\text{ticket}_0 = S[0..7]$, $\text{ticket}_1 = S[8..15]$, $\text{ticket}_2 = S[16..23]$ are outputs of the same permutation and therefore deterministically linked.

**Proposition 3.** *Under the PRP assumption on Poseidon2, $\text{ticket}_0$, $\text{ticket}_1$, and $\text{ticket}_2$ are mutually independent.*

*Proof.* Let $\pi: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$ be a PRP. For any fixed input, $\pi(x)$ is computationally indistinguishable from uniform over $\mathbb{F}_p^{24}$. Partition as $(A, B, C) \in \mathbb{F}_p^8 \times \mathbb{F}_p^8 \times \mathbb{F}_p^8$. Uniform on a product space implies independence of coordinate projections. Therefore:

$$P(A < T) = P(B < T) = P(C < T) = p = T/2^{248}$$
$$P(A < T \vee B < T \vee C < T) = 1 - (1-p)^3 = 3p - 3p^2 + p^3$$

This is exact under PRP, not an approximation. ∎

**Implication.** A miner who observes one ticket gains no information about the others. The only way to evaluate all tickets is to compute the full permutation—which already produces all three simultaneously. No early-termination optimization exists.

**Distinguisher reduction.** Any statistical test detecting correlation among the three ticket regions can be converted into a PRP distinguisher with equal advantage. Under PRP, no such efficient test exists.

### 6.5 PoW Hash Distribution

All three output regions are outputs of the same Poseidon2 permutation. In compression function mode, all 24 state elements are visible by design. Security relies on the permutation's PRP properties. An attacker who could predict one ticket from another would violate PRP—equivalent to breaking Poseidon2.

**Triple-ticket mining:** Under PRP, with three 248-bit tickets per permutation:

$$P(\text{valid}) = 1 - (1-p)^3 = 3p - 3p^2 + p^3, \quad p = T/2^{248}$$

Note: the number of tickets per permutation does not affect mining economics in a homogeneous network—difficulty adjustment absorbs any change in per-permutation success probability.

### 6.6 Quantum Resistance

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
| **ZK-SPoW** | **50% Pos2, 25% NTT, 20% SRAM** | $\mathcal{H}$ | $Z$ | **≈67%** | **Yes** |
| ZK-SPoW+HBM | 50% Pos2, 25% NTT, 20% HBM | $\mathcal{H}$ | $Z_{hbm}$ | ≈67% | Yes |

Pure PoW achieves ~1.9× hashrate on a die-area basis but produces no ZK proofs ($U = 0\%$). On hashes-per-watt, the gap narrows to ~1.1–1.2× (idle NTT and SRAM contribute static leakage). Pure Stwo cannot mine. ZK-SPoW achieves $U \approx 67\%$ and mining—the 33% gap is the cost of PoW integration.

### 7.2 Economic Dominance

Difficulty adjusts to total network hashrate. Per-ASIC mining revenue is $B/N$ regardless of absolute hashrate. The differentiator is ZK revenue:

$$\text{ZK-SPoW revenue} = B/N + Z \cdot F > B/N = \text{Pure PoW revenue} \quad (\text{for any } ZF > 0)$$

Pure PoW's ~1.1–1.2× power-efficiency advantage yields at most ~10–20% more mining revenue per watt. Once ZK fee income $Z \cdot F$ exceeds this margin, ZK-SPoW dominates.

**PoW security without ZK demand.** When $Z \cdot F = 0$, the ASIC operates as a conventional PoW miner with a hash/watt disadvantage due to die area overhead. Difficulty adjustment absorbs this. The overhead is the cost of optionality—it purchases the ability to capture ZK revenue when the market emerges.

---

## 8. Comparison with Prior Work

### 8.1 Proof-Level PoUW

Several proposals use ZK proof generation as useful work within PoW:

**arXiv 2510.09729.** This proposal treats proof completion as a lottery ticket—completing a STARK proof yields one PoW attempt. Two limitations arise. First, proofs take tens of milliseconds to seconds, making the scheme non-memoryless: miners closer to completion have a higher conditional probability of finding a block, violating the progress-free property (§2.1). Second, losing miners must discard their proofs—only the winner's proofs enter the chain. Under difficulty adjustment with $N$ competing miners, useful output is $\sim 1/N$ of total computation, converging toward pure PoW.

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

| Approach | PoW Granularity | Max Staleness | Progress-Free? | Losing miners' work | $U$ |
|---|---|---|---|---|---|
| SHA-256 (Bitcoin) | 1 hash = 1 trial | 0 (stateless) | **Yes** | Security only | 0% |
| kHeavyHash (Kaspa) | 1 hash = 1 trial | 0 (stateless) | **Yes** | Security only | 0% |
| Proof-level [2510.09729] | 1 proof = 1 trial | Tens of ms–s | **No** | Proofs discarded | $\sim 1/N$ |
| Nockchain [11] | Proof → hash | Seconds | Partially | Proofs discarded | $\sim 1/N$ |
| **ZK-SPoW** | **1 perm = 1 trial** | **~3 ms (measured)** | **Yes** (PRP) | **ZK work preserved** | **100%** |

ZK-SPoW is the only ZK-based PoW scheme that achieves the same progress-free property as traditional hash-based PoW, while simultaneously producing useful computation. The cost: the PRP assumption on Poseidon2 replaces the random oracle assumption on SHA-256/kHeavyHash.

---

## 9. Open Questions

1. **Stwo M31 Poseidon2 maturity.** The current Stwo production prover uses Blake2s or Poseidon over Stark252 for Merkle hashing; an M31-native Poseidon2 Merkle mode does not yet exist in production (only as an example with placeholder round constants—`1234`). Deploying ZK-SPoW requires promotion to production status.

2. **$R_p$ for Width-24 (resolved; pending independent verification).** $R_p = 22$ computed via Plonky3's formula [10]; independent cryptanalytic verification pending (§6.2).

3. **Stwo-Kaspa verifier.** Width-24 Poseidon2 compression is a different cryptographic function from Width-16 sponge. Plonky3 [10] provides a production Width-24 implementation. Remaining: (a) round constant finalization, (b) Stwo integration, (c) independent security certification.

4. **Hard fork governance.** kHeavyHash → Poseidon2 renders existing ASICs obsolete and requires community consensus.

5. **Mining pool protocol.** The 64-byte nonce requires stratum protocol updates. Each element must satisfy $0 \leq v_i < 2^{31}-1$ (valid M31 range).

6. **ZK market maturity.** Economic advantage depends on ZK proof demand. Timeline uncertain.

7. **Complementary bottleneck validation (partially resolved).** GPU validates: no throughput difference between input sources ($99.3\% \pm 0.3\%$, Appendix B.2), and byproduct ticket generation up to 136.39M tickets/s (Appendix B.1). ASIC-specific simultaneous execution claim requires ASIC implementation for full validation.

8. **(Resolved.)** Triple-ticket independence (§6.4) and trace grinding resistance (§6.3) resolved under PRP.

9. **Memoryless validation.** Empirical verification of Poisson block inter-arrival times in a test network running ZK-SPoW would complement the theoretical analysis of §2.

---

## Appendix A: ASIC Architecture Details

*Note: All parameters in this appendix are reference estimates for a hypothetical design. No chip has been fabricated.*

### A.1 ZK-Symbiotic ASIC Block Diagram

```
+-- ZK-SPoW ASIC (7nm, ~200W) -----------------------------------------+
|                                                                        |
|  +-- Poseidon2 Core Array (50% die) --------------------------------+ |
|  |  [Core 0] [Core 1] [Core 2] ... [Core N-1]                       | |
|  |  Each: Width-24 pipeline (M31), per-cycle MUX                     | |
|  |                                                                    | |
|  |  +---------+     +------------------+   +----------+              | |
|  |  |Input MUX|---->|Poseidon2 Pipeline|-->|Output RTR|              | |
|  |  +--+--+---+     | [R1][R2]...[R30] |   +--+---+---+              | |
|  |     |  |          +------------------+      |   |                  | |
|  |  SRAM  nonce                           SRAM  target               | |
|  |  data  counter                         write comparator            | |
|  +--------------------------------------------------------------------+ |
|                                                                        |
|  +-- NTT Butterfly Unit (25% die) ----------------------------------+ |
|  |  M31 multiply-accumulate array. STARK: LDE + FRI folding          | |
|  +--------------------------------------------------------------------+ |
|                                                                        |
|  +-- SRAM (20% die) ------------------------------------------------+ |
|  |  32 MB on-chip, ~200 GB/s bandwidth                                | |
|  +--------------------------------------------------------------------+ |
|                                                                        |
|  +-- Control & I/O (5% die) ----------------------------------------+ |
|  |  PoW controller, STARK controller, scheduler, network interface    | |
|  +--------------------------------------------------------------------+ |
+------------------------------------------------------------------------+
```

### A.2 M31 Field Arithmetic

```
a, b in F_p  where p = 2^31 - 1
c = a * b                      // 31 x 31 -> 62-bit product
c_hi = c[61:31]                // upper 31 bits
c_lo = c[30:0]                 // lower 31 bits
result = c_hi + c_lo           // Mersenne reduction
if result >= p: result -= p    // final correction
```

| Metric | M31 | Goldilocks ($2^{64} - 2^{32} + 1$) |
|---|---|---|
| Multiplier width | 31 × 31 → 62 bit | 64 × 64 → 128 bit |
| Reduction | 1 addition | shift + sub + add |
| Gate count | ~1,000 | ~3,500 |
| Latency | 1 cycle | 2–3 cycles |
| Area ratio | **1×** | 3.5× |

**M31 advantage:** 3.5× more multipliers per die → 3.5× more Poseidon2 cores.

### A.3 S-box and MDS

**S-box ($x^5 \bmod p$):** 3 sequential multiplications (x² → x⁴ → x⁵). Gate count: ~3,000 (M31).

**External MDS:** Block-circulant M4 structure. Width 24 = 6 groups of 4. Full external MDS: circ(2M4, M4, ..., M4)—all arithmetic is shift+add only.

**Internal MDS:** Sparse: $M_I = \mathbf{1}\mathbf{1}^\top + \text{diag}(V)$, $V = [-2, 1, 2, 4, \ldots, 2^{22}]$ (Plonky3 values). Width 24: 24 shift-add operations per internal round (+50% vs Width 16).

### A.4 Per-Round Gate Counts

| Component | External round (×8) | Internal round (×22) |
|---|---|---|
| S-box | 24 × 3K = 72K gates | 1 × 3K = 3K gates |
| MDS | M4 blocks (~12K gates) | diag + 11ᵀ (~24K gates) |
| Round constants | 24 × 100 = 2.4K | 1 × 100 = 0.1K |
| **Subtotal** | **~86K gates** | **~27K gates** |

**Total core (pipelined):** 8 × 86K + 22 × 27K = 688K + 594K ≈ **1,282K gates (Width 24)**. Standard Width 16: ~730K gates. Overhead: +76% core logic.

### A.5 Pipeline Design Options

Assuming 60M gate die, 50% allocated to Poseidon2 cores (30M gates). Each permutation produces 3 PoW tickets:

| Pipeline style | Gates/core | Cores | Perm/sec/core | Effective hash/sec |
|---|---|---|---|---|
| Folded (×1) | 86K | 348 | 33M | **35G** |
| 5-stage | 300K | 100 | 167M | **50G** |
| 10-stage | 550K | 54 | 333M | **54G** |
| Full (×30) | 1.4M | 21 | 1G | **63G** |

### A.6 SRAM Bandwidth and Throughput Allocation

Each Poseidon2 Merkle hash requires 96 bytes (32 read left + 32 read right + 32 write parent).

$$\text{SRAM bandwidth} \approx 200 \text{ GB/s}$$
$$\text{STARK hash throughput} = 200\text{G}/96 \approx 2.08\text{G hash/sec}$$
$$\text{Total Poseidon2 throughput} \approx 21\text{G perm/sec (21 cores @ 1 GHz)}$$
$$f_{sym} = 2.08/21 \approx 9.9\%$$

| Metric | Value | Note |
|---|---|---|
| Hardware STARK fraction ($f$) | ~10% | SRAM-bandwidth limited |
| Hardware PoW fraction | ~90% | Fills idle Poseidon2 cycles |
| $U$ (usefulness) | ≈ 67% | $t_0/t = 16/24$ |
| STARK proofs/sec | ~260 | 2.08G / 8M hashes per proof |
| PoW hashrate | ~63G effective | 21G perm/sec × 3 tickets |
| $U_{avg}$ | ≈ 6.7% | $f \times U = 0.10 \times 0.67$ |

$U_{avg}$ scales with memory bandwidth:

| Memory technology | Bandwidth | $f$ | $U_{avg}$ | Proofs/sec |
|---|---|---|---|---|
| SRAM 32 MB | 200 GB/s | ~10% | ~6.7% | ~260 |
| SRAM 64 MB | 400 GB/s | ~20% | ~13% | ~520 |
| HBM3 8 GB | 1.2 TB/s | ~60% | ~40% | ~1,560 |
| HBM3E 16 GB | 2.4 TB/s | ~100%† | ~67% | ~2,625† |

†At 2.4 TB/s, SRAM exceeds compute capacity; STARK fraction saturates at 100%.

### A.7 Die Area: kHeavyHash vs Poseidon2-PoW

| Metric | kHeavyHash ASIC | Poseidon2-PoW (Width 24) |
|---|---|---|
| Core area | ~150K gates | ~1.4M gates |
| Cores (60M gate die) | ~380 (95% utilized) | ~21 (50% allocated) |
| Throughput per core | ~1G/s | ~1G perm/s → 3G eff/s |
| Total chip hashrate | ~380G/s | ~63G/s effective |
| ZK proof capability | None | ~260 proofs/sec |

Poseidon2 has ~6× lower PoW hashrate per die. **This is absorbed by difficulty adjustment**—all miners use the same hash function.

### A.8 M31 vs Goldilocks

| Metric | Goldilocks | M31 |
|---|---|---|
| Element size | 64 bits | 31 bits |
| Multiplier gates | ~3,500 | ~1,000 |
| Poseidon2 width (ext.) | 12 | 24 |
| Hash output | 256 bits | 248 bits |
| STARK ecosystem | Plonky2/Plonky3 | **Stwo (potential Kaspa choice)** |

**M31 is the natural choice** if Kaspa adopts Stwo.

---

## Appendix B: GPU Validation

GPUs can freely allocate compute between ZK and PoW in software—the ZK:PoW ratio is a scheduling parameter. The ASIC-specific claim of simultaneous pipeline execution (§5.6) cannot be validated on GPU. What GPU validates: (1) STARK computation produces PoW tickets as a byproduct, and (2) no measurable throughput difference between input sources.

We implement a complete Width-24 Poseidon2 Circle STARK prover in CUDA—iCFFT/CFFT, Merkle tree commitment, constraint quotient, Fiat-Shamir, and FRI fold—as a single `stark.cu` file (~1600 lines). All Poseidon2 parameters match the paper specification: W=24, Rf=8, Rp=22, M31, compression function mode.

**Test environment:** NVIDIA GeForce RTX 4070 (46 SMs, 5483 CUDA cores, 1920 MHz base / 2475 MHz boost, 12 GB GDDR6X, 504 GB/s).

### B.1 PoW Tickets as a Byproduct of ZK

**Claim (§5.1).** Every Poseidon2 Merkle hash in the STARK simultaneously advances the ZK proof and produces 3 PoW tickets. A standard Width-16 prover produces 0 tickets.

**Method.** Run the GPU STARK prover for trace sizes $2^8$ through $2^{22}$. Count Merkle permutations per proof. Each produces 3 tickets.

| $k$ | Trace | STARK (ms) | Merkle perms | Proofs/s | PoW tickets/s |
|---|---|---|---|---|---|
| 8 | 256 | 2.30 | 374 | 434.8 | 487.8K |
| 10 | 1,024 | 3.19 | 1.5K | 313.7 | 1.43M |
| 12 | 4,096 | 4.20 | 6.1K | 238.2 | 4.38M |
| 14 | 16,384 | 5.75 | 24.6K | 173.8 | 12.80M |
| 16 | 65,536 | 8.41 | 98.3K | 118.9 | 35.06M |
| 18 | 262,144 | 13.65 | 393.2K | 73.3 | 86.41M |
| 20 | 1,048,576 | 34.60 | 1.57M | 28.9 | 136.39M |
| 22 | 4,194,304 | 163.48 | 6.29M | 6.1 | 115.46M |

Peak throughput: 136.39M PoW tickets/s at $k = 20$. At $k = 22$, STARK overhead (NTT, constraint evaluation) dominates.

### B.2 Input Independence

**Claim (§5.3).** No meaningful throughput difference between random nonce input (Pure PoW) and structured Merkle input (Symbiotic mode).

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

**Interpretation.** The 0.7% gap is GPU global memory I/O overhead, not input-dependent computation. Poseidon2's 30-round arithmetic dominates. Execution order has no measurable effect. On ASIC (SRAM latency ~1 cycle), this gap is expected to vanish.

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

[13] "Claiming bounties on small scale Poseidon and Poseidon2 instances using resultant-based algebraic attacks," IACR ePrint 2026/150, 2026.

[14] arXiv:2510.09729 (ZK-based PoW with proof-level lottery).
