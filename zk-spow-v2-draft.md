# ZK-SPoW: ZK-Symbiotic Proof of Work

**Anonymous** (Preprint. Work in progress.)

February 2026 — Version 2.0 (Draft)

---

## Abstract

Proof-of-work (PoW) blockchains expend energy solely for network security. Proof of Useful Work (PoUW) attempts to reclaim this cost—a direction Ball et al. [1] show faces fundamental deployment constraints. ZK-based approaches [14]—using STARK proof generation as useful work—are a natural candidate, but face two issues: stateful, multi-phase STARK proving yields trial intervals of tens of milliseconds to seconds (non-memoryless); and losing miners' proofs are discarded, so effective usefulness under difficulty adjustment is no better than pure PoW.

**ZK-SPoW** (ZK-Symbiotic Proof of Work) inverts the PoUW relationship: instead of making PoW computation useful, useful ZK computation (STARK Merkle hashing) naturally produces PoW tickets as a computational byproduct. This inversion resolves the non-memoryless problem—under the pseudorandom permutation (PRP) assumption, each Poseidon2 permutation within the STARK is computationally indistinguishable from an independent Bernoulli trial at nanosecond granularity, rather than proof-level intervals of tens of milliseconds to seconds. It also eliminates proof waste: losing miners' ZK computation remains useful regardless of PoW outcome. Header staleness is bounded by one Merkle commitment phase (~3 ms on GPU (measured), <1 ms on ASIC (projected)).

We instantiate with Width-24 Poseidon2 over M31 ($p = 2^{31}-1$): three PoW tickets per permutation, per-permutation usefulness $U = 100\%$ during STARK proving (proof-level: $\sim 1/N$; pure PoW: $0\%$). On ASIC, the Poseidon2 pipeline is compute-bound while STARK memory access is SRAM-limited, so the fraction of cycles in Symbiotic mode is $f_{sym} \approx 10\%$ (SRAM) to $\sim 60\%$ (HBM3), with remaining cycles filling Pure PoW at $U = 0\%$ (§5.5). Projected zero switching overhead between modes. Security claims assume final Poseidon2 production round constants; the current Stwo implementation uses placeholder values.

---

## 1. Introduction

PoW blockchains waste energy on security-only computation (§1.1), yet replacing it with useful work requires preserving memoryless block discovery (§1.2)—a property fundamentally at odds with stateful computation. Prior PoUW approaches face deployment constraints in this direction (§1.3). ZK-SPoW inverts the relationship, extracting memoryless PoW from useful STARK computation at the permutation level (§1.4).

### 1.1 The PoW Energy Problem

Traditional PoW schemes produce no output beyond network security. The entire energy expenditure is justified solely by the security guarantees it provides.

### 1.2 The Memoryless Requirement in Proof of Work

A PoW scheme is *progress-free* (or *memoryless*) if the probability of finding a valid block on any trial is independent of all previous trials. This property is fundamental:

1. **Fairness.** Progress-based mining gives miners with more accumulated work a higher instantaneous success probability, undermining proportional-hashrate fairness.
2. **Poisson block arrival.** Independent Bernoulli trials yield a Poisson block arrival process—a prerequisite for Nakamoto-style and DAG-based consensus security proofs.
3. **Difficulty adjustment.** Memoryless trials give expected block time $1/(N \cdot H_{rate} \cdot q)$. Progress introduces state-dependent variance that breaks difficulty estimation.

SHA-256 (Bitcoin) and kHeavyHash (Kaspa) are memoryless by construction: each hash evaluation is independent. The challenge: useful computation (STARK proving, optimization) is inherently stateful and progressive.

### 1.3 The PoUW Paradox and Its Inversion

Ball et al. [1] formalize **Proof of Useful Work (PoUW)** as a PoW scheme where the mining computation simultaneously produces useful output. Their strict definition requires: (1) the PoW computation itself produces useful output, (2) the verifier can confirm the usefulness, and (3) the useful output is bound to the PoW evidence.

The fundamental tension: PoW requires random exploration (nonce grinding), while useful computation requires specific, deterministic work. Prior PoUW constructions [1, 7, 8, 9] achieve provable security for specific problem classes, but require pre-hashing, SNARGs, or domain-specific verification that limits practical deployment in high-throughput blockchains.

**ZK-SPoW reverses this direction.** Instead of making PoW results useful, we start from useful computation (STARK proof generation) and observe that PoW tickets emerge as a natural mathematical byproduct:

> **Conventional PoUW:** PoW computation → try to make results useful → fundamental constraints [1]
>
> **ZK-SPoW:** Useful ZK computation → PoW tickets as mathematical byproduct → no contradiction

More precisely, ZK-SPoW is a **hardware-symbiotic architecture**: the same Poseidon2 permutation call simultaneously advances a STARK proof and produces PoW tickets. The STARK does not *require* PoW, and the PoW does not *require* the STARK—they share hardware and computation as an amortized optimization. This differs from Ball et al.'s strict PoUW definition: **blocks contain only the PoW hash, not the STARK proof** (§4.4). The current design does not enforce useful computation at the consensus layer. However, the operating mode *is* distinguishable—via nonce format conventions and mempool STARK proof correlation (§4.4)—providing a path to verifiable mode identification if the protocol chooses to enforce it. The binding is architectural (same permutation), observable (STARK proof evidence in mempool), and economic (ZK proof revenue). See §8.3 for a detailed comparison.

**Definition (ZK-SPoW).** A PoW scheme where the hash function is a width-extended Poseidon2 compression function operating on STARK Merkle data, such that every permutation simultaneously advances a ZK proof and produces PoW tickets.

The mechanism: STARK proof generation requires millions of Merkle hashes. Each Width-24 Poseidon2 Merkle hash takes $(left\_child, right\_child, header\_digest)$ as input and produces $(pow\_ticket_0, pow\_ticket_1, pow\_ticket_2)$ as output, where $pow\_ticket_0 = merkle\_parent$ simultaneously advances the ZK proof. All three output regions are checked against the difficulty target. The miner cannot choose the Merkle inputs—they are determined by the STARK computation.

**Definition (Usefulness).**

$$U = \frac{\text{ZK-contributing trials}}{\text{total mining trials}}$$

- **Continuous proving**: Every Poseidon2 permutation advances a ZK proof → $U = 100\%$. ZK work remains useful regardless of PoW outcome.
- **With idle gaps**: Pure PoW fallback trials (no pending ZK work) are wasted → $U$ decreases proportionally.

The per-permutation data overhead is 8/24 ≈ 33% (header digest occupying 8 of 24 state elements); this is the cost of PoW integration, not a usefulness loss.

**System-level usefulness.** $U$ is a per-permutation metric. On ASIC, the Poseidon2 pipeline is compute-bound while STARK Merkle hashing is SRAM-bandwidth-limited, so only a fraction $f_{sym}$ of Poseidon2 cycles execute in Symbiotic mode. The system-level time-averaged usefulness is $U_{sys} = f_{sym} \times U$, ranging from ~10% (SRAM) to ~60% (HBM3) depending on memory configuration (§5.5, Appendix A).

### 1.4 Our Solution: Permutation-Level PoW Extraction

The central insight of ZK-SPoW is that **non-memoryless computation can produce memoryless PoW**, provided the PoW granularity is finer than the computation's internal state.

A STARK proof is non-memoryless: it consists of trace generation, polynomial commitment (Merkle trees), Fiat-Shamir challenges, and FRI folding—each phase depends on the previous one. A miner who has completed 80% of a proof has "progressed" relative to one who has completed 20%.

However, ZK-SPoW does not use proof completion as the PoW event. Instead, **each individual Poseidon2 permutation within the STARK is an independent PoW trial.** Under the PRP assumption on Poseidon2:

- Each permutation output is pseudorandom, regardless of input structure
- The Merkle tree's feedback structure (parent outputs become child inputs at the next level) does not create exploitable correlation
- Each PoW trial takes nanoseconds (one permutation), not tens of milliseconds to seconds (one proof)

**Result:** ZK-SPoW is computationally progress-free (Definition 2, §2.1) despite embedding PoW within a stateful computation. Block discovery is computationally indistinguishable from a Poisson process under the PRP assumption, preserving the security assumptions of Nakamoto-style and DAG-based consensus.

This resolves the fundamental tension between useful computation and memoryless PoW—not by making the computation memoryless, but by extracting PoW at a granularity where the PRP assumption guarantees independence.

### 1.5 Contributions

1. **Memoryless PoW from non-memoryless computation.** We formalize *computational* progress-freedom for PoW schemes embedded in stateful computations, prove that ZK-SPoW achieves it under the PRP assumption on Poseidon2 via a real–ideal reduction (Theorem 1), and bound header staleness to one Merkle commitment phase (§2).

2. **Hardware-symbiotic architecture with observable binding.** We formalize ZK-Symbiotic Proof of Work, where useful ZK computation shares hardware with PoW mining—the same Poseidon2 permutation simultaneously advances a STARK proof and produces PoW tickets. The operating mode is distinguishable via nonce format and mempool proof correlation, though not enforced at consensus (§1.3, §4.4, §8.3).

3. **Width-24 Poseidon2 parameterization.** We specify Width-24 Poseidon2 over M31 in compression function mode (1 permutation per Merkle hash, vs 2 in sponge mode) and verify its security parameters: $R_p = 22$ internal rounds for 128-bit security with $d = 5$ (§6.2).

4. **Complementary bottleneck architecture.** We demonstrate that PoW mining (compute-bound) and STARK proof generation (memory-bound) share Poseidon2 hardware with projected zero-cycle switching overhead, and provide ASIC architecture estimates for a 7 nm implementation (§5, Appendix A).

5. **GPU empirical validation.** We implement a complete Width-24 Poseidon2 Circle STARK prover on GPU and validate: (a) STARK Merkle hashes produce PoW tickets as a computational byproduct (Appendix C.1); (b) no meaningful throughput difference between random nonce and structured Merkle input sources ($99.3\% \pm 0.3\%$, 10 runs, alternating order; Appendix C.2).

---

## 2. Memoryless PoW from Non-Memoryless Computation

This section formalizes the core theoretical contribution: how ZK-SPoW extracts progress-free PoW from the inherently stateful STARK proving process.

### 2.1 Progress-Free Property

**Definition 1 (Progress-Free PoW).** A PoW scheme with trial sequence $\tau_1, \tau_2, \ldots$ is *progress-free* if for all $n \geq 1$ and all outcomes $o_1, \ldots, o_n$ of the first $n$ trials:

$$P(\tau_{n+1} \text{ succeeds} \mid o_1, \ldots, o_n) = P(\tau_{n+1} \text{ succeeds}) = q$$

where $q$ is a fixed parameter determined by the difficulty target $T$. Equivalently, the success events form an i.i.d. Bernoulli($q$) sequence. Progress-free trials at rate $R = N \cdot H_{rate}$ yield Poisson block arrivals with rate $\lambda = Rq$—a prerequisite for Nakamoto-style and DAG-based consensus security proofs.

SHA-256 and kHeavyHash achieve Definition 1 information-theoretically under the random oracle model. For computational primitives like Poseidon2, the random oracle model does not apply. We introduce a computational relaxation:

**Definition 2 (Computationally Progress-Free PoW).** A PoW scheme is *computationally progress-free* with security parameter $\kappa$ if no probabilistic polynomial-time (PPT) adversary can distinguish the joint distribution of trial outcomes $(o_1, \ldots, o_n)$ from an i.i.d. Bernoulli($q$) sequence with non-negligible advantage in $\kappa$.

STARK proof generation is not even computationally progress-free at the proof level: it is stateful (trace → NTT → Merkle → Fiat-Shamir → FRI), and if proof completion is the PoW event, a miner at 80% completion has higher conditional success probability than one at 10%—a distinction any observer can make without breaking any cryptographic assumption.

### 2.2 Permutation-Level Independence

ZK-SPoW resolves this by defining the PoW trial at the individual permutation level, not the proof level.

**Setup.** Let $\pi: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$ be the Width-24 Poseidon2 permutation. In a STARK Merkle tree commitment phase, the prover evaluates $\pi$ on inputs $x_1, \ldots, x_P$ where:
- $x_j = (n_{L,j} \| n_{R,j} \| h_H)$ for Merkle node $j$
- $n_{L,j}, n_{R,j} \in \mathbb{F}_p^8$ are the left and right child hashes
- $h_H \in \mathbb{F}_p^8$ is the (fixed) header digest
- $P = \sum_{i=0}^{m}(N_i - 1)$ is the total number of internal Merkle nodes across all commitment phases

Each evaluation produces three PoW tickets: $\text{ticket}_{j,k} = \pi(x_j)[8k:8k+7]$ for $k \in \{0, 1, 2\}$.

**Assumption 1 (Input Distinctness).** The inputs $x_1, \ldots, x_P$ are pairwise distinct. In Circle STARK, Merkle tree leaves are evaluations of trace polynomials at $2^\ell$ distinct domain points; internal nodes are determined by their unique children. A collision at level $\ell > 0$ requires $\pi(x_j)[0:7] = \pi(x_{j'})[0:7]$ for distinct $x_j, x_{j'}$—probability at most $\binom{P}{2} \cdot p^{-8} \approx P^2/2^{248}$, negligible for $P \leq 10^7$. Note: this collision bound itself relies on output uniformity under PRP; Assumption 1 is therefore conditioned on the PRP assumption rather than independent of it. The conjunction "PRP + Assumption 1" is not circular: PRP guarantees pseudorandom outputs, which in turn make input collisions at higher tree levels negligible.

**Remark (Malicious input injection).** Since the protocol does not verify STARK internals (§4.4), a miner could feed arbitrary—including intentionally colliding—inputs to the Poseidon2 permutation instead of genuine Merkle data. This does not undermine PoW security for two reasons: (1) **PRP holds for all inputs.** Under the PRP assumption, $\pi(x)$ is pseudorandom for *any* $x$, including adversarially chosen ones. Even if a miner feeds the same input $x$ repeatedly, each evaluation of $\pi(x)$ produces the same output—the miner gains no advantage over trying distinct inputs, which under PRP each yield independent pseudorandom outputs. Repeating inputs strictly reduces the miner's ticket count. (2) **Self-harming only.** A miner who violates Assumption 1 by using colliding inputs produces fewer *distinct* Poseidon2 evaluations, reducing their own PoW ticket count without affecting other miners. The violation is detectable only by the miner themselves and has no network-level consequence. Assumption 1 is therefore a *rational behavior assumption*, not a protocol enforcement requirement: any violation is strictly self-penalizing.

**Theorem 1 (Permutation-Level Independence).** *Under the PRP assumption on Poseidon2 with security parameter $\kappa$ and Assumption 1, the joint distribution of PoW success events*

$$E_j = \{\exists\, k \in \{0,1,2\} : \text{ticket}_{j,k} < T\}, \quad j = 1, \ldots, P$$

*is computationally indistinguishable from an i.i.d. Bernoulli($q$) sequence, with $q = 1 - (1-p_t)^3$, $p_t = T/2^{248}$. That is, for any PPT distinguisher $\mathcal{D}$:*

$$|\Pr[\mathcal{D}(E_1, \ldots, E_P) = 1] - \Pr[\mathcal{D}(B_1, \ldots, B_P) = 1]| \leq \mathrm{negl}(\kappa)$$

*where $B_1, \ldots, B_P \stackrel{i.i.d.}{\sim} \mathrm{Bernoulli}(q)$.*

*Proof.* We proceed via a real–ideal argument.

**Real world.** The prover evaluates $\pi(x_1), \ldots, \pi(x_P)$ on distinct inputs (Assumption 1) and derives events $E_1, \ldots, E_P$.

**Ideal world.** Replace $\pi$ with a truly random function $f: \mathbb{F}_p^{24} \to \mathbb{F}_p^{24}$. For distinct inputs, the outputs $f(x_1), \ldots, f(x_P)$ are mutually independent and each uniform over $\mathbb{F}_p^{24}$. In this world:

**(a) Intra-permutation independence.** Each $f(x_j)$ is uniform over $\mathbb{F}_p^{24} = \mathbb{F}_p^8 \times \mathbb{F}_p^8 \times \mathbb{F}_p^8$. The three 8-element projections are mutually independent (product probability space), each uniform over $\mathbb{F}_p^8$. The event $\{\text{ticket}_{j,k} < T\}$ has probability $p_t = T/2^{248}$. Hence $P(E_j) = 1-(1-p_t)^3 = q$.

**(b) Inter-permutation independence.** For distinct inputs, the outputs $f(x_j)$ are independent, so the events $E_j$ are independent.

Thus in the ideal world, $\{E_j\}_{j=1}^P$ are i.i.d. Bernoulli($q$).

**Indistinguishability.** By the PRP assumption, no PPT adversary can distinguish $(\pi(x_1), \ldots, \pi(x_P))$ from $(f(x_1), \ldots, f(x_P))$ for distinct inputs. Since each $E_j$ is a deterministic function of the corresponding output, any PPT distinguisher for the event sequences can be composed with the event-extraction function to yield a PPT distinguisher for the outputs—contradicting the PRP assumption. ∎

**Remark (PRP–PRF switching).** A PRP is a permutation: distinct inputs yield distinct outputs, introducing a negligible anti-correlation absent in a random function. The statistical distance between a random permutation and a random function on $P$ queries is at most $\binom{P}{2}/|\mathbb{F}_p^{24}| \approx P^2/2^{744}$, negligible for any practical $P$. This is the standard PRP–PRF switching lemma; the proof above absorbs this cost into $\mathrm{negl}(\kappa)$.

**Remark on structured inputs.** Merkle tree feedback (parent outputs become child inputs) and Fiat-Shamir cascades (trace selection affects FRI trees) create structured, non-i.i.d. permutation inputs. This does not violate Theorem 1: the PRP guarantee holds for *any* sequence of distinct inputs, regardless of how they are generated—any weakness for structured inputs would constitute a Poseidon2 break. The total ticket count $P$ is trace-independent (§6.3).

**Corollary 1 (Computational Progress-Freedom).** *Under the PRP assumption, ZK-SPoW is computationally progress-free (Definition 2). Consequently, block discovery is computationally indistinguishable from a Poisson process.*

*Proof.* By Theorem 1, the PoW events are computationally indistinguishable from i.i.d. Bernoulli($q$). Any PPT consensus adversary $\mathcal{A}$ that gains non-negligible advantage (e.g., disproportionate mining share) by exploiting non-Poisson block arrivals must implicitly distinguish the real trial sequence from the ideal i.i.d. sequence. Formally: given $\mathcal{A}$ with advantage $\epsilon(\kappa)$ under real events but advantage $0$ under i.i.d. events, we construct a distinguisher $\mathcal{D}$ with advantage $\geq \epsilon(\kappa)$ by running $\mathcal{A}$ on the challenge sequence—contradicting Theorem 1 if $\epsilon$ is non-negligible.

**Side information.** A miner observes not only PoW outcomes $(E_1, \ldots, E_P)$ but also STARK execution state (e.g., which Merkle tree level is being computed, NTT completion status). This side information does not help predict future PoW outcomes: under PRP, the output of $\pi(x_j)$ is pseudorandom regardless of how $x_j$ was generated or what the miner knows about the computation's progress. Therefore the adversary's enlarged action space (e.g., deciding whether to abandon a phase upon new block arrival) does not invalidate the reduction—any strategy conditioned on STARK progress that yields consensus advantage can still be converted into a PRP distinguisher, since the PoW outcomes it ultimately exploits remain computationally indistinguishable from i.i.d. ∎

**Remark (Sufficiency for consensus security).** Nakamoto-style security proofs [Garay et al., 2015] and DAG-based security proofs [5] rely on block arrival being a Poisson process in two key ways: (1) bounding selfish-mining advantage via the memoryless property of inter-arrival times, and (2) difficulty adjustment convergence via the law of large numbers on i.i.d. trials. Both arguments are stated for PPT adversaries operating within a computational model. Replacing information-theoretic i.i.d. with computational indistinguishability from i.i.d. preserves these bounds: any PPT adversary achieving better-than-Poisson selfish-mining advantage or difficulty manipulation must distinguish the trial sequence from i.i.d., contradicting Theorem 1. The only scenario where computational progress-freedom is strictly weaker than information-theoretic progress-freedom is if the consensus security proof requires a *superpolynomial-time* argument—which no standard proof does.

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

**DAGKnight tolerance.** At 100 BPS, the theoretical block interval is 10 ms. However, the 10 ms figure assumes zero network propagation delay. In practice, global P2P networks exhibit propagation delays of 50–200 ms (ping RTT), and DAGKnight is specifically designed to tolerate this: its anticone parameter (typically $k \geq 10$) accepts blocks with parent sets many blocks deep, tolerating $\sim 100$+ ms of latency.

**Staleness is the cost of Symbiotic mode.** Traditional PoW (SHA-256, kHeavyHash) has zero staleness: each hash is independent and can reference the latest header. ZK-SPoW's Symbiotic mode fixes $h_H$ for one Merkle commitment phase, introducing staleness of 3.4 ms (GPU) or 0.05 ms (ASIC). This is an inherent cost of embedding PoW within a batched STARK computation—the Merkle tree must be committed with a consistent header digest. The relevant question is whether this cost is material relative to the network propagation delays that DAGKnight already tolerates.

**Quantitative impact (simulation).** We simulate a Poisson block arrival process at 100 BPS with log-normal network propagation delays, measuring anticone size (blocks concurrent with a given block) under varying staleness. $10 \times 60$ s runs per scenario:

| Network delay | No staleness | ASIC (0.05 ms) | GPU (3.4 ms) | GPU $\Delta$ |
|---|---|---|---|---|
| LAN (5 ms) | 1.00 | 1.02 (+1.0%) | 1.68 (+67%) | +0.68 blocks |
| Regional (20 ms) | 4.02 | 4.03 (+0.3%) | 4.70 (+17%) | +0.68 blocks |
| Global (50 ms) | 10.05 | 10.06 (+0.1%) | 10.73 (+6.8%) | +0.68 blocks |
| Global (100 ms) | 20.09 | 20.10 (+0.05%) | 20.77 (+3.4%) | +0.68 blocks |
| Global (200 ms) | 40.13 | 40.14 (+0.02%) | 40.81 (+1.7%) | +0.68 blocks |

The GPU staleness adds a constant ~0.68 blocks to the anticone regardless of network delay. On a realistic global network (50–200 ms propagation), this is a **1.7–6.8% marginal increase** over the baseline anticone of 10–40 blocks—well within DAGKnight's tolerance. ASIC staleness (+0.01 blocks) is unmeasurable. (Simulation source: `analysis/dagknight_staleness_sim.py`.)

**Selfish mining interaction.** An attacker cannot increase a victim's staleness—$\Delta_{stale}$ is determined by the victim's own hardware. An attacker who announces blocks at STARK phase boundaries can force honest miners to restart phases, wasting partial STARK computation. However: (1) the attacker gains no PoW advantage, since the victim's PoW tickets from the stale phase remain valid; (2) the cost to the victim is at most one phase of STARK work (~3.4 ms × $f_{sym}$); (3) the attacker cannot control timing at millisecond precision over a network with 50–200 ms propagation jitter. This "phase disruption" degrades ZK throughput, not PoW security.

**Incentive-compatible behavior.** When a new block arrives mid-phase, a rational miner faces a choice: (1) complete the current phase with the now-stale header, or (2) abandon and restart with the new header. Completing is incentive-compatible: the PoW tickets remain valid (staleness is within DAGKnight tolerance), and the STARK proof progress is preserved. Abandoning wastes the partial phase computation. A miner should update $h_H$ at the next natural phase boundary.

**Trace size tradeoff.** Larger traces ($2^\ell$) produce more useful ZK work per proof but increase the largest Merkle tree size $N_{max}$, raising $\Delta_{stale}$. At $\ell = 20$, $\Delta_{stale} \approx 3.4$ ms (GPU); at $\ell = 22$, $\Delta_{stale} \approx 13.6$ ms, exceeding one block interval at 100 BPS. The protocol should constrain $\ell$ such that $N_{max}/R_{perm}$ remains below the block interval. On ASIC ($R_{perm} \approx 21$ Gperm/s), even $\ell = 22$ yields $\Delta_{stale} \approx 0.2$ ms—well within bounds.

### 2.4 Comparison with Proof-Level Approaches

The key differentiator of ZK-SPoW is the *granularity* at which PoW operates within the ZK computation.

| Property | SHA-256 / kHeavyHash | Proof-Level [14] | Nockchain [11] | **ZK-SPoW** |
|---|---|---|---|---|
| PoW granularity | 1 hash = 1 trial | 1 proof = 1 ticket | Proof → hash | **1 perm = 1 trial** |
| Trial duration | Nanoseconds | Tens of ms–s | Seconds + μs | **Nanoseconds** |
| Progress-free? | **Yes** (info-theoretic) | **No** — sunk cost | **Final hash: yes**; proof phase: no | **Yes** (computational, PRP) |
| Max staleness | 0 (stateless) | Proof duration | Proof duration | **~3 ms (measured)** |
| Losing miners' work | Security only | Proofs discarded | Proofs discarded | **ZK work preserved** |
| Useful work | None | Proof = PoW | Proof then hash | **Perm = PoW + ZK** |
| $U$ (per-perm) | 0% | $\sim 1/N$ | $\sim 1/N$ | **100%** (system: $f_{sym}$-limited, §5.5) |

ZK-SPoW operates at the finest possible granularity—individual permutations (nanoseconds)—achieving computational progress-freedom equivalent to the information-theoretic progress-freedom of traditional hash-based PoW (Definition 2, §2.1). The STARK proof continues regardless of PoW outcomes: losing miners' ZK computation remains useful.

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
| $H$ | Block header (all consensus fields; see §4.4). Not to be confused with per-miner trial rate $H_{rate}$ |
| $h_H$ | Header digest: $\text{PoseidonSponge}(H \text{ excluding nonce}) \in \mathbb{F}_p^k$ |
| $k$ | Header digest element count ($k = 8$ for symmetric I/O and three PoW tickets) |
| $(v_1, v_2)$ | Nonce: $v_1, v_2 \in \mathbb{F}_p^8$ |
| $T$ | Target $\in \mathbb{F}_p^8$ (difficulty-adjusted) |
| $p_t$ | Single-ticket success probability: $p_t = T/2^{248}$ |
| $q$ | Per-permutation success probability: $q = 1 - (1-p_t)^3$ |
| $S$ | Poseidon2 state after permutation, $S \in \mathbb{F}_p^t$ |
| $N$ | Number of miners in the network. Distinguished from $N_i$ (Merkle tree leaf count) by subscript |
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

**Note on ticket granularity: 3 tickets vs 1 ticket.** The 3 × 8-element partition is a protocol convention, not a cryptographic necessity. An alternative design reads all 24 output elements as a single 744-bit value and adjusts the difficulty target accordingly ($q_{single} = T_{744}/2^{744}$ vs $q_{triple} = 1-(1-T_{248}/2^{248})^3 \approx 3 T_{248}/2^{248}$). Both achieve equivalent expected block times via difficulty adjustment. We choose 3 × 248-bit tickets because: (1) **Stwo compatibility:** 8-element output matches the Stwo hash convention, enabling direct reuse of Merkle commitment infrastructure; (2) **Symbiotic mode constraint:** in Symbiotic mode, S[0..7] *must* be the Merkle parent (used by the STARK), so it has a designated role—the 3-ticket design naturally partitions the output into "STARK-functional" and "PoW-only" regions; (3) **Hardware simplicity:** three 248-bit comparators are smaller and faster than one 744-bit comparator. The 3-ticket design is not strictly necessary—a single-ticket design with 744-bit comparison is mathematically equivalent under PRP (Proposition 3, §6.4)—but the engineering tradeoffs favor the 3-ticket partition.

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

The only structural change is the nonce expansion from `u64` (8 bytes) to `[F_p; 16]` (64 bytes, +56 bytes per block). At 100 BPS this adds 5.6 KB/s to network bandwidth—negligible relative to the ~12.5 MB/s baseline (125 KB/block × 100 BPS).

**Block hash vs PoW hash.** Block identity and PoW use separate hash functions. The block hash (e.g., Blake2b-256 for DAG references) is unchanged; only the PoW function is replaced.

**STARK proofs are NOT included in the block.** They are submitted as independent transactions in the mempool, providing economic value to the ZK ecosystem (vProgs fees). This eliminates the +3–5 MB/s bandwidth overhead that would result from mandatory per-block STARK proofs at 100 BPS.

**Mode distinguishability.** Although blocks do not contain STARK proofs, the operating mode (Symbiotic vs Pure PoW) is distinguishable through two independent mechanisms if the protocol chooses to use them:

1. **Nonce format convention.** The protocol can define that Pure PoW nonces must zero-pad a designated region (e.g., `v2[4..7] = 0`). In Symbiotic mode, the corresponding nonce positions contain Merkle child data—Poseidon2 outputs that are zero with probability $1/p^4 \approx 2^{-124}$. A verifier inspects the nonce field in the block header: zeros in the designated region indicate Pure PoW; non-zeros indicate a Symbiotic claim. The cost is a reduction in Pure PoW nonce space from 16 to 12 elements (372 bits)—no practical impact on hashrate.

2. **Mempool proof correlation.** Each STARK proof submitted to the mempool commits to Merkle trees constructed with a specific header digest $h_H$ as a salt in every node (§4.2). The verifier of the STARK proof requires $h_H$ to recompute Merkle opening paths. By matching a proof's $h_H$ against a block header, any node can verify *after the fact* that a given block was mined during Symbiotic computation for that specific header. This requires no protocol changes—the binding evidence is inherent in the STARK proof structure.

Neither mechanism is mandatory for consensus. The current design treats mode distinguishability as optional: the PoW function is valid regardless of mode, and ZK revenue provides the economic incentive for Symbiotic mining. However, if future protocol evolution requires rewarding or mandating useful work, these mechanisms provide a path to verifiable mode identification without modifying the PoW function itself.

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

**Header freshness.** A STARK proof spans multiple Merkle commitment phases—typically $O(10)$. The header digest is fixed per phase; between phases it updates to the current chain tip. Maximum staleness is one Merkle phase—see Theorem 2 (§2.3).

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
|  Switching cost: 0 throughput overhead (combinational MUX, ~300 gates; ≤1 cycle latency) |
|  Hashrate: invariant across all modes                 |
+-------------------------------------------------------+
```

The transition is **per-cycle and linear**, not a discrete mode switch. The pipeline is always full—the ratio is determined by SRAM bandwidth.

### 5.4 Hashrate Invariance

**Proposition 2.** *Total PoW hashrate $\mathcal{H}$ is independent of the operating mode.*

*Argument.* $\mathcal{H} = N_{cores} \times \text{throughput\_per\_core}$. Each core's throughput is 1 hash per pipeline depth cycles (fully pipelined), regardless of input source. The input MUX is combinational logic (~300 gates, 2:1 selection on 16 × 31-bit words).

**MUX critical path impact.** The MUX adds gate delay to the pipeline's first stage. At 7 nm, a 2:1 MUX contributes ~20–30 ps of propagation delay. The Poseidon2 pipeline's critical path is the S-box ($x \mapsto x^5$, requiring two M31 multiplications in series: ~200–300 ps at 7 nm). The MUX delay is <15% of the critical path—absorbable within the pipeline stage's timing margin without reducing clock frequency. If timing closure requires it, the MUX can be retimed into a dedicated pipeline stage at the cost of 1 cycle of latency (not throughput). The "zero cycles overhead" claim refers to throughput (pipelined), not latency; we clarify: **zero throughput overhead, ≤1 cycle latency overhead.**

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

**Sponge mode vs compression function mode.** Poseidon2 [3] provides 128-bit security in sponge mode via an indifferentiability argument: with capacity $c = 8$ M31 elements (248 bits), the sponge construction is indifferentiable from a random oracle up to $2^{124}$ queries. ZK-SPoW operates in compression function mode (§4.2), where all 24 input elements are visible—there is no hidden capacity. Security in this mode relies on different, more direct properties of the permutation $\pi$:

- **Collision resistance:** Finding $x \neq x'$ with $\pi(x)[0:7] = \pi(x')[0:7]$ requires $\Omega(2^{124})$ work (birthday bound on 248-bit output). This secures STARK Merkle binding.
- **Preimage resistance:** Finding $x$ with $\pi(x)[0:7] < T$ requires $\Omega(1/p_t)$ work. This secures PoW.
- **PRP (pseudorandom permutation):** Outputs are computationally indistinguishable from random, regardless of input structure. This secures progress-freedom (Theorem 1).

These are strictly weaker assumptions than sponge-mode indifferentiability—collision/preimage resistance of a permutation is implied by, but does not require, random oracle behavior. The Poseidon2 literature [2, 3] primarily analyzes the sponge construction, raising a natural question: does the round count $R_p = 22$ assume the existence of hidden capacity elements?

**Round count is capacity-independent.** We verify from Plonky3's source [10] that the round number calculation depends on exactly four parameters: field prime $p$, state width $t$, S-box degree $d$, and security level $M$. Capacity does not appear. The six security constraints (statistical, interpolation, three Gröbner basis variants, and the algebraic bound from [15]) target the *permutation itself*—they bound the number of rounds needed to resist distinguishing attacks against $\pi$, not against a sponge construction built from $\pi$. Capacity $c$ is a separate, orthogonal parameter that determines sponge-mode security as $\min(\text{permutation security}, 2^{c/2})$. For compression function mode, only permutation security applies, and $R_p = 22$ (with margin $R_f += 2$, $R_p \times 1.075$) targets 128-bit permutation security directly.

**Compression function mode risks.** While the round count is sound, operating without capacity has two consequences: (1) the full 744-bit state is visible, giving an attacker more information for algebraic cryptanalysis (e.g., CICO attacks [12] become formulated over the full state rather than a reduced output space); (2) there is no "safety net"—in sponge mode, even a partial permutation break may be absorbed by the capacity, whereas in compression mode any permutation weakness directly impacts security. We consider this acceptable because: (a) the 30-round configuration's algebraic properties (S-box degree growth, MDS diffusion) are mode-independent; (b) the Poseidon2 paper [3] explicitly recommends compression function mode for Merkle trees; (c) any attack on the compression function implies an attack on the sponge with equal or lower complexity (strictly more information available to the attacker).

**Single primitive dependency.**

| Component | Traditional (e.g., Kaspa [6]) | ZK-SPoW |
|---|---|---|
| PoW | kHeavyHash (Blake2b + cSHAKE256) | Poseidon2 |
| STARK | Poseidon2 | Poseidon2 |
| Independence | PoW ≠ STARK | PoW = STARK |

**Correlated failure mode.** A cryptographic break in Poseidon2 compromises both PoW security and STARK validity simultaneously—a qualitatively different risk profile from traditional designs where PoW and STARK use independent primitives. We assess this risk quantitatively:

- **Current margin.** The 30-round configuration ($R_f = 8$, $R_p = 22$) includes a $+2$ external round margin and $\times 1.075$ internal round margin over the minimum required by known attacks. Merz and Rodríguez García [12] improve algebraic CICO attacks by $2^{106}$ for one parameter set but conclude the full-round primitive "does not fall short of its claimed security level." Resultant-based attacks [13] solve instances with $\leq 10$ total rounds—far below the 30-round configuration.
- **Margin erosion trajectory.** The attack improvement of [12] targets the sparse structure of $M_I$, a design feature inherent to Poseidon2. Further advances along this line are plausible. If a future attack reduces security below $2^{128}$ for Width-24/30-round, both PoW and STARK would require parameter updates simultaneously.
- **Fallback strategy.** A Poseidon2 break does not require abandoning ZK-SPoW. The framework is parametric: increasing $R_p$ (more internal rounds) restores security at the cost of throughput and die area. In the extreme case, the chain can revert to Pure PoW mode with a different hash function while maintaining block production—Symbiotic mode is an overlay, not a prerequisite for consensus. A hard fork would be required in either case (as it would for any PoW hash function break).

### 6.2 Width-24 Security

**Collision resistance (STARK binding).** With 248-bit output (8 M31 elements), the birthday bound is $2^{124}$, providing 124-bit collision security.

**STARK soundness.** The header digest acts as a fixed salt in each Merkle hash—determined before proof generation, cannot be chosen adaptively. STARK soundness reduces to collision resistance with a fixed third input—a strictly easier assumption.

**Preimage resistance (PoW security).** The 30-round Poseidon2 permutation (8 external + 22 internal) satisfies this: no known attack reduces preimage search below the generic $2^{248}$ bound.

**$R_p$ for Width-24.** $R_p = 22$ for 128-bit security at $d = 5$ over M31, computed via Plonky3's round number formula [10]. The formula applies six security constraints from [2, 3] (statistical, interpolation, three Gröbner basis variants) plus the algebraic attack bound from [15], then adds margin: $R_f += 2$, $R_p \times 1.075$. Pre-margin optimum: $(R_f, R_p) = (6, 20)$; post-margin: $(8, 22)$. The binding constraint is statistical ($R_{f,1} = 6$, since $M = 128 \leq \lfloor \log_2 p - 2 \rfloor \times (t+1) = 700$). **Crucially, the formula depends only on $(p, t, d, M)$—capacity is not a parameter** (see §6.1). Total rounds: 30 (8 + 22). S-box operations: $t \times R_f + R_p = 24 \times 8 + 22 = 214$ per permutation; in compression function mode (§4.2), this yields 25% fewer S-boxes per Merkle hash than Width-16 sponge (214 vs $2 \times 142 = 284$). Supplementary: $M_I^k$ is invertible for $k = 1..48$ (subspace trail resistance). Algebraic degree after 30 rounds exceeds $2^{69}$, above the $2^{64}$ threshold for 128-bit interpolation security.

**Recent cryptanalysis.** See §6.1 (Current margin) for detailed analysis of [12] and [13].

### 6.3 Trace Grinding Resistance

Trace selection in Symbiotic mode provides zero advantage for PoW mining under the PRP assumption. We summarize the results here; full proofs are in Appendix B.

Under PRP, three invariants hold:

1. **Ticket count invariance.** Total permutations $P = \sum_{i=0}^{m}(N_i - 1)$ is protocol-determined and trace-independent. The miner cannot increase ticket count by selecting a different trace (Appendix B.2).

2. **Distribution invariance.** The number of valid tickets follows $V \sim \text{Binomial}(P, q)$, where both $P$ and $q = 1 - (1-p_t)^3$ are trace-independent. The *entire distribution* is invariant under trace selection (Appendix B.3).

3. **Grinding is dominated.** Header digest grinding is equivalent to nonce grinding at equal cost (Appendix B.4). Multi-trial grinding ($K$ traces, best-of-$K$) is strictly dominated by $K \times P$ pure PoW permutations (Appendix B.5). Merkle tree feedback does not violate PRP (Appendix B.6).

### 6.4 Triple-Ticket Independence

The three PoW tickets from a single permutation are deterministically linked but computationally independent under PRP.

**Proposition 3.** *Under the PRP assumption, the joint distribution of $\text{ticket}_0 = S[0..7]$, $\text{ticket}_1 = S[8..15]$, and $\text{ticket}_2 = S[16..23]$ is computationally indistinguishable from three mutually independent uniform samples over $\mathbb{F}_p^8$, each with success probability $p_t = T/2^{248}$. The per-permutation success probability is $q = 1 - (1-p_t)^3$.*

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

**Equilibrium condition.** Let $\alpha \approx 1.1\text{–}1.2$ be the Pure PoW hash/watt advantage. ZK-SPoW ASICs survive in equilibrium when:

$$Z \cdot F \geq (\alpha - 1) \cdot \frac{\mathcal{H}_{spow}}{\mathcal{H}_{total}} \cdot B$$

i.e., per-miner ZK revenue exceeds the ~10–20% mining revenue gap. Let $D$ be the network-wide ZK proof demand (proofs/s) and $N_{spow}$ the number of ZK-SPoW miners, each with throughput $Z$. In steady state $N_{spow} \cdot Z \leq D$ (excess proving capacity earns no revenue). The critical demand threshold is:

$$D_{min} = N_{spow} \cdot Z \cdot \mathbb{1}[Z \cdot F \geq (\alpha-1) \cdot B/N]$$

**Demand scarcity risk.** If $D \ll N_{spow} \cdot Z$, most ZK-SPoW cycles execute in Pure PoW mode ($U_{sys} \approx 0\%$), and ZK revenue per miner drops to $F \cdot D / N_{spow}$. When this falls below $(\alpha-1) \cdot B/N$, Pure PoW ASICs dominate, and rational miners abandon ZK-SPoW hardware—a collapse scenario. This is the fundamental macro-economic risk: **ZK-SPoW's usefulness depends on external ZK demand matching a significant fraction of the network's proving capacity.**

**Order-of-magnitude estimate.** A Kaspa-scale network with $N_{spow} = 10{,}000$ ASICs at $Z = 260$ proofs/s (Appendix A) produces ~2.6M proofs/s of aggregate capacity. Current blockchain ZK demand (rollup proofs, zkVMs) is orders of magnitude below this. ZK-SPoW viability therefore requires either: (1) a sustained, large-scale ZK proof market (e.g., universal zkVM proving as a service), or (2) protocol-level subsidization of ZK proving alongside block rewards. Neither condition is guaranteed.

**PoW security without ZK demand.** When $Z \cdot F = 0$, the ASIC operates as a conventional PoW miner with a hash/watt disadvantage due to die area overhead. Difficulty adjustment absorbs this. The overhead is the cost of optionality—it purchases the ability to capture ZK revenue when the market emerges. Network security is never compromised by insufficient ZK demand; only the economic advantage of ZK-SPoW hardware is affected.

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

**Memorylessness nuance.** Nockchain's final hash-after-proof step is itself memoryless: the hash evaluation is independent of previous hashes. However, the proof computation phase preceding it is progressive (a miner closer to proof completion has higher conditional probability of reaching the hash step sooner). The overall block arrival process is therefore a *renewal process* where inter-arrival times are the sum of a progressive proof phase and a memoryless hash phase—not a pure Poisson process. The practical impact depends on proof duration relative to block interval: if proofs take seconds and blocks arrive every few seconds, the progressive component is significant. ZK-SPoW avoids this entirely by operating at permutation granularity (nanoseconds).

### 8.3 Relationship to Ball et al.

Ball et al. [1] formalize PoUW in the direction **PoW → useful output**. ZK-SPoW operates inversely: **useful computation → PoW output**.

| Ball et al. criterion | PoW → useful (their framework) | Useful → PoW (ZK-SPoW) |
|---|---|---|
| PoW produces useful output | Partial | **Yes**: every Symbiotic permutation advances a STARK proof |
| Verifier confirms usefulness | Not enforced | **Distinguishable but not enforced**: mode is identifiable via nonce format and mempool proof correlation (§4.4), but consensus does not require it |
| Useful output bound to PoW | Not enforced | **Architectural + observable**: same permutation produces both; STARK proof in mempool contains `h_H` binding evidence (§4.4) |

**Mode distinguishability without enforcement.** Blocks do not contain STARK proofs, but the operating mode is *not* indistinguishable. Two mechanisms allow mode identification (§4.4): (1) a nonce format convention that makes Pure PoW blocks identifiable from the header alone, and (2) mempool STARK proofs whose Merkle commitments contain the block's `h_H` as a salt, providing after-the-fact cryptographic evidence of Symbiotic mining. The current design treats these as optional—consensus validity depends only on the PoW hash, and economic incentives (ZK proof revenue) drive Symbiotic adoption. However, these mechanisms mean the protocol *could* enforce or reward useful work in future iterations without modifying the PoW function.

Ball et al.'s hardness results constrain the PoW → useful direction. ZK-SPoW sidesteps these constraints by reversing the direction: useful computation happens to produce PoW-valid outputs. The binding is architectural (same permutation call) and observable (STARK proof evidence), though not currently enforced at the consensus layer. A miner *could* run Pure PoW mode exclusively, forgoing ZK revenue. ZK-SPoW is best characterized as a **hardware-symbiotic architecture** with a **path to verifiable PoUW**—the evidence infrastructure exists; enforcement is a policy decision.

### 8.4 Memorylessness Comparison

See §2.4 (Table 1) for the full comparison. ZK-SPoW is the only ZK-based PoW scheme that achieves computational progress-freedom (Definition 2) equivalent to the information-theoretic progress-freedom of traditional hash-based PoW, while simultaneously producing useful computation. The cost: the PRP assumption on Poseidon2 replaces the random oracle assumption on traditional hash functions.

---

## 9. Open Questions

1. **Compression function mode: dedicated cryptanalysis.** We verify that Plonky3's $R_p = 22$ formula targets permutation-level security independent of capacity (§6.1, §6.2). However, the Poseidon2 cryptanalysis literature [2, 3, 12, 15] primarily evaluates sponge-mode usage. A dedicated analysis of algebraic attack complexity against the bare Width-24 permutation in compression function mode—specifically, CICO attacks [12] over the full 744-bit state and Gröbner basis complexity without capacity reduction—would strengthen the 128-bit security claim. Until such analysis, the claim should be considered well-founded but not independently verified for this specific mode.

2. **Memoryless validation.** Empirical verification of Poisson block inter-arrival times in a test network running ZK-SPoW would complement the theoretical analysis of §2.

3. **Output quality under production constants.** The GPU experiments (Appendix C) use placeholder round constants. Statistical testing of output pseudorandomness (e.g., NIST SP 800-22) under production Poseidon2 constants would validate that the PRP assumption holds in practice, not only in theory.

4. **ZK demand viability.** §7.2 derives the equilibrium condition and demand scarcity risk. The critical open question is whether sustained ZK proof demand at the required scale ($\gtrsim 10^5$ proofs/s network-wide) will materialize. A dynamic simulation of ZK-SPoW vs Pure PoW ASIC competition under stochastic ZK demand—including miner entry/exit dynamics and difficulty adjustment feedback—would quantify the collapse threshold more precisely. The source of demand (who buys proofs, and why ZK-SPoW miners are preferred over dedicated proving services) remains unaddressed.

5. **Full DAGKnight consensus simulation.** §2.3 simulates anticone size impact (1.7–6.8% marginal increase for GPU, unmeasurable for ASIC). A full DAGKnight consensus simulation—modeling blue set selection, confirmation times, and throughput under heterogeneous staleness—would provide stronger quantitative guarantees. The "phase disruption" attack (§2.3) warrants game-theoretic analysis, though its practical feasibility is limited by network propagation jitter (50–200 ms).

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

**NoC area and power.** Delivering 200 GB/s of SRAM bandwidth to 21 Poseidon2 cores requires a Network-on-Chip (NoC) or crossbar interconnect. At 7 nm, a 200 GB/s mesh NoC (e.g., ring or 2D mesh topology) consumes approximately 1–3% of die area and 0.5–1.5 W, depending on topology and wire length. The 5% Control & I/O budget (§A.1) is tight: it must accommodate the NoC, PoW/STARK schedulers, difficulty comparison logic, and network interface. A more realistic breakdown may require 7–10% for Control, I/O, and interconnect, reducing Poseidon2 core allocation from 50% to ~47% (20 cores instead of 21). This is a ~5% hashrate reduction—non-negligible but does not qualitatively change the Pareto analysis (§7). The die allocation in §A.1 should be considered approximate; detailed physical design is beyond this paper's scope.

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

Under PRP, for any distinct inputs $x_1, \ldots, x_P$ to the permutation, the outputs are jointly pseudorandom. Input distinctness holds with overwhelming probability (Assumption 1, §2.2).

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

$$P(A < T) = p_t = T/2^{248}$$
$$P(B < T) = p_t$$
$$P(C < T) = p_t$$
$$P(A < T \wedge B < T \wedge C < T) = p_t^3$$
$$P(A < T \vee B < T \vee C < T) = 1 - (1-p_t)^3 = 3p_t - 3p_t^2 + p_t^3$$

The quantity $q = 1 - (1-p_t)^3$ used in §6.5 and Appendix B.2–B.5 is exact under PRP, not an approximation. ∎

---

## Appendix C: GPU Validation

**Scope and limitations.** GPUs can freely allocate compute between ZK and PoW in software—the ZK:PoW ratio is a scheduling parameter. The ASIC-specific claim of simultaneous pipeline execution (§5.6) cannot be validated on GPU. What GPU validates: (1) STARK computation produces PoW tickets as a byproduct, and (2) no measurable throughput difference between input sources.

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

[1] M. Ball, A. Rosen, M. Sabin, and P. N. Vasudevan, "Proofs of Useful Work," IACR ePrint 2017/203, 2017. https://eprint.iacr.org/2017/203

[2] L. Grassi, D. Khovratovich, C. Rechberger, A. Roy, and M. Schofnegger, "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems," USENIX Security 2021. https://eprint.iacr.org/2019/458

[3] L. Grassi, D. Khovratovich, and M. Schofnegger, "Poseidon2: A Faster Version of the Poseidon Hash Function," IACR ePrint 2023/323, 2023. https://eprint.iacr.org/2023/323

[4] StarkWare, "Stwo: A STARK Prover." https://github.com/starkware-libs/stwo

[5] Y. Sompolinsky and M. Sutton, "The DAG KNIGHT Protocol: A Parameterless Generalization of Nakamoto Consensus," IACR ePrint 2022/1494, 2022. https://eprint.iacr.org/2022/1494

[6] Kaspa, "kHeavyHash Specification." https://github.com/kaspanet/rusty-kaspa

[7] M. Fitzi, A. Kiayias, G. Panagiotakos, and A. Russell, "Ofelimos: Combinatorial Optimization via Proof-of-Useful-Work," Crypto 2022. https://eprint.iacr.org/2021/1379

[8] I. Komargodski and O. Weinstein, "Proofs of Useful Work from Arbitrary Matrix Multiplication," IACR ePrint 2025/685, 2025. https://eprint.iacr.org/2025/685

[9] Y. Bar-On, I. Komargodski, and O. Weinstein, "Proof of Work With External Utilities," arXiv:2505.21685, 2025. https://arxiv.org/abs/2505.21685

[10] Plonky3, "A Toolkit for Polynomial IOPs." https://github.com/Plonky3/Plonky3 (accessed 2026-02-16).

[11] Nockchain, "The zkPoW L1." https://www.nockchain.org/ (accessed 2026-02-18).

[12] S.-P. Merz and À. Rodríguez García, "Skipping Class: Algebraic Attacks exploiting weak matrices and operation modes of Poseidon2(b)," IACR ePrint 2026/306, 2026. https://eprint.iacr.org/2026/306

[13] A. Bak, A. Bariant, A. Boeuf, M. Hostettler, and G. Jazeron, "Claiming bounties on small scale Poseidon and Poseidon2 instances using resultant-based algebraic attacks," IACR ePrint 2026/150, 2026. https://eprint.iacr.org/2026/150

[14] S. Oleksak, R. Gazdik, M. Peresini, and I. Homoliak, "Zk-SNARK Marketplace with Proof of Useful Work," arXiv:2510.09729, 2025. https://arxiv.org/abs/2510.09729

[15] T. Ashur, T. Buschman, and M. Mahzoun, "Algebraic Cryptanalysis of HADES Design Strategy: Application to POSEIDON and Poseidon2," IACR ePrint 2023/537, 2023. https://eprint.iacr.org/2023/537
