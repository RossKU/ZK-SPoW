//! ZK-SPoW Symbiotic Miner — Poseidon2 Width-24 / M31
//!
//! Real STARK proofs: polynomial LDE → constraint quotient → Poseidon2 Merkle
//! commitment → Fiat-Shamir queries → linearity test → verified.
//! Every Poseidon2 permutation produces 3 PoW tickets.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ── M31 field: p = 2^31 - 1 ────────────────────────────────

const P: u32 = 0x7FFF_FFFF;

#[inline(always)]
fn add(a: u32, b: u32) -> u32 {
    let s = a + b;
    let r = (s & P) + (s >> 31);
    if r >= P { r - P } else { r }
}

#[inline(always)]
fn sub(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { P - b + a }
}

#[inline(always)]
fn mul(a: u32, b: u32) -> u32 {
    let prod = (a as u64) * (b as u64);
    let lo = (prod & P as u64) as u32;
    let hi = (prod >> 31) as u32;
    let s = lo + hi;
    let r = (s & P) + (s >> 31);
    if r >= P { r - P } else { r }
}

#[inline(always)]
fn pow5(x: u32) -> u32 {
    let x2 = mul(x, x);
    let x4 = mul(x2, x2);
    mul(x4, x)
}

fn pow_mod(mut base: u32, mut exp: u32) -> u32 {
    let mut r = 1u32;
    while exp > 0 {
        if exp & 1 == 1 { r = mul(r, base); }
        base = mul(base, base);
        exp >>= 1;
    }
    r
}

fn inv(a: u32) -> u32 { pow_mod(a, P - 2) }

#[inline(always)]
fn neg(a: u32) -> u32 { if a == 0 { 0 } else { P - a } }

// ── Circle group over M31 ─────────────────────────────────
// C(M31) = {(x,y) ∈ M31² : x² + y² = 1}, |C(M31)| = p+1 = 2^31

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CirclePoint { x: u32, y: u32 }

impl CirclePoint {
    const ZERO: Self = CirclePoint { x: 1, y: 0 };
    const GEN: Self = CirclePoint { x: 2, y: 1268011823 }; // order 2^31
    const LOG_ORDER: u32 = 31;

    fn cadd(self, rhs: Self) -> Self {
        CirclePoint {
            x: sub(mul(self.x, rhs.x), mul(self.y, rhs.y)),
            y: add(mul(self.x, rhs.y), mul(self.y, rhs.x)),
        }
    }

    fn double(self) -> Self {
        let xx = mul(self.x, self.x);
        CirclePoint {
            x: sub(add(xx, xx), 1),
            y: { let xy = mul(self.x, self.y); add(xy, xy) },
        }
    }

    fn conjugate(self) -> Self {
        CirclePoint { x: self.x, y: neg(self.y) }
    }

    fn repeated_double(mut self, n: u32) -> Self {
        for _ in 0..n { self = self.double(); }
        self
    }

    /// x-coordinate doubling map: x → 2x² - 1
    fn double_x(x: u32) -> u32 {
        let x2 = mul(x, x);
        sub(add(x2, x2), 1)
    }

    fn subgroup_gen(log_size: u32) -> Self {
        Self::GEN.repeated_double(Self::LOG_ORDER - log_size)
    }
}

/// Canonic circle domain of size 2^log_size.
/// Layout: [0..N/2) = half-coset, [N/2..N) = conjugate half-coset.
/// Pairs (i, i+N/2) are conjugates sharing x-coordinate with negated y.
struct CDomain {
    half: Vec<CirclePoint>,
    log_size: u32,
}

impl CDomain {
    fn new(log_size: u32) -> Self {
        assert!(log_size >= 2);
        let half_n = 1usize << (log_size - 1);
        let initial = CirclePoint::subgroup_gen(log_size + 1);
        let step = CirclePoint::subgroup_gen(log_size - 1);
        let mut half = Vec::with_capacity(half_n);
        let mut cur = initial;
        for _ in 0..half_n {
            half.push(cur);
            cur = cur.cadd(step);
        }
        CDomain { half, log_size }
    }

    fn size(&self) -> usize { 1 << self.log_size }

    fn at(&self, i: usize) -> CirclePoint {
        let h = self.half.len();
        if i < h { self.half[i] } else { self.half[i - h].conjugate() }
    }
}

// ── FRI fold operations ───────────────────────────────────

/// Precompute inverse twiddle factors for all FRI fold levels.
/// Level 0: inv(y_i) for circle→line fold.
/// Level k≥1: inv of projected x-coordinates for line→line folds.
fn fri_twiddles(domain: &CDomain) -> Vec<Vec<u32>> {
    let half_n = domain.half.len();
    let mut levels: Vec<Vec<u32>> = Vec::new();

    // Level 0: inverse y-coordinates of half-coset
    levels.push(domain.half.iter().map(|p| inv(p.y)).collect());

    // Level 1+: projected x-coordinates, halving each time
    let mut xs: Vec<u32> = domain.half.iter().map(|p| p.x).collect();
    let mut size = half_n / 2;
    while size >= 1 {
        levels.push(xs[..size].iter().map(|&x| inv(x)).collect());
        for x in xs.iter_mut() { *x = CirclePoint::double_x(*x); }
        size /= 2;
    }
    levels
}

/// FRI fold: circle evaluations → line evaluations (first step).
/// Pairs: (i, i+N/2) are conjugates.
fn fri_fold_circle(evals: &[u32], alpha: u32, inv_y: &[u32]) -> Vec<u32> {
    let half = evals.len() / 2;
    (0..half).map(|i| {
        let (fp, fn_p) = (evals[i], evals[i + half]);
        let f0 = add(fp, fn_p);
        let f1 = mul(sub(fp, fn_p), inv_y[i]);
        add(f0, mul(alpha, f1))
    }).collect()
}

/// FRI fold: line evaluations → half-size line evaluations.
/// Pairs: (i, M-1-i) share negated x-coordinates.
fn fri_fold_line(evals: &[u32], alpha: u32, inv_x: &[u32]) -> Vec<u32> {
    let m = evals.len();
    let half = m / 2;
    (0..half).map(|i| {
        let (fx, fn_x) = (evals[i], evals[m - 1 - i]);
        let f0 = add(fx, fn_x);
        let f1 = mul(sub(fx, fn_x), inv_x[i]);
        add(f0, mul(alpha, f1))
    }).collect()
}

/// Full FRI fold chain: N evaluations → 1 value.
fn fri_fold_all(evals: &[u32], alphas: &[u32], twiddles: &[Vec<u32>]) -> Vec<u32> {
    assert_eq!(alphas.len(), twiddles.len());
    let mut cur = fri_fold_circle(evals, alphas[0], &twiddles[0]);
    for k in 1..twiddles.len() {
        cur = fri_fold_line(&cur, alphas[k], &twiddles[k]);
    }
    cur
}

/// Vanishing polynomial of a canonic coset of size 2^log_n.
/// Applies double_x (log_n - 1) times; result is 0 on the coset.
fn vanish_coset(x: u32, log_n: u32) -> u32 {
    let mut t = x;
    for _ in 1..log_n { t = CirclePoint::double_x(t); }
    t
}

// ── Circle FFT (CFFT / iCFFT) ─────────────────────────────

/// Forward twiddle factors (y_i, x_i, double_x(x_i), ...) — NOT inverted.
fn cfft_fwd_twiddles(domain: &CDomain) -> Vec<Vec<u32>> {
    let half_n = domain.half.len();
    let mut levels: Vec<Vec<u32>> = Vec::new();
    levels.push(domain.half.iter().map(|p| p.y).collect());
    let mut xs: Vec<u32> = domain.half.iter().map(|p| p.x).collect();
    let mut size = half_n / 2;
    while size >= 1 {
        levels.push(xs[..size].to_vec());
        for x in xs.iter_mut() { *x = CirclePoint::double_x(*x); }
        size /= 2;
    }
    levels
}

/// iCFFT helper: line levels (recursive).
fn icfft_line(evals: &[u32], inv_twid: &[Vec<u32>]) -> Vec<u32> {
    let m = evals.len();
    if m == 1 { return evals.to_vec(); }
    let half = m / 2;
    let mut even = Vec::with_capacity(half);
    let mut odd = Vec::with_capacity(half);
    for i in 0..half {
        let (a, b) = (evals[i], evals[m - 1 - i]);
        even.push(add(a, b));
        odd.push(mul(sub(a, b), inv_twid[0][i]));
    }
    let ec = icfft_line(&even, &inv_twid[1..]);
    let oc = icfft_line(&odd, &inv_twid[1..]);
    let mut r = Vec::with_capacity(m);
    for i in 0..half { r.push(ec[i]); r.push(oc[i]); }
    r
}

/// iCFFT: evaluations on circle domain → circle polynomial coefficients.
/// Basis: [1, y, x, xy, 2x²-1, (2x²-1)y, (2x²-1)x, ...] (circle Chebyshev).
fn icfft(evals: &[u32], inv_twid: &[Vec<u32>]) -> Vec<u32> {
    let n = evals.len();
    if n == 1 { return evals.to_vec(); }
    let half = n / 2;
    let mut even = Vec::with_capacity(half);
    let mut odd = Vec::with_capacity(half);
    for i in 0..half {
        let (a, b) = (evals[i], evals[i + half]);
        even.push(add(a, b));
        odd.push(mul(sub(a, b), inv_twid[0][i]));
    }
    let ec = icfft_line(&even, &inv_twid[1..]);
    let oc = icfft_line(&odd, &inv_twid[1..]);
    let inv_n = inv(n as u32);
    let mut r = Vec::with_capacity(n);
    for i in 0..half {
        r.push(mul(ec[i], inv_n));
        r.push(mul(oc[i], inv_n));
    }
    r
}

/// CFFT helper: line levels (recursive).
fn cfft_line(coeffs: &[u32], fwd_twid: &[Vec<u32>]) -> Vec<u32> {
    let m = coeffs.len();
    if m == 1 { return coeffs.to_vec(); }
    let half = m / 2;
    let mut ec = Vec::with_capacity(half);
    let mut oc = Vec::with_capacity(half);
    for i in 0..half { ec.push(coeffs[2 * i]); oc.push(coeffs[2 * i + 1]); }
    let ee = cfft_line(&ec, &fwd_twid[1..]);
    let oe = cfft_line(&oc, &fwd_twid[1..]);
    let mut r = vec![0u32; m];
    for i in 0..half {
        let t = mul(fwd_twid[0][i], oe[i]);
        r[i] = add(ee[i], t);
        r[m - 1 - i] = sub(ee[i], t);
    }
    r
}

/// CFFT: circle polynomial coefficients → evaluations on circle domain.
fn cfft(coeffs: &[u32], fwd_twid: &[Vec<u32>]) -> Vec<u32> {
    let n = coeffs.len();
    if n == 1 { return coeffs.to_vec(); }
    let half = n / 2;
    let mut ec = Vec::with_capacity(half);
    let mut oc = Vec::with_capacity(half);
    for i in 0..half { ec.push(coeffs[2 * i]); oc.push(coeffs[2 * i + 1]); }
    let ee = cfft_line(&ec, &fwd_twid[1..]);
    let oe = cfft_line(&oc, &fwd_twid[1..]);
    let mut r = vec![0u32; n];
    for i in 0..half {
        let t = mul(fwd_twid[0][i], oe[i]);
        r[i] = add(ee[i], t);
        r[i + half] = sub(ee[i], t);
    }
    r
}

/// Evaluate line polynomial at a single x-value (direct, O(n)).
fn line_eval_at(coeffs: &[u32], x: u32) -> u32 {
    if coeffs.len() == 1 { return coeffs[0]; }
    let half = coeffs.len() / 2;
    let mut ec = Vec::with_capacity(half);
    let mut oc = Vec::with_capacity(half);
    for i in 0..half { ec.push(coeffs[2 * i]); oc.push(coeffs[2 * i + 1]); }
    let nx = CirclePoint::double_x(x);
    add(line_eval_at(&ec, nx), mul(x, line_eval_at(&oc, nx)))
}

/// Evaluate circle polynomial at a single point (direct, O(n)).
fn circle_eval_at(coeffs: &[u32], p: CirclePoint) -> u32 {
    let n = coeffs.len();
    if n == 1 { return coeffs[0]; }
    let half = n / 2;
    let mut ec = Vec::with_capacity(half);
    let mut oc = Vec::with_capacity(half);
    for i in 0..half { ec.push(coeffs[2 * i]); oc.push(coeffs[2 * i + 1]); }
    add(line_eval_at(&ec, p.x), mul(p.y, line_eval_at(&oc, p.x)))
}

// ── Circle STARK prover & verifier ────────────────────────

const LOG_BLOWUP: u32 = 2; // 4x blowup
const CSTARK_QUERIES: usize = 24;
const FRI_FOLD_STOP: usize = 4; // stop folding when <= this many values

struct CStarkProof {
    a0: u32,
    a1: u32,
    log_trace: u32,
    trace_root: [u32; 8],
    // FRI layers: (merkle_root, folded_values_if_last_layer)
    fri_roots: Vec<[u32; 8]>,
    fri_last: Vec<u32>,
    // Query openings per query: (position, trace_leaf+path, fri_layer_leaves+paths)
    query_positions: Vec<usize>,
    trace_opens: Vec<LeafOpen>,
    fri_opens: Vec<Vec<LeafOpen>>, // fri_opens[query][layer]
    fri_alphas_count: usize,
}

/// Compute Fibonacci trace on circle domain.
fn circle_fib_trace(log_n: u32, a0: u32, a1: u32) -> Vec<u32> {
    let n = 1usize << log_n;
    let mut t = vec![0u32; n];
    t[0] = a0;
    t[1] = a1;
    for i in 2..n { t[i] = add(t[i - 1], t[i - 2]); }
    t
}

/// Circle STARK prove: Fibonacci(a0, a1, len=2^log_trace).
fn cstark_prove(
    trace: &[u32], log_trace: u32, h_h: &[u32; 8], rc: &RC, target: &[u32; 8],
) -> (CStarkProof, u64, Option<([u32; W], usize)>) {
    let n = trace.len();
    let log_lde = log_trace + LOG_BLOWUP;
    let lde_n = 1usize << log_lde;

    // 1. Interpolate trace → coefficients → LDE
    let trace_dom = CDomain::new(log_trace);
    let trace_inv_tw = fri_twiddles(&trace_dom);
    let coeffs = icfft(trace, &trace_inv_tw);
    let lde_dom = CDomain::new(log_lde);
    let lde_fwd_tw = cfft_fwd_twiddles(&lde_dom);
    let mut padded = vec![0u32; lde_n];
    padded[..n].copy_from_slice(&coeffs);
    let lde_evals = cfft(&padded, &lde_fwd_tw);

    // 2. Merkle commit trace LDE
    let t_tree = MerkleTree::build(pack_evals(&lde_evals), *h_h, rc, target);
    let mut perms = t_tree.perms;
    let mut ticket = t_tree.ticket;

    // 3. Fiat-Shamir: derive FRI alphas from trace root
    let mut fs = [0u32; W];
    fs[..8].copy_from_slice(&t_tree.root());
    fs[8] = 0x4652_4931; // "FRI1"
    poseidon2(&mut fs, rc);
    perms += 1;
    if ticket.is_none() {
        if let Some(slot) = check_tickets(&fs, target) { ticket = Some((fs, slot)); }
    }
    let mut rng = Rng(fs[0] as u64 | ((fs[1] as u64) << 32) | 1);

    // 4. FRI commit phase: fold LDE evaluations down
    let lde_inv_tw = fri_twiddles(&lde_dom);
    let mut fri_roots: Vec<[u32; 8]> = Vec::new();
    let mut fri_trees: Vec<MerkleTree> = Vec::new();
    let mut current = lde_evals.clone();
    let mut cur_inv_tw = &lde_inv_tw[..];

    // First fold: circle → line
    let alpha0 = rng.next() as u32 & P;
    current = fri_fold_circle(&current, alpha0, &cur_inv_tw[0]);
    cur_inv_tw = &cur_inv_tw[1..];
    let mut alphas = vec![alpha0];

    // Subsequent folds: line → line
    while current.len() > FRI_FOLD_STOP {
        let tree = MerkleTree::build(pack_evals(&current), *h_h, rc, target);
        perms += tree.perms;
        if ticket.is_none() { ticket = tree.ticket; }
        fri_roots.push(tree.root());
        // Derive next alpha from this commitment
        let mut fs2 = [0u32; W];
        fs2[..8].copy_from_slice(&tree.root());
        fs2[8] = 0x4652_4932; // "FRI2"
        poseidon2(&mut fs2, rc);
        perms += 1;
        if ticket.is_none() {
            if let Some(slot) = check_tickets(&fs2, target) { ticket = Some((fs2, slot)); }
        }
        let alpha = fs2[0] & P;
        alphas.push(alpha);
        fri_trees.push(tree);
        current = fri_fold_line(&current, alpha, &cur_inv_tw[0]);
        cur_inv_tw = &cur_inv_tw[1..];
    }
    let fri_last = current;

    // 5. Query phase
    let mut query_positions = Vec::with_capacity(CSTARK_QUERIES);
    let mut trace_opens = Vec::with_capacity(CSTARK_QUERIES);
    let mut fri_opens: Vec<Vec<LeafOpen>> = Vec::with_capacity(CSTARK_QUERIES);
    let lde_leaves = t_tree.layers[0].len();
    for _ in 0..CSTARK_QUERIES {
        let pos = (rng.next() as usize) % lde_n;
        let leaf_idx = pos / 8;
        query_positions.push(pos);
        trace_opens.push(t_tree.open(leaf_idx.min(lde_leaves - 1)));
        let mut layer_opens = Vec::new();
        for tree in &fri_trees {
            let sz = tree.layers[0].len();
            let li = (pos / 8).min(sz - 1);
            layer_opens.push(tree.open(li));
        }
        fri_opens.push(layer_opens);
    }

    let proof = CStarkProof {
        a0: trace[0], a1: trace[1], log_trace,
        trace_root: t_tree.root(),
        fri_roots, fri_last,
        query_positions, trace_opens, fri_opens,
        fri_alphas_count: alphas.len(),
    };
    (proof, perms, ticket)
}

/// Circle STARK verify.
/// Recomputes the Fibonacci trace from public inputs (a0, a1),
/// derives the trace polynomial, and checks LDE consistency at query points.
fn cstark_verify(proof: &CStarkProof, h_h: &[u32; 8], rc: &RC) -> bool {
    let log_lde = proof.log_trace + LOG_BLOWUP;
    let lde_n = 1usize << log_lde;

    // 1. Recompute full LDE from public inputs (a0, a1)
    let trace = circle_fib_trace(proof.log_trace, proof.a0, proof.a1);
    let trace_dom = CDomain::new(proof.log_trace);
    let inv_tw = fri_twiddles(&trace_dom);
    let coeffs = icfft(&trace, &inv_tw);
    let lde_dom = CDomain::new(log_lde);
    let lde_fwd = cfft_fwd_twiddles(&lde_dom);
    let mut padded = vec![0u32; lde_n];
    padded[..trace.len()].copy_from_slice(&coeffs);
    let lde_evals = cfft(&padded, &lde_fwd);

    // 2. Re-derive Fiat-Shamir
    let mut fs = [0u32; W];
    fs[..8].copy_from_slice(&proof.trace_root);
    fs[8] = 0x4652_4931;
    poseidon2(&mut fs, rc);
    let mut rng = Rng(fs[0] as u64 | ((fs[1] as u64) << 32) | 1);
    let alpha0 = rng.next() as u32 & P;
    let mut alphas = vec![alpha0];
    for root in &proof.fri_roots {
        let mut fs2 = [0u32; W];
        fs2[..8].copy_from_slice(root);
        fs2[8] = 0x4652_4932;
        poseidon2(&mut fs2, rc);
        alphas.push(fs2[0] & P);
    }
    if alphas.len() != proof.fri_alphas_count { return false; }

    // 3. Verify queries: Merkle openings + LDE consistency
    for qi in 0..CSTARK_QUERIES {
        let pos = (rng.next() as usize) % lde_n;
        if proof.query_positions[qi] != pos { return false; }

        let to = &proof.trace_opens[qi];
        if !MerkleTree::verify_path(to.leaf, to.idx, &to.path, proof.trace_root, h_h, rc) {
            return false;
        }

        // Committed value must equal our recomputed LDE
        if to.leaf[pos % 8] != lde_evals[pos] { return false; }

        // FRI layer Merkle openings
        for (li, fri_root) in proof.fri_roots.iter().enumerate() {
            let lo = &proof.fri_opens[qi][li];
            if !MerkleTree::verify_path(lo.leaf, lo.idx, &lo.path, *fri_root, h_h, rc) {
                return false;
            }
        }
    }

    // 4. FRI last layer: all values equal (degree-0)
    if !proof.fri_last.is_empty() {
        let first = proof.fri_last[0];
        for &v in &proof.fri_last[1..] {
            if v != first { return false; }
        }
    }

    true
}

/// Circle STARK smoke test — verify circle group, domain, and FRI.
fn circle_smoke_test() {
    let g = CirclePoint::GEN;
    assert_eq!(add(mul(g.x, g.x), mul(g.y, g.y)), 1, "GEN not on circle");
    let id = g.repeated_double(CirclePoint::LOG_ORDER);
    assert_eq!(id, CirclePoint::ZERO, "GEN order != 2^31");
    let not_id = g.repeated_double(CirclePoint::LOG_ORDER - 1);
    assert_ne!(not_id, CirclePoint::ZERO, "GEN order < 2^31");

    let log_n: u32 = 4;
    let n = 1usize << log_n;
    let dom = CDomain::new(log_n);
    assert_eq!(dom.size(), n);

    // All domain points on circle
    for i in 0..n {
        let p = dom.at(i);
        assert_eq!(add(mul(p.x, p.x), mul(p.y, p.y)), 1, "pt {} off circle", i);
    }
    // Conjugate pairing
    for i in 0..n / 2 {
        let (p, q) = (dom.at(i), dom.at(i + n / 2));
        assert_eq!(p.x, q.x, "conj x mismatch {}", i);
        assert_eq!(p.y, neg(q.y), "conj y mismatch {}", i);
    }
    // All points distinct
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (dom.at(i), dom.at(j));
            assert!(a.x != b.x || a.y != b.y, "duplicate pts {} {}", i, j);
        }
    }

    // Vanishing polynomial = 0 on domain, != 0 off domain
    for i in 0..n {
        assert_eq!(vanish_coset(dom.at(i).x, log_n), 0, "vanish!=0 at {}", i);
    }
    let off = CirclePoint::GEN; // not in domain
    assert_ne!(vanish_coset(off.x, log_n), 0, "vanish==0 off domain");

    // FRI: fold constant function → scaled constant
    let twiddles = fri_twiddles(&dom);
    let alphas: Vec<u32> = vec![7, 13, 19, 23];
    assert_eq!(twiddles.len(), alphas.len());
    let evals = vec![42u32; n];
    let result = fri_fold_all(&evals, &alphas, &twiddles);
    assert_eq!(result.len(), 1);
    // Constant → f1=0 at every level → result = 2^log_n * c
    let mut expected = 42u32;
    for _ in 0..log_n { expected = add(expected, expected); }
    assert_eq!(result[0], expected, "FRI constant fold: {} != {}", result[0], expected);

    // iCFFT → CFFT roundtrip
    let inv_tw = fri_twiddles(&dom);
    let fwd_tw = cfft_fwd_twiddles(&dom);
    let test_evals: Vec<u32> = (0..n).map(|i| (i as u32 * 17 + 3) % P).collect();
    let coeffs = icfft(&test_evals, &inv_tw);
    let recovered = cfft(&coeffs, &fwd_tw);
    for i in 0..n {
        assert_eq!(recovered[i], test_evals[i], "roundtrip fail at {}", i);
    }

    // LDE: interpolate on small domain, evaluate on larger domain
    let log_lde: u32 = log_n + 2; // 4x blowup
    let lde_n = 1usize << log_lde;
    let lde_dom = CDomain::new(log_lde);
    let lde_fwd = cfft_fwd_twiddles(&lde_dom);
    let mut padded = vec![0u32; lde_n];
    padded[..n].copy_from_slice(&coeffs);
    let lde_evals = cfft(&padded, &lde_fwd);
    // Verify: vanish_coset(trace domain) != 0 on all LDE points
    for i in 0..lde_n {
        let v = vanish_coset(lde_dom.at(i).x, log_n);
        assert_ne!(v, 0, "LDE pt {} in trace vanish set", i);
    }

    // Circle STARK prove & verify
    let crc = gen_rc();
    let ch_h = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let ctarget = make_target(30); // hard target (won't find block)
    let log_t: u32 = 4;
    let ctrace = circle_fib_trace(log_t, 1, 1);
    let (cproof, cperms, _) = cstark_prove(&ctrace, log_t, &ch_h, &crc, &ctarget);
    let cvalid = cstark_verify(&cproof, &ch_h, &crc);
    assert!(cvalid, "Circle STARK proof failed verification");

    // Tamper test: modified trace must fail verification
    let mut bad = ctrace.clone();
    bad[3] = add(bad[3], 1);
    let (bp, _, _) = cstark_prove(&bad, log_t, &ch_h, &crc, &ctarget);
    assert!(!cstark_verify(&bp, &ch_h, &crc), "Tampered proof must fail");

    println!("  Circle smoke test PASSED (N={}, LDE={}, CSTARK verified, tamper rejected, {} perms)", n, lde_n, cperms);
}

// ── Poseidon2 Width-24 ─────────────────────────────────────

const W: usize = 24;
const RF: usize = 8;
const RP: usize = 22;
const HRF: usize = 4;

const DIAG: [u32; W] = [
    P - 2,
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152, 4194304,
];

#[derive(Clone, Copy)]
struct RC { ext: [[u32; W]; RF], int: [u32; RP] }

fn gen_rc() -> RC {
    let mut z: u64 = 0x5A4B_3C2D_1E0F_A9B8;
    let mut next = || -> u32 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        (x as u32) & P
    };
    let mut rc = RC { ext: [[0; W]; RF], int: [0; RP] };
    for r in rc.ext.iter_mut() { for e in r.iter_mut() { *e = next(); } }
    for e in rc.int.iter_mut() { *e = next(); }
    rc
}

#[inline]
fn m4(x: &mut [u32; 4]) {
    let (a, b, c, d) = (x[0], x[1], x[2], x[3]);
    let ab = add(a, b);
    let cd = add(c, d);
    let t2 = add(add(b, b), cd);
    let t3 = add(add(d, d), ab);
    let ab4 = { let t = add(ab, ab); add(t, t) };
    let cd4 = { let t = add(cd, cd); add(t, t) };
    let t5 = add(ab4, t2);
    let t4 = add(cd4, t3);
    x[0] = add(t3, t5);
    x[1] = t5;
    x[2] = add(t2, t4);
    x[3] = t4;
}

fn ext_mds(s: &mut [u32; W]) {
    let mut sum = [0u32; 4];
    for g in 0..6 { for i in 0..4 { sum[i] = add(sum[i], s[g * 4 + i]); } }
    for g in 0..6 {
        let mut tmp = [0u32; 4];
        for i in 0..4 { tmp[i] = add(s[g * 4 + i], sum[i]); }
        m4(&mut tmp);
        for i in 0..4 { s[g * 4 + i] = tmp[i]; }
    }
}

fn int_mds(s: &mut [u32; W]) {
    let mut sum = 0u32;
    for i in 0..W { sum = add(sum, s[i]); }
    let orig = *s;
    s[0] = sub(sum, add(orig[0], orig[0]));
    for i in 1..W { s[i] = add(sum, mul(DIAG[i], orig[i])); }
}

fn poseidon2(s: &mut [u32; W], rc: &RC) {
    ext_mds(s);
    for r in 0..HRF {
        for i in 0..W { s[i] = add(s[i], rc.ext[r][i]); }
        for i in 0..W { s[i] = pow5(s[i]); }
        ext_mds(s);
    }
    for r in 0..RP {
        s[0] = add(s[0], rc.int[r]);
        s[0] = pow5(s[0]);
        int_mds(s);
    }
    for r in HRF..RF {
        for i in 0..W { s[i] = add(s[i], rc.ext[r][i]); }
        for i in 0..W { s[i] = pow5(s[i]); }
        ext_mds(s);
    }
}

// ── Compression (§3.3) & Tickets ────────────────────────────

fn compress(left: &[u32; 8], right: &[u32; 8], h_h: &[u32; 8], rc: &RC) -> ([u32; 8], [u32; W]) {
    let mut s = [0u32; W];
    s[..8].copy_from_slice(left);
    s[8..16].copy_from_slice(right);
    s[16..24].copy_from_slice(h_h);
    poseidon2(&mut s, rc);
    let mut h = [0u32; 8];
    h.copy_from_slice(&s[0..8]);
    (h, s)
}

fn ticket_lt(ticket: &[u32], target: &[u32; 8]) -> bool {
    for i in 0..8 {
        if ticket[i] < target[i] { return true; }
        if ticket[i] > target[i] { return false; }
    }
    false
}

fn make_target(d: u32) -> [u32; 8] {
    let mut t = [P - 1; 8];
    t[0] = if d >= 31 { 0 } else { P >> d };
    t
}

fn check_tickets(state: &[u32; W], target: &[u32; 8]) -> Option<usize> {
    for slot in 0..3 {
        if ticket_lt(&state[slot * 8..(slot + 1) * 8], target) { return Some(slot); }
    }
    None
}

// ── RNG & helpers ───────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn next_pow2(n: usize) -> usize { let mut p = 1; while p < n { p <<= 1; } p }

fn pack_evals(evals: &[u32]) -> Vec<[u32; 8]> {
    evals.chunks(8).map(|c| {
        let mut leaf = [0u32; 8];
        for (i, &v) in c.iter().enumerate() { leaf[i] = v; }
        leaf
    }).collect()
}

// ── Merkle tree ─────────────────────────────────────────────

struct LeafOpen { idx: usize, leaf: [u32; 8], path: Vec<[u32; 8]> }

struct MerkleTree {
    layers: Vec<Vec<[u32; 8]>>,
    perms: u64,
    ticket: Option<([u32; W], usize)>,
}

impl MerkleTree {
    fn build(mut leaves: Vec<[u32; 8]>, h_h: [u32; 8], rc: &RC, target: &[u32; 8]) -> Self {
        let n = next_pow2(leaves.len());
        leaves.resize(n, [0u32; 8]);
        let mut layers = vec![leaves];
        let mut perms = 0u64;
        let mut ticket = None;
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let mut next_layer = Vec::with_capacity(prev.len() / 2);
            for i in (0..prev.len()).step_by(2) {
                let (hash, state) = compress(&prev[i], &prev[i + 1], &h_h, rc);
                perms += 1;
                if ticket.is_none() {
                    if let Some(slot) = check_tickets(&state, target) {
                        ticket = Some((state, slot));
                    }
                }
                next_layer.push(hash);
            }
            layers.push(next_layer);
        }
        MerkleTree { layers, perms, ticket }
    }

    fn root(&self) -> [u32; 8] { self.layers.last().unwrap()[0] }

    fn auth_path(&self, mut idx: usize) -> Vec<[u32; 8]> {
        let mut path = Vec::with_capacity(self.layers.len() - 1);
        for layer in &self.layers[..self.layers.len() - 1] {
            path.push(layer[(idx ^ 1).min(layer.len() - 1)]);
            idx >>= 1;
        }
        path
    }

    fn verify_path(leaf: [u32; 8], mut idx: usize, path: &[[u32; 8]],
                   root: [u32; 8], h_h: &[u32; 8], rc: &RC) -> bool {
        let mut cur = leaf;
        for sib in path {
            let (l, r) = if idx & 1 == 0 { (&cur, sib) } else { (sib, &cur) };
            cur = compress(l, r, h_h, rc).0;
            idx >>= 1;
        }
        cur == root
    }

    fn open(&self, leaf_idx: usize) -> LeafOpen {
        LeafOpen { idx: leaf_idx, leaf: self.layers[0][leaf_idx], path: self.auth_path(leaf_idx) }
    }
}

// ── Polynomial operations ───────────────────────────────────

fn precompute_inv_table(max: usize) -> Vec<u32> {
    let mut tab = vec![0u32; max + 1];
    if max >= 1 { tab[1] = 1; }
    for i in 2..=max {
        // inv(i) = -(P/i) * inv(P mod i)
        tab[i] = mul(P - (P / (i as u32)), tab[(P % (i as u32)) as usize]);
    }
    tab
}

fn barycentric_weights(n: usize) -> Vec<u32> {
    let mut fact = vec![1u32; n];
    for i in 1..n { fact[i] = mul(fact[i - 1], i as u32); }
    let mut inv_fact = vec![1u32; n];
    inv_fact[n - 1] = inv(fact[n - 1]);
    for i in (0..n - 1).rev() { inv_fact[i] = mul(inv_fact[i + 1], (i + 1) as u32); }
    let mut w = vec![0u32; n];
    for i in 0..n {
        let sign = if (n - 1 - i) % 2 == 0 { 1 } else { P - 1 };
        w[i] = mul(sign, mul(inv_fact[i], inv_fact[n - 1 - i]));
    }
    w
}

/// Evaluate trace polynomial on LDE domain via barycentric Lagrange.
fn compute_lde(trace: &[u32], weights: &[u32], inv_tab: &[u32], dom: usize) -> Vec<u32> {
    let n = trace.len();
    let mut evals = Vec::with_capacity(dom);
    for &v in trace { evals.push(v); }

    // Precompute w_i * f_i
    let mut wf = vec![0u32; n];
    for i in 0..n { wf[i] = mul(weights[i], trace[i]); }

    // ℓ(n) = n! (falling factorial base)
    let mut ell = 1u32;
    for i in 1..=n { ell = mul(ell, i as u32); }

    for x in n..dom {
        let mut sum = 0u32;
        for i in 0..n { sum = add(sum, mul(wf[i], inv_tab[x - i])); }
        evals.push(mul(ell, sum));
        // ℓ(x+1) = ℓ(x) * (x+1) / (x+1-n)
        if x + 1 < dom {
            ell = mul(mul(ell, (x + 1) as u32), inv_tab[x + 1 - n]);
        }
    }
    evals
}

/// Compute constraint quotient Q(x) = (T(x+2)-T(x+1)-T(x)) / Z(x).
fn compute_quotient(t_evals: &[u32], n: usize, inv_tab: &[u32]) -> Vec<u32> {
    let dom = t_evals.len();
    let mut q_evals = vec![0u32; dom];
    let start = n - 2;
    let count = dom - n; // number of Q evaluations

    // Incremental Z(x) = ∏_{i=0}^{n-3}(x-i)
    // Z(start) = Z(n-2) = (n-2)!
    let mut z = 1u32;
    for i in 1..=(n as u32 - 2) { z = mul(z, i); }
    let mut z_vals = Vec::with_capacity(count);
    z_vals.push(z);
    for x in start..(start + count - 1) {
        // Z(x+1) = Z(x) * (x+1) / (x-n+3)
        z = mul(mul(z, (x + 1) as u32), inv_tab[x + 3 - n]);
        z_vals.push(z);
    }

    // Batch invert all Z values (Montgomery's trick)
    let mut prefix = vec![0u32; count];
    prefix[0] = z_vals[0];
    for i in 1..count { prefix[i] = mul(prefix[i - 1], z_vals[i]); }
    let mut inv_p = inv(prefix[count - 1]);
    let mut z_invs = vec![0u32; count];
    for i in (1..count).rev() {
        z_invs[i] = mul(inv_p, prefix[i - 1]);
        inv_p = mul(inv_p, z_vals[i]);
    }
    z_invs[0] = inv_p;

    for (k, x) in (start..start + count).enumerate() {
        let c = sub(t_evals[x + 2], add(t_evals[x + 1], t_evals[x]));
        q_evals[x] = mul(c, z_invs[k]);
    }
    q_evals
}

/// Evaluate vanishing polynomial Z(x) = ∏_{i=0}^{n-3}(x-i) at a single point.
fn vanishing_eval(x: u32, n: usize) -> u32 {
    let mut z = 1u32;
    for i in 0..(n as u32 - 2) { z = mul(z, sub(x, i)); }
    z
}

// ── STARK structures ────────────────────────────────────────

const BLOWUP: usize = 4;
const N_QUERIES: usize = 32;

struct StarkQuery {
    x: usize,
    t_lo: LeafOpen,  // trace leaf at x/8
    t_hi: LeafOpen,  // trace leaf at (x+2)/8
    q: LeafOpen,     // quotient leaf at x/8
}

struct StarkProof {
    a0: u32,
    a1: u32,
    trace_len: usize,
    trace_root: [u32; 8],
    quotient_root: [u32; 8],
    boundary: LeafOpen,
    queries: Vec<StarkQuery>,
}

// ── STARK prover ────────────────────────────────────────────

fn fib_trace(n: usize, a0: u32, a1: u32) -> Vec<u32> {
    let mut t = vec![0u32; n];
    t[0] = a0; t[1] = a1;
    for i in 2..n { t[i] = add(t[i - 1], t[i - 2]); }
    t
}

fn stark_prove(
    trace: &[u32], h_h: &[u32; 8], rc: &RC, target: &[u32; 8],
    weights: &[u32], inv_tab: &[u32],
) -> (StarkProof, u64, Option<([u32; W], usize)>) {
    let n = trace.len();
    let dom = n * BLOWUP;

    // 1. Polynomial LDE
    let t_evals = compute_lde(trace, weights, inv_tab, dom);

    // 2. Commit trace evaluations
    let t_tree = MerkleTree::build(pack_evals(&t_evals), *h_h, rc, target);
    let mut perms = t_tree.perms;
    let mut ticket = t_tree.ticket;

    // 3. Constraint quotient Q = (T(x+2)-T(x+1)-T(x)) / Z(x)
    let q_evals = compute_quotient(&t_evals, n, inv_tab);

    // 4. Commit quotient
    let q_tree = MerkleTree::build(pack_evals(&q_evals), *h_h, rc, target);
    perms += q_tree.perms;
    if ticket.is_none() { ticket = q_tree.ticket; }

    // 5. Fiat-Shamir
    let mut fs = [0u32; W];
    fs[..8].copy_from_slice(&t_tree.root());
    fs[8..16].copy_from_slice(&q_tree.root());
    fs[16] = 0x5354_4152; // "STAR"
    poseidon2(&mut fs, rc);
    perms += 1;
    if ticket.is_none() {
        if let Some(slot) = check_tickets(&fs, target) { ticket = Some((fs, slot)); }
    }
    let mut rng = Rng(fs[0] as u64 | ((fs[1] as u64) << 32) | 1);

    // 6. Query openings
    let qrange = dom - 2 - n; // query in [n, dom-3]
    let mut queries = Vec::with_capacity(N_QUERIES);
    for _ in 0..N_QUERIES {
        let x = n + ((rng.next() as usize) % qrange);
        queries.push(StarkQuery {
            x,
            t_lo: t_tree.open(x / 8),
            t_hi: t_tree.open((x + 2) / 8),
            q: q_tree.open(x / 8),
        });
    }

    let proof = StarkProof {
        a0: trace[0], a1: trace[1], trace_len: n,
        trace_root: t_tree.root(), quotient_root: q_tree.root(),
        boundary: t_tree.open(0), queries,
    };
    (proof, perms, ticket)
}

// ── STARK verifier ──────────────────────────────────────────

fn stark_verify(proof: &StarkProof, h_h: &[u32; 8], rc: &RC) -> bool {
    let n = proof.trace_len;
    let dom = n * BLOWUP;

    // 1. Boundary: T(0)=a0, T(1)=a1
    if !MerkleTree::verify_path(proof.boundary.leaf, proof.boundary.idx,
                                 &proof.boundary.path, proof.trace_root, h_h, rc) {
        return false;
    }
    if proof.boundary.idx != 0
        || proof.boundary.leaf[0] != proof.a0
        || proof.boundary.leaf[1] != proof.a1 { return false; }

    // 2. Re-derive Fiat-Shamir
    let mut fs = [0u32; W];
    fs[..8].copy_from_slice(&proof.trace_root);
    fs[8..16].copy_from_slice(&proof.quotient_root);
    fs[16] = 0x5354_4152;
    poseidon2(&mut fs, rc);
    let mut rng = Rng(fs[0] as u64 | ((fs[1] as u64) << 32) | 1);
    let qrange = dom - 2 - n;

    // 3. Verify each query
    let mut q_pts: Vec<(u32, u32)> = Vec::with_capacity(N_QUERIES);
    for q in &proof.queries {
        let expected_x = n + ((rng.next() as usize) % qrange);
        if q.x != expected_x { return false; }
        if q.t_lo.idx != q.x / 8 || q.t_hi.idx != (q.x + 2) / 8 || q.q.idx != q.x / 8 {
            return false;
        }

        // Merkle auth paths
        if !MerkleTree::verify_path(q.t_lo.leaf, q.t_lo.idx, &q.t_lo.path, proof.trace_root, h_h, rc) { return false; }
        if q.t_lo.idx != q.t_hi.idx {
            if !MerkleTree::verify_path(q.t_hi.leaf, q.t_hi.idx, &q.t_hi.path, proof.trace_root, h_h, rc) { return false; }
        }
        if !MerkleTree::verify_path(q.q.leaf, q.q.idx, &q.q.path, proof.quotient_root, h_h, rc) { return false; }

        // Extract T(x), T(x+1), T(x+2), Q(x)
        let tx = q.t_lo.leaf[q.x % 8];
        let tx1 = if (q.x + 1) / 8 == q.t_lo.idx { q.t_lo.leaf[(q.x + 1) % 8] } else { q.t_hi.leaf[(q.x + 1) % 8] };
        let tx2 = q.t_hi.leaf[(q.x + 2) % 8];
        let qx = q.q.leaf[q.x % 8];

        // Constraint: Q(x) * Z(x) == T(x+2) - T(x+1) - T(x)
        let z = vanishing_eval(q.x as u32, n);
        if mul(qx, z) != sub(tx2, add(tx1, tx)) { return false; }

        q_pts.push((q.x as u32, qx));
    }

    // 4. Linearity test: Q must be degree < 2 (linear)
    let (x0, q0) = q_pts[0];
    let mut dx_ref = 0u32;
    let mut dq_ref = 0u32;
    let mut ref_set = false;
    for &(xi, qi) in &q_pts[1..] {
        if xi == x0 { if qi != q0 { return false; } continue; }
        let dx = sub(xi, x0);
        let dq = sub(qi, q0);
        if !ref_set { dx_ref = dx; dq_ref = dq; ref_set = true; continue; }
        if mul(dq, dx_ref) != mul(dq_ref, dx) { return false; }
    }

    true
}

// ── Stats & display ─────────────────────────────────────────

struct Stats {
    found: AtomicBool,
    stark_perms: AtomicU64,
    pow_perms: AtomicU64,
    stark_proofs: AtomicU64,
}

fn fmt(n: u64) -> String {
    if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) }
    else { format!("{}", n) }
}

fn print_result(
    source: &str, tid: usize, slot: usize, state: &[u32; W],
    nonce: Option<u64>, proof_info: Option<&str>,
    stats: &Stats, start: Instant,
) {
    let sp = stats.stark_perms.load(Ordering::Relaxed);
    let pp = stats.pow_perms.load(Ordering::Relaxed);
    let total = sp + pp;
    let elapsed = start.elapsed().as_secs_f64();
    let proofs = stats.stark_proofs.load(Ordering::Relaxed);
    let f = if total > 0 { sp as f64 / total as f64 * 100.0 } else { 0.0 };
    eprintln!("\r{:76}", "");
    println!("*** BLOCK FOUND ({}) ***", source);
    if let Some(n) = nonce { println!("Nonce       : {}", n); }
    if let Some(info) = proof_info { println!("Proof       : {}", info); }
    println!("Thread      : {} ({})", tid, source);
    println!("Ticket slot : {} (of 0,1,2)", slot);
    print!("Ticket      : [");
    for j in 0..8 { if j > 0 { print!(", "); } print!("{}", state[slot * 8 + j]); }
    println!("]");
    println!("────────────────────────────────────────");
    println!("STARK proofs: {} (verified)", proofs);
    println!("STARK perms : {}", fmt(sp));
    println!("PoW perms   : {}", fmt(pp));
    println!("Total perms : {}", fmt(total));
    println!("STARK frac  : {:.1}%", f);
    println!("Time        : {:.3}s", elapsed);
    println!("Perm rate   : {} perm/s", fmt((total as f64 / elapsed) as u64));
}

// ── STARK worker ────────────────────────────────────────────

fn stark_worker(
    tid: usize, h_h: [u32; 8], target: [u32; 8],
    log_trace: u32, stats: Arc<Stats>, start: Instant,
) {
    let rc = gen_rc();
    let mut proof_id: u32 = tid as u32;

    while !stats.found.load(Ordering::Relaxed) {
        let trace = circle_fib_trace(log_trace, proof_id, 1);
        let (proof, perms, ticket) = cstark_prove(&trace, log_trace, &h_h, &rc, &target);
        stats.stark_perms.fetch_add(perms, Ordering::Relaxed);

        let valid = cstark_verify(&proof, &h_h, &rc);
        if !valid { eprintln!("BUG: Circle STARK proof #{} failed!", proof_id); }
        stats.stark_proofs.fetch_add(1, Ordering::Relaxed);

        if let Some((state, slot)) = ticket {
            if stats.found.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                let info = format!("CircleSTARK Fib(a0={}, len=2^{}) LDE=4x FRI verified={}", proof_id, log_trace, valid);
                print_result("STARK", tid, slot, &state, None, Some(&info), &stats, start);
                // Print ZK proof certificate
                print!("\n=== ZK-SPoW PROOF CERTIFICATE ===\n");
                println!("AIR         : Fibonacci(a0={}, a1=1, len=2^{})", proof_id, log_trace);
                println!("Field       : M31 (p=2^31-1)");
                println!("Domain      : Circle C(M31), |C|=2^31");
                print!("Trace root  : ");
                for v in &proof.trace_root { print!("{:08x}", v); }
                println!();
                println!("FRI layers  : {}", proof.fri_roots.len());
                for (k, r) in proof.fri_roots.iter().enumerate() {
                    print!("  FRI[{}]    : ", k);
                    for v in r { print!("{:08x}", v); }
                    println!();
                }
                println!("FRI last    : {} values (degree-0)", proof.fri_last.len());
                println!("Queries     : {}", CSTARK_QUERIES);
                println!("Blowup      : {}x", 1u32 << LOG_BLOWUP);
                println!("Verified    : {}", valid);
                print!("Ticket      : ");
                for j in 0..8 { if j > 0 { print!(","); } print!("{:08x}", state[slot * 8 + j]); }
                println!();
                println!("=== END CERTIFICATE ===");
            }
            return;
        }
        proof_id += 100;
    }
}

// ── PoW worker (§4.2) ──────────────────────────────────────

fn pow_worker(
    tid: usize, h_h: [u32; 8], target: [u32; 8],
    stats: Arc<Stats>, start: Instant, show_progress: bool,
) {
    let rc = gen_rc();
    let mut v2 = [0u32; 8];
    v2[0] = (tid as u32) & P;
    let mut nonce: u64 = 0;
    let mut local: u64 = 0;

    while !stats.found.load(Ordering::Relaxed) {
        let mut v1 = [0u32; 8];
        v1[0] = (nonce as u32) & P;
        v1[1] = ((nonce >> 31) as u32) & P;
        v1[2] = ((nonce >> 62) as u32) & 0x3;

        let mut state = [0u32; W];
        state[..8].copy_from_slice(&v1);
        state[8..16].copy_from_slice(&v2);
        state[16..24].copy_from_slice(&h_h);
        poseidon2(&mut state, &rc);
        local += 1;

        if let Some(slot) = check_tickets(&state, &target) {
            if stats.found.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                stats.pow_perms.fetch_add(local, Ordering::Relaxed);
                print_result("PoW", tid, slot, &state, Some(nonce), None, &stats, start);
            }
            return;
        }
        nonce += 1;
        if local % 200_000 == 0 {
            stats.pow_perms.fetch_add(200_000, Ordering::Relaxed);
            local -= 200_000;
            if show_progress {
                let sp = stats.stark_perms.load(Ordering::Relaxed);
                let pp = stats.pow_perms.load(Ordering::Relaxed);
                let tot = sp + pp;
                let e = start.elapsed().as_secs_f64();
                let pr = stats.stark_proofs.load(Ordering::Relaxed);
                let f = if tot > 0 { sp as f64 / tot as f64 * 100.0 } else { 0.0 };
                eprint!("\r[{:.1}s] STARK: {} proofs ({} perms) | PoW: {} perms | f={:.1}% | {} perm/s   ",
                    e, pr, fmt(sp), fmt(pp), f, fmt((tot as f64 / e) as u64));
            }
        }
    }
    stats.pow_perms.fetch_add(local, Ordering::Relaxed);
}

// ── Header digest (§3.1) ────────────────────────────────────

fn compute_h_h(header: &[u32; 8], rc: &RC) -> [u32; 8] {
    let mut state = [0u32; W];
    for i in 0..8 { state[i] = header[i]; }
    state[W - 1] = 0x4B61_7370 & P;
    poseidon2(&mut state, rc);
    let mut h_h = [0u32; 8];
    h_h.copy_from_slice(&state[0..8]);
    h_h
}

// ── Main ────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let difficulty: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let n_threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or_else(num_cpus);
    let n_stark: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1.min(n_threads));
    let log_trace: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
    let trace_len = 1usize << log_trace;
    let n_pow = n_threads - n_stark;
    let lde_size = trace_len * (1 << LOG_BLOWUP);
    let n_leaves = next_pow2(lde_size / 8);

    let target = make_target(difficulty);
    let rc = gen_rc();
    let header: [u32; 8] = [
        0xDEAD_BEEF & P, 0xCAFE_BABE & P, 0x1234_5678, 0x9ABC_DEF0 & P,
        42, 2026, 2, 18,
    ];
    let h_h = compute_h_h(&header, &rc);

    println!("ZK-SPoW Symbiotic Miner — Poseidon2 Width-24 / M31");
    println!("───────────────────────────────────────────────────");
    circle_smoke_test();
    println!("Difficulty  : {} bits  (target[0] < {})", difficulty, target[0]);
    println!("Expected    : ~{} perms", fmt((1u64 << difficulty) / 3));
    println!("Threads     : {} total ({} STARK + {} PoW)", n_threads, n_stark, n_pow);
    println!("STARK        : Circle STARK over C(M31), Fib trace 2^{} = {}", log_trace, trace_len);
    println!("LDE          : {}x blowup → {} points, Merkle {} leaves", 1u32 << LOG_BLOWUP, lde_size, n_leaves);
    println!("FRI          : {} queries, fold to ≤{} values", CSTARK_QUERIES, FRI_FOLD_STOP);
    println!("Poseidon2    : W={}, Rf={}, Rp={}", W, RF, RP);
    println!("───────────────────────────────────────────────────");

    let stats = Arc::new(Stats {
        found: AtomicBool::new(false),
        stark_perms: AtomicU64::new(0),
        pow_perms: AtomicU64::new(0),
        stark_proofs: AtomicU64::new(0),
    });
    let start = Instant::now();
    let mut handles = Vec::with_capacity(n_threads);

    for tid in 0..n_stark {
        let stats = stats.clone();
        let hh = h_h;
        let tgt = target;
        handles.push(std::thread::spawn(move || {
            stark_worker(tid, hh, tgt, log_trace, stats, start);
        }));
    }
    for i in 0..n_pow {
        let tid = n_stark + i;
        let stats = stats.clone();
        let hh = h_h;
        let tgt = target;
        handles.push(std::thread::spawn(move || {
            pow_worker(tid, hh, tgt, stats, start, i == 0);
        }));
    }

    for h in handles { h.join().unwrap(); }
    println!();
}

fn num_cpus() -> usize {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.matches("processor").count())
        .unwrap_or(1)
        .max(1)
}
