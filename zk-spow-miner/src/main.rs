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
    trace_len: usize, stats: Arc<Stats>, start: Instant,
) {
    let rc = gen_rc();
    let dom = trace_len * BLOWUP;
    let weights = barycentric_weights(trace_len);
    let inv_tab = precompute_inv_table(dom);
    let mut proof_id: u32 = tid as u32;

    while !stats.found.load(Ordering::Relaxed) {
        let trace = fib_trace(trace_len, proof_id, 1);
        let (proof, perms, ticket) = stark_prove(&trace, &h_h, &rc, &target, &weights, &inv_tab);
        stats.stark_perms.fetch_add(perms, Ordering::Relaxed);

        let valid = stark_verify(&proof, &h_h, &rc);
        if !valid { eprintln!("BUG: proof #{} failed verification!", proof_id); }
        stats.stark_proofs.fetch_add(1, Ordering::Relaxed);

        if let Some((state, slot)) = ticket {
            if stats.found.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                let info = format!("Fib(a0={}, len={}) LDE={}x Q_deg<2 verified={}", proof_id, trace_len, BLOWUP, valid);
                print_result("STARK", tid, slot, &state, None, Some(&info), &stats, start);
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
    let trace_len: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(256);
    let n_pow = n_threads - n_stark;
    let n_leaves = next_pow2(trace_len * BLOWUP / 8);

    let target = make_target(difficulty);
    let rc = gen_rc();
    let header: [u32; 8] = [
        0xDEAD_BEEF & P, 0xCAFE_BABE & P, 0x1234_5678, 0x9ABC_DEF0 & P,
        42, 2026, 2, 18,
    ];
    let h_h = compute_h_h(&header, &rc);

    println!("ZK-SPoW Symbiotic Miner — Poseidon2 Width-24 / M31");
    println!("───────────────────────────────────────────────────");
    println!("Difficulty  : {} bits  (target[0] < {})", difficulty, target[0]);
    println!("Expected    : ~{} perms", fmt((1u64 << difficulty) / 3));
    println!("Threads     : {} total ({} STARK + {} PoW)", n_threads, n_stark, n_pow);
    println!("STARK AIR   : Fibonacci over M31 (trace_len={})", trace_len);
    println!("Polynomial  : degree<{}, LDE {}x on {} points", trace_len, BLOWUP, trace_len * BLOWUP);
    println!("Commitment  : 2 Poseidon2 Merkle trees ({} leaves each)", n_leaves);
    println!("Queries     : {} per proof, quotient degree<2 linearity test", N_QUERIES);
    println!("Params      : W={}, Rf={}, Rp={}", W, RF, RP);
    println!("I/O layout  : [v1/left(8) | v2/right(8) | h_H(8)] — compression, no capacity");
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
            stark_worker(tid, hh, tgt, trace_len, stats, start);
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
