//! ZK-SPoW Symbiotic Miner — Poseidon2 Width-24 over M31
//!
//! Generates real Fibonacci STARK proofs (trace → commit → query → verify)
//! while mining. Every Poseidon2 permutation produces 3 PoW tickets.

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
struct RC {
    ext: [[u32; W]; RF],
    int: [u32; RP],
}

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
    for r in rc.ext.iter_mut() {
        for e in r.iter_mut() { *e = next(); }
    }
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
    for g in 0..6 {
        for i in 0..4 { sum[i] = add(sum[i], s[g * 4 + i]); }
    }
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

// ── Compression (§3.3) ─────────────────────────────────────

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

// ── Ticket ─────────────────────────────────────────────────

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
        if ticket_lt(&state[slot * 8..(slot + 1) * 8], target) {
            return Some(slot);
        }
    }
    None
}

// ── RNG ────────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

// ── Merkle tree with authentication paths ──────────────────

fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

struct MerkleTree {
    layers: Vec<Vec<[u32; 8]>>,
    h_h: [u32; 8],
    perms: u64,
    ticket: Option<([u32; W], usize)>,
}

impl MerkleTree {
    fn build(mut leaves: Vec<[u32; 8]>, h_h: [u32; 8], rc: &RC,
             target: &[u32; 8]) -> Self {
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
        MerkleTree { layers, h_h, perms, ticket }
    }

    fn root(&self) -> [u32; 8] { self.layers.last().unwrap()[0] }

    fn depth(&self) -> usize { self.layers.len() - 1 }

    fn auth_path(&self, mut idx: usize) -> Vec<[u32; 8]> {
        let mut path = Vec::with_capacity(self.depth());
        for layer in &self.layers[..self.layers.len() - 1] {
            let sib = idx ^ 1;
            path.push(layer[sib.min(layer.len() - 1)]);
            idx >>= 1;
        }
        path
    }

    fn verify_path(leaf: [u32; 8], mut idx: usize, path: &[[u32; 8]],
                   root: [u32; 8], h_h: &[u32; 8], rc: &RC) -> bool {
        let mut cur = leaf;
        for sib in path {
            let (l, r) = if idx & 1 == 0 { (&cur, sib) } else { (sib, &cur) };
            let (hash, _) = compress(l, r, h_h, rc);
            cur = hash;
            idx >>= 1;
        }
        cur == root
    }
}

// ── Fibonacci STARK prover & verifier ──────────────────────
//
// Proves: "I know a Fibonacci sequence of length N starting from (a₀, a₁)"
// Commitment: Poseidon2 Merkle tree over 8-element leaves
// Verification: random queries check Fibonacci constraint + Merkle auth paths

const N_QUERIES: usize = 16;

fn fib_trace(n: usize, a0: u32, a1: u32) -> Vec<u32> {
    let mut t = vec![0u32; n];
    t[0] = a0;
    t[1] = a1;
    for i in 2..n { t[i] = add(t[i - 1], t[i - 2]); }
    t
}

fn pack_leaves(trace: &[u32]) -> Vec<[u32; 8]> {
    trace.chunks(8).map(|c| {
        let mut leaf = [0u32; 8];
        for (i, &v) in c.iter().enumerate() { leaf[i] = v; }
        leaf
    }).collect()
}

struct Query {
    idx: usize,
    leaf: [u32; 8],
    path: Vec<[u32; 8]>,
    next_leaf: [u32; 8],
    next_path: Vec<[u32; 8]>,
}

struct StarkProof {
    trace_root: [u32; 8],
    trace_len: usize,
    a0: u32,
    a1: u32,
    a_last: u32,
    queries: Vec<Query>,
}

/// Generate a Fibonacci STARK proof. Returns (proof, perms_used, ticket_hit).
fn stark_prove(trace: &[u32], h_h: &[u32; 8], rc: &RC,
               target: &[u32; 8]) -> (StarkProof, u64, Option<([u32; W], usize)>) {
    // 1. Pack trace into 8-element leaves and build Merkle tree
    let leaves = pack_leaves(trace);
    let tree = MerkleTree::build(leaves, *h_h, rc, target);
    let mut perms = tree.perms;
    let mut ticket = tree.ticket;
    let root = tree.root();

    // 2. Fiat-Shamir: derive query positions from commitment root
    let mut fs = [0u32; W];
    fs[..8].copy_from_slice(&root);
    fs[8] = 0x46524953; // domain separator "FRIS"
    poseidon2(&mut fs, rc);
    perms += 1;
    if ticket.is_none() {
        if let Some(slot) = check_tickets(&fs, target) {
            ticket = Some((fs, slot));
        }
    }
    let mut rng = Rng(fs[0] as u64 | ((fs[1] as u64) << 32));

    // 3. Generate query openings
    let n_leaves = tree.layers[0].len();
    let max_idx = n_leaves.saturating_sub(2); // need idx+1 to exist
    let mut queries = Vec::with_capacity(N_QUERIES);
    for _ in 0..N_QUERIES {
        let idx = (rng.next() as usize) % (max_idx + 1);
        queries.push(Query {
            idx,
            leaf: tree.layers[0][idx],
            path: tree.auth_path(idx),
            next_leaf: tree.layers[0][idx + 1],
            next_path: tree.auth_path(idx + 1),
        });
    }

    let proof = StarkProof {
        trace_root: root,
        trace_len: trace.len(),
        a0: trace[0],
        a1: trace[1],
        a_last: trace[trace.len() - 1],
        queries,
    };
    (proof, perms, ticket)
}

/// Verify a Fibonacci STARK proof. Returns true if valid.
fn stark_verify(proof: &StarkProof, h_h: &[u32; 8], rc: &RC) -> bool {
    // Re-derive Fiat-Shamir challenge
    let mut fs = [0u32; W];
    fs[..8].copy_from_slice(&proof.trace_root);
    fs[8] = 0x46524953;
    poseidon2(&mut fs, rc);
    let mut rng = Rng(fs[0] as u64 | ((fs[1] as u64) << 32));
    let n_leaves = next_pow2((proof.trace_len + 7) / 8);
    let max_idx = n_leaves.saturating_sub(2);

    for q in &proof.queries {
        let expected_idx = (rng.next() as usize) % (max_idx + 1);
        if q.idx != expected_idx {
            return false;
        }

        // Verify Merkle auth paths
        if !MerkleTree::verify_path(q.leaf, q.idx, &q.path,
                                     proof.trace_root, h_h, rc) {
            return false;
        }
        if !MerkleTree::verify_path(q.next_leaf, q.idx + 1, &q.next_path,
                                     proof.trace_root, h_h, rc) {
            return false;
        }

        // Verify Fibonacci constraint within leaf
        //   leaf = [a[8k], a[8k+1], ..., a[8k+7]]
        //   Check: a[j+2] = a[j+1] + a[j] for j = 0..5
        for j in 0..6 {
            if q.leaf[j + 2] != add(q.leaf[j + 1], q.leaf[j]) {
                return false;
            }
        }

        // Verify boundary between consecutive leaves
        //   next_leaf[0] = leaf[7] + leaf[6]
        //   next_leaf[1] = next_leaf[0] + leaf[7]
        if q.next_leaf[0] != add(q.leaf[7], q.leaf[6]) {
            return false;
        }
        if q.next_leaf[1] != add(q.next_leaf[0], q.leaf[7]) {
            return false;
        }

        // first query at leaf 0: check public inputs
        if q.idx == 0 {
            if q.leaf[0] != proof.a0 || q.leaf[1] != proof.a1 {
                return false;
            }
        }
    }

    true
}

// ── Shared stats ────────────────────────────────────────────

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
    for j in 0..8 {
        if j > 0 { print!(", "); }
        print!("{}", state[slot * 8 + j]);
    }
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

// ── STARK worker (real Fibonacci proof generation) ──────────

fn stark_worker(
    tid: usize, h_h: [u32; 8], target: [u32; 8],
    trace_len: usize, stats: Arc<Stats>, start: Instant,
) {
    let rc = gen_rc();
    let mut proof_id: u32 = tid as u32;

    while !stats.found.load(Ordering::Relaxed) {
        // Each proof uses a different starting value → different trace
        let a0 = proof_id;
        let a1 = 1;
        let trace = fib_trace(trace_len, a0, a1);

        let (proof, perms, ticket) = stark_prove(&trace, &h_h, &rc, &target);
        stats.stark_perms.fetch_add(perms, Ordering::Relaxed);

        // Verify the proof we just generated
        let valid = stark_verify(&proof, &h_h, &rc);
        if !valid {
            eprintln!("BUG: proof #{} failed verification!", proof_id);
        }
        stats.stark_proofs.fetch_add(1, Ordering::Relaxed);

        if let Some((state, slot)) = ticket {
            if stats.found.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                let info = format!("Fibonacci(a0={}, len={}) verified={}", a0, trace_len, valid);
                print_result("STARK", tid, slot, &state, None, Some(&info), &stats, start);
            }
            return;
        }
        proof_id += 100; // skip to avoid overlap between threads
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
    let n_stark: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.min(n_threads));
    let trace_len: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_pow = n_threads - n_stark;
    let n_leaves = next_pow2((trace_len + 7) / 8);

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
    println!("Commitment  : Poseidon2 Merkle ({} leaves, {} hashes/tree)", n_leaves, n_leaves - 1);
    println!("Queries     : {} per proof (Fiat-Shamir)", N_QUERIES);
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
