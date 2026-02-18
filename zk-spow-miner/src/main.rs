//! ZK-SPoW Symbiotic Miner — Poseidon2 Width-24 over M31
//!
//! Runs STARK Merkle tree construction and PoW nonce grinding in parallel.
//! Every Poseidon2 permutation (both STARK and PoW) produces 3 PoW tickets.
//! A block is found when any ticket < target.

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
    P - 2, // -2 mod p
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

/// M4 from Poseidon2 paper: [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]]
#[inline]
fn m4(x: &mut [u32; 4]) {
    let (a, b, c, d) = (x[0], x[1], x[2], x[3]);
    let ab = add(a, b);
    let cd = add(c, d);
    let bb = add(b, b);
    let dd = add(d, d);
    let t2 = add(bb, cd);
    let t3 = add(dd, ab);
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
    for i in 1..W {
        s[i] = add(sum, mul(DIAG[i], orig[i]));
    }
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

// ── Merkle compression ──────────────────────────────────────

/// Poseidon2 compression (§3.3): state = [left ‖ right ‖ h_H], permute.
/// Compression function mode — all 24 elements visible, no hidden capacity.
/// Returns (merkle_parent = state[0..8], full state for ticket checking).
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

// ── Ticket & target ─────────────────────────────────────────

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

// ── Simple RNG ──────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn m31(&mut self) -> u32 { (self.next() as u32) & P }
}

// ── Shared counters ─────────────────────────────────────────

struct Stats {
    found: AtomicBool,
    stark_perms: AtomicU64,
    pow_perms: AtomicU64,
    stark_proofs: AtomicU64,
}

fn print_result(
    source: &str,
    tid: usize,
    slot: usize,
    state: &[u32; W],
    nonce: Option<u64>,
    stats: &Stats,
    start: Instant,
) {
    let sp = stats.stark_perms.load(Ordering::Relaxed);
    let pp = stats.pow_perms.load(Ordering::Relaxed);
    let total = sp + pp;
    let elapsed = start.elapsed().as_secs_f64();
    let proofs = stats.stark_proofs.load(Ordering::Relaxed);
    let f = if total > 0 { sp as f64 / total as f64 * 100.0 } else { 0.0 };

    eprintln!("\r                                                                  ");
    println!("*** BLOCK FOUND ({}) ***", source);
    if let Some(n) = nonce { println!("Nonce       : {}", n); }
    println!("Thread      : {} ({})", tid, source);
    println!("Ticket slot : {} (of 0,1,2)", slot);
    print!("Ticket      : [");
    for j in 0..8 {
        if j > 0 { print!(", "); }
        print!("{}", state[slot * 8 + j]);
    }
    println!("]");
    println!("────────────────────────────────────────");
    println!("STARK proofs: {}", proofs);
    println!("STARK perms : {}", fmt(sp));
    println!("PoW perms   : {}", fmt(pp));
    println!("Total perms : {}", fmt(total));
    println!("STARK frac  : {:.1}%", f);
    println!("Time        : {:.3}s", elapsed);
    println!("Perm rate   : {} perm/s", fmt((total as f64 / elapsed) as u64));
}

fn fmt(n: u64) -> String {
    if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}K", n as f64 / 1e3) }
    else { format!("{}", n) }
}

// ── STARK worker ────────────────────────────────────────────

fn stark_worker(
    tid: usize,
    h_h: [u32; 8],
    target: [u32; 8],
    tree_sz: usize,
    stats: Arc<Stats>,
    start: Instant,
) {
    let rc = gen_rc();
    let mut rng = Rng(0xABCD_EF01_2345_6789 ^ (tid as u64).wrapping_mul(0x9E3779B97F4A7C15));

    while !stats.found.load(Ordering::Relaxed) {
        // Generate trace leaves (simulates transaction/trace data)
        let mut leaves: Vec<[u32; 8]> = Vec::with_capacity(tree_sz);
        for _ in 0..tree_sz {
            let mut leaf = [0u32; 8];
            for e in &mut leaf { *e = rng.m31(); }
            leaves.push(leaf);
        }

        // Build Merkle tree bottom-up, checking tickets at every node
        let mut level = leaves;
        let mut local = 0u64;
        let mut win: Option<([u32; W], usize)> = None;

        while level.len() > 1 && win.is_none() && !stats.found.load(Ordering::Relaxed) {
            let mut next = Vec::with_capacity((level.len() + 1) / 2);
            for pair in level.chunks(2) {
                if pair.len() < 2 { next.push(pair[0]); continue; }
                let (hash, state) = compress(&pair[0], &pair[1], &h_h, &rc);
                local += 1;
                if let Some(slot) = check_tickets(&state, &target) {
                    win = Some((state, slot));
                    break;
                }
                next.push(hash);
            }
            level = next;
        }

        stats.stark_perms.fetch_add(local, Ordering::Relaxed);

        if let Some((state, slot)) = win {
            if stats.found.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                print_result("STARK", tid, slot, &state, None, &stats, start);
            }
            return;
        }
        stats.stark_proofs.fetch_add(1, Ordering::Relaxed);
    }
}

// ── PoW worker (§4.2) ───────────────────────────────────────
//
// Pure PoW mode: S = Poseidon2_π(v₁ ‖ v₂ ‖ h_H)
//   v₁, v₂ ∈ F_p^8 (16 M31 elements = 64-byte nonce)
//   h_H ∈ F_p^8 (header digest)
// Compression function mode — all 24 visible, no capacity.

fn pow_worker(
    tid: usize,
    h_h: [u32; 8],
    target: [u32; 8],
    stats: Arc<Stats>,
    start: Instant,
    show_progress: bool,
) {
    let rc = gen_rc();
    // v₂[0] = thread ID → each thread has unique nonce subspace
    let mut v2 = [0u32; 8];
    v2[0] = (tid as u32) & P;
    let mut nonce: u64 = 0;
    let mut local: u64 = 0;

    while !stats.found.load(Ordering::Relaxed) {
        // Map nonce counter → v₁ ∈ F_p^8
        let mut v1 = [0u32; 8];
        v1[0] = (nonce as u32) & P;
        v1[1] = ((nonce >> 31) as u32) & P;
        v1[2] = ((nonce >> 62) as u32) & 0x3;

        // S = Poseidon2_π(v₁ ‖ v₂ ‖ h_H)  — compression function
        let mut state = [0u32; W];
        state[..8].copy_from_slice(&v1);
        state[8..16].copy_from_slice(&v2);
        state[16..24].copy_from_slice(&h_h);

        poseidon2(&mut state, &rc);
        local += 1;

        if let Some(slot) = check_tickets(&state, &target) {
            if stats.found.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                stats.pow_perms.fetch_add(local, Ordering::Relaxed);
                print_result("PoW", tid, slot, &state, Some(nonce), &stats, start);
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
//
// h_H = PoseidonSponge(H excluding nonce) ∈ F_p^8
// Simplified: single Poseidon2 permutation of padded header → first 8 elements.
// Production: Width-16 sponge (Stwo standard) over full serialized header.

fn compute_h_h(header: &[u32; 8], rc: &RC) -> [u32; 8] {
    let mut state = [0u32; W];
    for i in 0..8 { state[i] = header[i]; }
    // Domain separator in last element to distinguish from data
    state[W - 1] = 0x4B61_7370 & P; // "Kasp" truncated
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
    let tree_sz: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_pow = n_threads - n_stark;

    let target = make_target(difficulty);
    let rc = gen_rc();

    // Demo header (prev_hash ‖ merkle_root ‖ timestamp ‖ bits)
    let header: [u32; 8] = [
        0xDEAD_BEEF & P, 0xCAFE_BABE & P, 0x1234_5678, 0x9ABC_DEF0 & P,
        42, 2026, 2, 18,
    ];
    // h_H = PoseidonSponge(header excluding nonce) → 8 M31 elements
    let h_h = compute_h_h(&header, &rc);

    println!("ZK-SPoW Symbiotic Miner — Poseidon2 Width-24 / M31");
    println!("───────────────────────────────────────────────────");
    println!("Difficulty  : {} bits  (target[0] < {})", difficulty, target[0]);
    println!("Expected    : ~{} perms", fmt((1u64 << difficulty) / 3));
    println!("Threads     : {} total ({} STARK + {} PoW)", n_threads, n_stark, n_pow);
    println!("Tree size   : {} leaves ({} hashes/tree)", tree_sz, tree_sz - 1);
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

    // STARK workers: Poseidon2_π(n_L ‖ n_R ‖ h_H)  §4.1
    for tid in 0..n_stark {
        let stats = stats.clone();
        let hh = h_h;
        let tgt = target;
        handles.push(std::thread::spawn(move || {
            stark_worker(tid, hh, tgt, tree_sz, stats, start);
        }));
    }

    // PoW workers: Poseidon2_π(v₁ ‖ v₂ ‖ h_H)  §4.2
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
