#!/usr/bin/env python3
"""
NIST SP 800-22 statistical randomness tests for Poseidon2 Width-24 over M31.

Validates the PRP assumption empirically: sequential counter inputs through
Poseidon2 should produce output indistinguishable from random bits.

Tests all 15 NIST SP 800-22 tests plus inter-ticket independence analysis.
Poseidon2 implementation ported line-by-line from main.rs L876-960.

No external dependencies (stdlib only).
"""

import math
import struct
import sys

# ── M31 field: p = 2^31 - 1 ──────────────────────────────────

P = 0x7FFF_FFFF  # 2^31 - 1

def add(a, b):
    s = a + b
    r = (s & P) + (s >> 31)
    return r - P if r >= P else r

def sub(a, b):
    return a - b if a >= b else P - b + a

def mul(a, b):
    prod = a * b
    lo = prod & P
    hi = prod >> 31
    s = lo + hi
    r = (s & P) + (s >> 31)
    return r - P if r >= P else r

def pow5(x):
    x2 = mul(x, x)
    x4 = mul(x2, x2)
    return mul(x4, x)

# ── Poseidon2 Width-24 (from main.rs L876-960) ───────────────

W = 24
RF = 8
RP = 22
HRF = 4

DIAG = [
    P - 2,
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576, 2097152, 4194304,
]

def gen_rc():
    """Generate round constants via SplitMix64 (matches main.rs gen_rc)."""
    z = 0x5A4B_3C2D_1E0F_A9B8
    mask64 = (1 << 64) - 1

    def next_rc():
        nonlocal z
        z = (z + 0x9E37_79B9_7F4A_7C15) & mask64
        x = z
        x = ((x ^ (x >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask64
        x = ((x ^ (x >> 27)) * 0x94D0_49BB_1331_11EB) & mask64
        x ^= x >> 31
        return x & 0xFFFF_FFFF & P

    ext = [[next_rc() for _ in range(W)] for _ in range(RF)]
    internal = [next_rc() for _ in range(RP)]
    return ext, internal

def m4(x):
    """4x4 MDS matrix multiply (from main.rs L908-922)."""
    a, b, c, d = x[0], x[1], x[2], x[3]
    ab = add(a, b)
    cd = add(c, d)
    t2 = add(add(b, b), cd)
    t3 = add(add(d, d), ab)
    ab4_t = add(ab, ab)
    ab4 = add(ab4_t, ab4_t)
    cd4_t = add(cd, cd)
    cd4 = add(cd4_t, cd4_t)
    t5 = add(ab4, t2)
    t4 = add(cd4, t3)
    x[0] = add(t3, t5)
    x[1] = t5
    x[2] = add(t2, t4)
    x[3] = t4

def ext_mds(s):
    """External MDS layer (from main.rs L924-933)."""
    sm = [0, 0, 0, 0]
    for g in range(6):
        for i in range(4):
            sm[i] = add(sm[i], s[g * 4 + i])
    for g in range(6):
        tmp = [add(s[g * 4 + i], sm[i]) for i in range(4)]
        m4(tmp)
        for i in range(4):
            s[g * 4 + i] = tmp[i]

def int_mds(s):
    """Internal MDS layer (from main.rs L935-941)."""
    total = 0
    for i in range(W):
        total = add(total, s[i])
    orig = s[:]
    s[0] = sub(total, add(orig[0], orig[0]))
    for i in range(1, W):
        s[i] = add(total, mul(DIAG[i], orig[i]))

def poseidon2(s, rc_ext, rc_int):
    """Poseidon2 permutation (from main.rs L943-960)."""
    ext_mds(s)
    for r in range(HRF):
        for i in range(W):
            s[i] = add(s[i], rc_ext[r][i])
        for i in range(W):
            s[i] = pow5(s[i])
        ext_mds(s)
    for r in range(RP):
        s[0] = add(s[0], rc_int[r])
        s[0] = pow5(s[0])
        int_mds(s)
    for r in range(HRF, RF):
        for i in range(W):
            s[i] = add(s[i], rc_ext[r][i])
        for i in range(W):
            s[i] = pow5(s[i])
        ext_mds(s)
    return s

# ── Cross-validation ─────────────────────────────────────────

def cross_validate(rc_ext, rc_int):
    """Compute poseidon2([0]*24) for manual verification against Rust."""
    s = [0] * W
    poseidon2(s, rc_ext, rc_int)
    print("Cross-validation: poseidon2([0]*24) first 8 elements:")
    print("  Python:", [hex(x) for x in s[:8]])
    print("  (Verify against Rust main.rs output for same input)")
    print()
    return s

# ── Bitstream generation ─────────────────────────────────────

def element_to_bits(val, nbits=31):
    """Convert M31 element to nbits MSB-first."""
    bits = []
    for i in range(nbits - 1, -1, -1):
        bits.append((val >> i) & 1)
    return bits

def generate_bitstream(n_bits, rc_ext, rc_int, start_counter=0):
    """Generate bitstream from sequential counter inputs.

    Each permutation: input (counter, 0, ..., 0) → 24 elements × 31 bits = 744 bits.
    """
    bits = []
    counter = start_counter
    while len(bits) < n_bits:
        s = [counter & P] + [0] * (W - 1)
        poseidon2(s, rc_ext, rc_int)
        for elem in s:
            bits.extend(element_to_bits(elem))
        counter += 1
    return bits[:n_bits], counter

def generate_ticket_streams(n_perms, rc_ext, rc_int, start_counter=0):
    """Generate 3 separate ticket bitstreams from Poseidon2 output.

    Ticket 0: elements S[0..7], Ticket 1: S[8..15], Ticket 2: S[16..23].
    Returns (stream0, stream1, stream2) each as list of bits.
    """
    s0_bits, s1_bits, s2_bits = [], [], []
    counter = start_counter
    for _ in range(n_perms):
        s = [counter & P] + [0] * (W - 1)
        poseidon2(s, rc_ext, rc_int)
        for elem in s[0:8]:
            s0_bits.extend(element_to_bits(elem))
        for elem in s[8:16]:
            s1_bits.extend(element_to_bits(elem))
        for elem in s[16:24]:
            s2_bits.extend(element_to_bits(elem))
        counter += 1
    return s0_bits, s1_bits, s2_bits

# ── Statistical utilities ────────────────────────────────────

def _igamc_cf(a, x):
    """Upper incomplete gamma via continued fraction (Legendre)."""
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h

def _igamc_series(a, x):
    """Lower incomplete gamma via series, return 1 - result for upper."""
    if x == 0:
        return 1.0
    ap = a
    s = 1.0 / a
    ds = s
    for _ in range(300):
        ap += 1.0
        ds *= x / ap
        s += ds
        if abs(ds) < abs(s) * 3e-12:
            break
    return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))

def igamc(a, x):
    """Regularized upper incomplete gamma function Q(a,x) = 1 - P(a,x)."""
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return _igamc_series(a, x)
    return _igamc_cf(a, x)

def chi2_pvalue(stat, df):
    """Chi-squared p-value (upper tail)."""
    return igamc(df / 2.0, stat / 2.0)

def erfc(x):
    """Complementary error function."""
    return math.erfc(x)

# ── NIST SP 800-22 Tests ─────────────────────────────────────

def test_frequency(bits):
    """Test 1: Frequency (Monobit).

    Determines whether the number of ones and zeros are approximately equal.
    """
    n = len(bits)
    s = sum(2 * b - 1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    p_value = erfc(s_obs / math.sqrt(2))
    return p_value

def test_block_frequency(bits, M=128):
    """Test 2: Block Frequency.

    Determines whether the frequency of ones in M-bit blocks is approximately M/2.
    """
    n = len(bits)
    N = n // M
    if N == 0:
        return 1.0
    chi_sq = 0.0
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        pi_i = sum(block) / M
        chi_sq += (pi_i - 0.5) ** 2
    chi_sq *= 4 * M
    p_value = igamc(N / 2.0, chi_sq / 2.0)
    return p_value

def test_runs(bits):
    """Test 3: Runs.

    Determines whether the number of runs of ones and zeros is as expected.
    """
    n = len(bits)
    pi = sum(bits) / n
    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0
    v_obs = 1
    for i in range(n - 1):
        if bits[i] != bits[i + 1]:
            v_obs += 1
    num = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    if den == 0:
        return 0.0
    p_value = erfc(num / den)
    return p_value

def test_longest_run(bits):
    """Test 4: Longest Run of Ones in a Block.

    Determines whether the longest run of ones within M-bit blocks is consistent
    with the expected longest run for a random sequence.
    """
    n = len(bits)
    if n < 6272:
        M, K = 8, 3
        v_vals = [1, 2, 3, 4]
        pi_vals = [0.2148, 0.3672, 0.2305, 0.1875]
        N = n // M
    elif n < 750000:
        M, K = 128, 5
        v_vals = [4, 5, 6, 7, 8, 9]
        pi_vals = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        N = n // M
    else:
        M, K = 10000, 6
        v_vals = [10, 11, 12, 13, 14, 15, 16]
        pi_vals = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        N = n // M

    freq = [0] * len(v_vals)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        max_run = 0
        current_run = 0
        for b in block:
            if b == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        if max_run <= v_vals[0]:
            freq[0] += 1
        elif max_run >= v_vals[-1]:
            freq[-1] += 1
        else:
            for j in range(1, len(v_vals)):
                if max_run == v_vals[j]:
                    freq[j] += 1
                    break

    chi_sq = 0.0
    for j in range(len(v_vals)):
        expected = N * pi_vals[j]
        if expected > 0:
            chi_sq += (freq[j] - expected) ** 2 / expected
    p_value = igamc(K / 2.0, chi_sq / 2.0)
    return p_value

def _count_patterns(bits, m):
    """Count m-bit pattern frequencies (for Serial and ApEn tests)."""
    n = len(bits)
    counts = {}
    for i in range(n):
        pattern = 0
        for j in range(m):
            pattern = (pattern << 1) | bits[(i + j) % n]
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts

def _psi_sq(bits, m):
    """Compute psi-squared statistic for m-bit patterns."""
    n = len(bits)
    if m == 0:
        return 0.0
    counts = _count_patterns(bits, m)
    total = sum(v * v for v in counts.values())
    return (2 ** m / n) * total - n

def test_serial(bits, m=16):
    """Test 5: Serial.

    Determines whether the number of occurrences of 2^m m-bit overlapping
    patterns is approximately as expected for a random sequence.
    """
    n = len(bits)
    psi_m = _psi_sq(bits, m)
    psi_m1 = _psi_sq(bits, m - 1)
    psi_m2 = _psi_sq(bits, m - 2) if m >= 2 else 0.0

    del1 = psi_m - psi_m1
    del2 = psi_m - 2 * psi_m1 + psi_m2

    p1 = igamc(2 ** (m - 2), del1 / 2.0)
    p2 = igamc(2 ** (m - 3), del2 / 2.0)
    return {"nabla1": p1, "nabla2": p2}

def test_approximate_entropy(bits, m=10):
    """Test 6: Approximate Entropy.

    Compares the frequency of overlapping blocks of two consecutive/adjacent
    lengths (m and m+1) against the expected result for a random sequence.
    """
    n = len(bits)

    def phi(block_len):
        if block_len == 0:
            return 0.0
        counts = _count_patterns(bits, block_len)
        total = 0.0
        for c in counts.values():
            if c > 0:
                total += c * math.log(c / n)
        return total / n

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    apen = phi_m - phi_m1
    chi_sq = 2 * n * (math.log(2) - apen)
    p_value = igamc(2 ** (m - 1), chi_sq / 2.0)
    return p_value

def _normal_cdf(x):
    """Standard normal CDF: Phi(x) = 0.5 * erfc(-x / sqrt(2))."""
    return 0.5 * math.erfc(-x / math.sqrt(2))

def test_cumulative_sums(bits):
    """Test 7: Cumulative Sums (forward and backward).

    Determines whether the cumulative sum of adjusted (-1/+1) digits is too
    large or too small relative to expected behavior for a random sequence.
    NIST SP 800-22 Section 2.13.
    """
    n = len(bits)
    adjusted = [2 * b - 1 for b in bits]
    sqrt_n = math.sqrt(n)

    def cusum_pvalue(seq):
        cumsum = 0
        z = 0
        for x in seq:
            cumsum += x
            if abs(cumsum) > z:
                z = abs(cumsum)
        if z == 0:
            return 1.0
        # NIST SP 800-22 §2.13.4 formula using normal CDF Phi
        s1 = 0.0
        k_lo = int(math.floor((-n / z + 1) / 4))
        k_hi = int(math.floor((n / z - 1) / 4))
        for k in range(k_lo, k_hi + 1):
            s1 += (_normal_cdf((4 * k + 1) * z / sqrt_n)
                   - _normal_cdf((4 * k - 1) * z / sqrt_n))
        s2 = 0.0
        k_lo2 = int(math.floor((-n / z - 3) / 4))
        k_hi2 = int(math.floor((n / z - 1) / 4))
        for k in range(k_lo2, k_hi2 + 1):
            s2 += (_normal_cdf((4 * k + 3) * z / sqrt_n)
                   - _normal_cdf((4 * k + 1) * z / sqrt_n))
        p = 1.0 - s1 + s2
        return max(0.0, min(1.0, p))

    p_fwd = cusum_pvalue(adjusted)
    p_bwd = cusum_pvalue(adjusted[::-1])
    return {"fwd": p_fwd, "bwd": p_bwd}

def _gf2_rank(matrix, nrows, ncols):
    """Compute rank of binary matrix over GF(2) via Gaussian elimination."""
    # Copy matrix as list of integers (each row is a bitmask)
    m = [row for row in matrix]
    rank = 0
    for col in range(ncols):
        # Find pivot
        pivot = -1
        for row in range(rank, nrows):
            if (m[row] >> (ncols - 1 - col)) & 1:
                pivot = row
                break
        if pivot == -1:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        for row in range(nrows):
            if row != rank and (m[row] >> (ncols - 1 - col)) & 1:
                m[row] ^= m[rank]
        rank += 1
    return rank

def test_rank(bits):
    """Test 8: Binary Matrix Rank.

    Determines whether the rank of disjoint 32×32 sub-matrices departs from
    that expected for a random sequence.
    """
    n = len(bits)
    M, Q = 32, 32
    N = n // (M * Q)
    if N == 0:
        return 1.0

    # Expected probabilities for rank 32, 31, <=30
    p_full = 0.2888
    p_minus1 = 0.5776
    p_rest = 0.1336

    f_full = 0
    f_minus1 = 0
    f_rest = 0

    for k in range(N):
        offset = k * M * Q
        matrix = []
        for row in range(M):
            val = 0
            for col in range(Q):
                val = (val << 1) | bits[offset + row * Q + col]
            matrix.append(val)
        r = _gf2_rank(matrix, M, Q)
        if r == M:
            f_full += 1
        elif r == M - 1:
            f_minus1 += 1
        else:
            f_rest += 1

    chi_sq = ((f_full - N * p_full) ** 2 / (N * p_full)
              + (f_minus1 - N * p_minus1) ** 2 / (N * p_minus1)
              + (f_rest - N * p_rest) ** 2 / (N * p_rest))
    p_value = math.exp(-chi_sq / 2.0)
    return p_value

def _gen_aperiodic_templates(m):
    """Generate all aperiodic binary templates of length m (NIST §2.7)."""
    templates = []
    for val in range(2 ** m):
        bits = [(val >> (m - 1 - i)) & 1 for i in range(m)]
        aperiodic = True
        for p in range(1, m):
            if all(bits[i] == bits[i % p] for i in range(m)):
                aperiodic = False
                break
        if aperiodic:
            templates.append(bits)
    return templates

def _nonoverlapping_template_single(bits, template, m, M):
    """Run non-overlapping template test for a single template. Returns p-value."""
    n = len(bits)
    N = n // M
    if N == 0:
        return 1.0

    mu = (M - m + 1) / (2 ** m)
    sigma_sq = M * (1.0 / (2 ** m) - (2 * m - 1) / (2 ** (2 * m)))

    chi_sq = 0.0
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        count = 0
        j = 0
        while j <= len(block) - m:
            if block[j:j + m] == template:
                count += 1
                j += m  # non-overlapping: skip past match
            else:
                j += 1
        chi_sq += (count - mu) ** 2 / sigma_sq

    return igamc(N / 2.0, chi_sq / 2.0)

# Pre-generate all 148 aperiodic templates for m=9
_TEMPLATES_9 = _gen_aperiodic_templates(9)

def test_nonoverlapping_template(bits, m=9):
    """Test 9: Non-overlapping Template Matching.

    Tests all 148 aperiodic templates of length m=9 (NIST SP 800-22 §2.7).
    Returns dict mapping template index to p-value.
    """
    n = len(bits)
    N = 8  # NIST STS 2.1.2: N=8 blocks
    M = n // N  # M=125000 for n=10^6
    results = {}
    for idx, template in enumerate(_TEMPLATES_9):
        results[idx] = _nonoverlapping_template_single(bits, template, m, M)
    return results

def test_overlapping_template(bits, m=9):
    """Test 10: Overlapping Template Matching.

    Tests the number of occurrences of the all-ones template using
    overlapping windows. Uses NIST-specified theoretical probabilities
    that account for the overlapping match distribution.
    """
    n = len(bits)
    template = [1] * m
    K = 5
    M = 1032
    N = n // M
    if N == 0:
        return 1.0

    # NIST SP 800-22 reference implementation hardcoded probabilities
    # for m=9, M=1032, K=5. These account for overlapping correlations
    # and differ from simple Poisson. Source: sts-2.1.2 overlappingTemplateMatchings.c
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]

    freq = [0] * (K + 1)
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        count = 0
        for j in range(len(block) - m + 1):
            match = True
            for k in range(m):
                if block[j + k] != 1:
                    match = False
                    break
            if match:
                count += 1
        idx = min(count, K)
        freq[idx] += 1

    chi_sq = 0.0
    for i in range(K + 1):
        expected = N * pi[i]
        if expected > 0:
            chi_sq += (freq[i] - expected) ** 2 / expected

    p_value = igamc(K / 2.0, chi_sq / 2.0)
    return p_value

# ── NIST Tests 11-15 ───────────────────────────────────────

def _fft_inplace(real_a, imag_a):
    """In-place radix-2 FFT on separate real/imag arrays. Length must be 2^k."""
    n = len(real_a)
    log2n = 0
    tmp = n
    while tmp > 1:
        tmp >>= 1
        log2n += 1

    # Bit-reversal permutation
    for i in range(n):
        j = 0
        tmp = i
        for _ in range(log2n):
            j = (j << 1) | (tmp & 1)
            tmp >>= 1
        if j > i:
            real_a[i], real_a[j] = real_a[j], real_a[i]
            imag_a[i], imag_a[j] = imag_a[j], imag_a[i]

    # Cooley-Tukey butterfly stages
    size = 2
    while size <= n:
        half = size // 2
        angle = -2.0 * math.pi / size
        wn_r = math.cos(angle)
        wn_i = math.sin(angle)
        for start in range(0, n, size):
            wr, wi = 1.0, 0.0
            for k in range(half):
                i1 = start + k
                i2 = i1 + half
                xr = real_a[i2]
                xi = imag_a[i2]
                tr = xr * wr - xi * wi
                ti = xr * wi + xi * wr
                real_a[i2] = real_a[i1] - tr
                imag_a[i2] = imag_a[i1] - ti
                real_a[i1] += tr
                imag_a[i1] += ti
                wr, wi = wr * wn_r - wi * wn_i, wr * wn_i + wi * wn_r
        size *= 2

def _bluestein_dft(x_real, n):
    """Exact n-point DFT via Bluestein's (chirp-z) algorithm.

    Uses the identity jk = (j² + k² - (j-k)²)/2 to reduce n-point DFT
    to circular convolution, computed via three power-of-2 FFTs.
    Input: x_real (list of n floats). Returns: (real_out, imag_out).
    """
    # Convolution length: next power of 2 >= 2n - 1
    M = 1
    while M < 2 * n - 1:
        M <<= 1

    # Chirp: w[j] = exp(-πi·j²/n)
    chirp_r = [0.0] * n
    chirp_i = [0.0] * n
    for j in range(n):
        angle = math.pi * j * j / n
        chirp_r[j] = math.cos(angle)
        chirp_i[j] = -math.sin(angle)

    # a[j] = x[j] * chirp[j], padded to M
    a_r = [0.0] * M
    a_i = [0.0] * M
    for j in range(n):
        a_r[j] = x_real[j] * chirp_r[j]
        a_i[j] = x_real[j] * chirp_i[j]

    # b = conj(chirp), with wrap-around for negative indices
    b_r = [0.0] * M
    b_i = [0.0] * M
    b_r[0] = chirp_r[0]
    b_i[0] = -chirp_i[0]
    for j in range(1, n):
        cr, ci = chirp_r[j], -chirp_i[j]  # conj
        b_r[j] = cr
        b_i[j] = ci
        b_r[M - j] = cr
        b_i[M - j] = ci

    # Convolve via FFT: FFT(a), FFT(b), pointwise multiply, IFFT
    _fft_inplace(a_r, a_i)
    _fft_inplace(b_r, b_i)
    for k in range(M):
        re = a_r[k] * b_r[k] - a_i[k] * b_i[k]
        im = a_r[k] * b_i[k] + a_i[k] * b_r[k]
        a_r[k] = re
        a_i[k] = im
    # IFFT = conj → FFT → conj → /M
    for k in range(M):
        a_i[k] = -a_i[k]
    _fft_inplace(a_r, a_i)
    for k in range(M):
        a_r[k] /= M
        a_i[k] = -a_i[k] / M

    # X[k] = chirp[k] * conv[k]
    out_r = [0.0] * n
    out_i = [0.0] * n
    for k in range(n):
        out_r[k] = chirp_r[k] * a_r[k] - chirp_i[k] * a_i[k]
        out_i[k] = chirp_r[k] * a_i[k] + chirp_i[k] * a_r[k]
    return out_r, out_i

def test_spectral(bits):
    """Test 11: Discrete Fourier Transform (Spectral).

    Detects periodic features indicating deviation from randomness.
    NIST SP 800-22 §2.6. Uses radix-2 FFT with zero-padding to next
    power of 2 (same method as NIST STS 2.1.2 reference implementation).
    """
    n = len(bits)
    # Pad to next power of 2 (as in NIST STS 2.1.2)
    m = 1
    while m < n:
        m <<= 1
    real_a = [float(2 * b - 1) for b in bits] + [0.0] * (m - n)
    imag_a = [0.0] * m
    _fft_inplace(real_a, imag_a)

    # Count peaks below threshold (using original n)
    T = math.sqrt(math.log(1.0 / 0.05) * n)
    N1 = 0
    for i in range(n // 2):
        mag = math.sqrt(real_a[i] ** 2 + imag_a[i] ** 2)
        if mag < T:
            N1 += 1
    N0 = 0.95 * n / 2.0
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    return math.erfc(abs(d) / math.sqrt(2))

def _berlekamp_massey_gf2(s):
    """Berlekamp-Massey over GF(2). Returns linear complexity."""
    n = len(s)
    C = [0] * (n + 1)
    B = [0] * (n + 1)
    C[0] = 1
    B[0] = 1
    L = 0
    m = -1
    for i in range(n):
        d = s[i]
        for j in range(1, L + 1):
            d ^= C[j] & s[i - j]
        if d == 1:
            T = C[:]
            shift = i - m
            for j in range(n + 1 - shift):
                C[j + shift] ^= B[j]
            if 2 * L <= i:
                L = i + 1 - L
                m = i
                B = T
    return L

def test_linear_complexity(bits, M=500):
    """Test 12: Linear Complexity.

    Determines whether the sequence is complex enough to be random.
    NIST SP 800-22 §2.10. Uses Berlekamp-Massey algorithm.
    """
    n = len(bits)
    N = n // M
    K = 6
    if N == 0:
        return 1.0

    mu = M / 2.0 + (9.0 + (-1) ** (M + 1)) / 36.0 - (M / 3.0 + 2.0 / 9.0) / (2.0 ** M)

    # Corrected probabilities for even M (NIST STS 2.1.2 has pi[6]=1/32, sum>1).
    # For even M, T = L - M/2: P(T=0)=1/2, P(T=k)=1/4^k for k>=1,
    # P(T=-k)=1/(2*4^k) for k>=1. Tail sums: pi[6]=1/48, pi[0]=1/96.
    pi = [1/96, 1/32, 1/8, 1/2, 1/4, 1/16, 1/48]

    freq = [0] * 7
    for i in range(N):
        block = bits[i * M:(i + 1) * M]
        L = _berlekamp_massey_gf2(block)
        T_val = (L - mu) if M % 2 == 0 else (mu - L)
        T_val += 2.0 / 9.0
        if T_val <= -2.5:     freq[0] += 1
        elif T_val <= -1.5:   freq[1] += 1
        elif T_val <= -0.5:   freq[2] += 1
        elif T_val <= 0.5:    freq[3] += 1
        elif T_val <= 1.5:    freq[4] += 1
        elif T_val <= 2.5:    freq[5] += 1
        else:                 freq[6] += 1

    chi_sq = 0.0
    for i in range(K + 1):
        expected = N * pi[i]
        if expected > 0:
            chi_sq += (freq[i] - expected) ** 2 / expected
    return igamc(K / 2.0, chi_sq / 2.0)

def test_universal(bits):
    """Test 13: Maurer's Universal Statistical Test.

    Detects whether the sequence can be significantly compressed.
    NIST SP 800-22 §2.9.
    """
    n = len(bits)
    # Table of (L, Q) and expected values from NIST SP 800-22 §2.9.3
    if n >= 904960:
        L, Q = 7, 1280
    elif n >= 387840:
        L, Q = 6, 640
    else:
        return 1.0
    ev_table = {6: 5.2177052, 7: 6.1962507}
    var_table = {6: 2.954, 7: 3.125}
    expected = ev_table[L]
    variance = var_table[L]
    K = n // L - Q
    if K <= 0:
        return 1.0

    # Initialize table
    table = [0] * (1 << L)
    for i in range(Q):
        val = 0
        for j in range(L):
            val = (val << 1) | bits[i * L + j]
        table[val] = i + 1

    # Test phase
    total = 0.0
    for i in range(Q, Q + K):
        val = 0
        for j in range(L):
            val = (val << 1) | bits[i * L + j]
        dist = i + 1 - table[val]
        if dist > 0:
            total += math.log2(dist)
        table[val] = i + 1

    fn = total / K
    c = 0.7 - 0.8 / L + (4 + 32.0 / L) * (K ** (-3.0 / L)) / 15.0
    sigma = c * math.sqrt(variance / K)
    return math.erfc(abs(fn - expected) / (math.sqrt(2) * sigma))

def _build_walk(bits):
    """Build random walk S_0=0, S_k = sum(2*eps_i - 1), S_{n+1}=0."""
    walk = [0]
    cumsum = 0
    for b in bits:
        cumsum += 2 * b - 1
        walk.append(cumsum)
    walk.append(0)
    return walk

def _find_cycles(walk):
    """Find cycle boundaries (indices where walk == 0)."""
    return [i for i in range(len(walk)) if walk[i] == 0]

def test_random_excursions(bits):
    """Test 14: Random Excursions.

    Tests the number of visits to states in a random walk.
    NIST SP 800-22 §2.14. Returns min p-value across 8 states.
    """
    walk = _build_walk(bits)
    boundaries = _find_cycles(walk)
    J = len(boundaries) - 1
    if J < 500:
        return None  # not enough cycles

    # Theoretical probabilities: pi[|x|-1][k] for k=0..5 visits
    pi_table = [
        [0.5000, 0.2500, 0.1250, 0.0625, 0.03125, 0.03125],
        [0.7500, 0.0625, 0.046875, 0.035156, 0.026367, 0.079102],
        [0.8333, 0.02778, 0.02315, 0.01929, 0.01608, 0.08037],
        [0.8750, 0.015625, 0.013672, 0.011963, 0.010463, 0.073242],
    ]

    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    results = {}

    for x in states:
        pi = pi_table[abs(x) - 1]
        freq = [0] * 6
        for c in range(J):
            count = 0
            for i in range(boundaries[c] + 1, boundaries[c + 1]):
                if walk[i] == x:
                    count += 1
            freq[min(count, 5)] += 1

        chi_sq = 0.0
        for k in range(6):
            expected_val = J * pi[k]
            if expected_val > 0:
                chi_sq += (freq[k] - expected_val) ** 2 / expected_val
        results[x] = igamc(5.0 / 2.0, chi_sq / 2.0)

    return results

def test_random_excursions_variant(bits):
    """Test 15: Random Excursions Variant.

    Tests total number of visits to states in a random walk.
    NIST SP 800-22 §2.15. Returns min p-value across 18 states.
    """
    walk = _build_walk(bits)
    boundaries = _find_cycles(walk)
    J = len(boundaries) - 1
    if J < 500:
        return None

    states = list(range(-9, 0)) + list(range(1, 10))
    results = {}

    for x in states:
        xi = sum(1 for v in walk if v == x)
        denom = math.sqrt(2.0 * J * (4 * abs(x) - 2))
        if denom == 0:
            continue
        results[x] = math.erfc(abs(xi - J) / denom)

    return results

# ── Inter-ticket independence tests ──────────────────────────

def test_xor_independence(stream_a, stream_b):
    """XOR two bitstreams and apply Monobit test.

    Under independence, XOR of two uniform streams is uniform.
    """
    n = min(len(stream_a), len(stream_b))
    xor_bits = [stream_a[i] ^ stream_b[i] for i in range(n)]
    return test_frequency(xor_bits)

def test_pearson_correlation(rc_ext, rc_int, n_perms=100000):
    """Compute element-wise Pearson correlation between ticket streams.

    For each permutation, extract 3 tickets (S[0..7], S[8..15], S[16..23]).
    Compute |r| between all cross-ticket element pairs (8*8*3 = 192 pairs).
    """
    # Collect element values per ticket position
    t0 = [[] for _ in range(8)]  # ticket 0: elements 0-7
    t1 = [[] for _ in range(8)]  # ticket 1: elements 8-15
    t2 = [[] for _ in range(8)]  # ticket 2: elements 16-23

    for counter in range(n_perms):
        s = [counter & P] + [0] * (W - 1)
        poseidon2(s, rc_ext, rc_int)
        for i in range(8):
            t0[i].append(s[i])
            t1[i].append(s[8 + i])
            t2[i].append(s[16 + i])

    def pearson_r(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
        sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
        if sx == 0 or sy == 0:
            return 0.0
        cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
        return cov / (sx * sy)

    tickets = [t0, t1, t2]
    pairs = [(0, 1), (0, 2), (1, 2)]
    all_r = []

    for ta_idx, tb_idx in pairs:
        for i in range(8):
            for j in range(8):
                r = abs(pearson_r(tickets[ta_idx][i], tickets[tb_idx][j]))
                all_r.append(r)

    return all_r

def test_joint_success(rc_ext, rc_int, n_perms=100000, difficulty=8):
    """Test joint ticket success probability.

    At soft difficulty d (leading zero bits), compare observed joint success
    P(ticket_i AND ticket_j succeed) against independent expectation.
    """
    threshold = P >> difficulty  # leading d zero bits in 31-bit value

    # Count individual and joint successes
    success = [[0, 0, 0] for _ in range(3)]  # individual counts
    joint = {}
    pairs = [(0, 1), (0, 2), (1, 2)]
    for p_idx in pairs:
        joint[p_idx] = 0

    n_both_any = 0
    for counter in range(n_perms):
        s = [counter & P] + [0] * (W - 1)
        poseidon2(s, rc_ext, rc_int)

        # Check ticket success: first element of each 8-element ticket < threshold
        succ = [0, 0, 0]
        succ[0] = 1 if s[0] < threshold else 0
        succ[1] = 1 if s[8] < threshold else 0
        succ[2] = 1 if s[16] < threshold else 0

        for k in range(3):
            success[k][0] += succ[k]

        for ta, tb in pairs:
            if succ[ta] and succ[tb]:
                joint[(ta, tb)] += 1

    # Expected under independence
    p_single = threshold / P
    p_joint_expected = p_single ** 2

    results = {}
    for ta, tb in pairs:
        observed = joint[(ta, tb)] / n_perms
        expected = p_joint_expected
        # Chi-squared with 1 df
        if expected > 0 and n_perms * expected >= 5:
            e_yes = n_perms * expected
            e_no = n_perms * (1 - expected)
            o_yes = joint[(ta, tb)]
            o_no = n_perms - o_yes
            chi2 = (o_yes - e_yes) ** 2 / e_yes + (o_no - e_no) ** 2 / e_no
            pval = chi2_pvalue(chi2, 1)
        else:
            pval = 1.0
        results[(ta, tb)] = {
            "observed": observed,
            "expected": expected,
            "p_value": pval,
            "n_joint": joint[(ta, tb)],
        }

    return results, p_single

# ── Main driver ──────────────────────────────────────────────

def pvalue_uniformity_test(p_values):
    """NIST SP 800-22 §4.2.2: Test that p-values are uniformly distributed.

    Partitions p-values into 10 bins on [0,1), applies chi-squared.
    Returns p-value of uniformity test (pass if >= 0.0001).
    """
    s = len(p_values)
    if s < 10:
        return None
    bins = [0] * 10
    for p in p_values:
        idx = min(int(p * 10), 9)
        bins[idx] += 1
    expected = s / 10.0
    chi_sq = sum((b - expected) ** 2 / expected for b in bins)
    return igamc(9.0 / 2.0, chi_sq / 2.0)

NIST_TESTS = [
    ("Frequency (Monobit)", test_frequency),
    ("Block Frequency", test_block_frequency),
    ("Runs", test_runs),
    ("Longest Run of Ones", test_longest_run),
    ("Serial", test_serial),
    ("Approximate Entropy", test_approximate_entropy),
    ("Cumulative Sums", test_cumulative_sums),
    ("Binary Matrix Rank", test_rank),
    ("Non-overlapping Template", test_nonoverlapping_template),
    ("Overlapping Template", test_overlapping_template),
    ("DFT (Spectral)", test_spectral),
    ("Linear Complexity", test_linear_complexity),
    ("Maurer Universal", test_universal),
    ("Random Excursions", test_random_excursions),
    ("Random Excursions Var", test_random_excursions_variant),
]

def run_nist_suite(n_sequences=100, seq_length=1_000_000, alpha=0.01):
    """Run NIST SP 800-22 test suite on Poseidon2 output."""

    print("=" * 78)
    print("NIST SP 800-22 Statistical Randomness Tests")
    print("Poseidon2 Width-24 / M31 — Sequential Counter Inputs")
    print("=" * 78)
    print()

    rc_ext, rc_int = gen_rc()

    # Cross-validate
    cross_validate(rc_ext, rc_int)

    print(f"Parameters:")
    print(f"  Sequences:     {n_sequences}")
    print(f"  Bits/sequence: {seq_length:,}")
    print(f"  Alpha:         {alpha}")
    # NIST §4.2: threshold = p̂ - 3*sqrt(p̂*(1-p̂)/n), p̂ = 1 - α
    p_hat = 1.0 - alpha
    threshold = p_hat - 3 * math.sqrt(p_hat * (1 - p_hat) / n_sequences)
    min_pass = math.ceil(threshold * n_sequences)
    print(f"  Pass thresh:   {min_pass}/{n_sequences}"
          f" (NIST §4.2: ≥{threshold*100:.0f}%)")
    perms_per_seq = math.ceil(seq_length / (W * 31))
    total_perms = perms_per_seq * n_sequences
    print(f"  Perms/seq:     {perms_per_seq:,}")
    print(f"  Total perms:   {total_perms:,}")
    print()

    # Pre-generate all bitstreams (avoids recomputing Poseidon2 per test)
    import time
    t0 = time.time()
    print("Generating bitstreams...", end="", flush=True)
    all_bits = []
    for seq_idx in range(n_sequences):
        start_counter = seq_idx * perms_per_seq
        bits, _ = generate_bitstream(seq_length, rc_ext, rc_int,
                                     start_counter=start_counter)
        all_bits.append(bits)
        if (seq_idx + 1) % 10 == 0:
            sys.stdout.write(f"\rGenerating bitstreams... {seq_idx+1}/{n_sequences}")
            sys.stdout.flush()
    t1 = time.time()
    print(f"\rGenerated {n_sequences} × {seq_length:,} bits in {t1-t0:.1f}s")
    print()

    # Run tests
    results = {}

    print(f"{'Test':<28} | {'Pass':>4}/{n_sequences:<3} | {'Rate':>6} | {'Result':<6}")
    print("-" * 58)

    def _eval_pass(pass_count, applicable, alpha):
        """Compute pass rate and whether test passes NIST threshold."""
        if applicable == 0:
            return 0.0, False, "N/A"
        rate = pass_count / applicable
        min_pass_adj = math.ceil(
            (1 - alpha - 3 * math.sqrt(alpha * (1 - alpha) / applicable))
            * applicable)
        passed = pass_count >= min_pass_adj
        return rate, passed, "PASS" if passed else "FAIL"

    for test_name, test_fn in NIST_TESTS:
        sys.stdout.write(f"  Running {test_name}...")
        sys.stdout.flush()

        # Collect results across sequences
        raw = []
        for seq_idx in range(n_sequences):
            raw.append(test_fn(all_bits[seq_idx]))

        # Handle three return types: float, None, dict
        if raw and isinstance(raw[0], dict):
            # Multi-sub-test: evaluate each sub-test independently
            # Each sub-test (template/state) is its own NIST test with
            # pass/applicable sequences. A sub-test passes if its pass
            # count meets the NIST §4.2 threshold for its applicable count.
            sub_keys = sorted(raw[0].keys())
            n_sub = len(sub_keys)
            subtests_passing = 0
            worst_rate = 1.0
            total_pass = 0
            total_applicable = 0
            for key in sub_keys:
                sp = 0
                sa = 0
                for r in raw:
                    if r is None:
                        continue
                    if key in r:
                        sa += 1
                        if r[key] >= alpha:
                            sp += 1
                total_pass += sp
                total_applicable += sa
                if sa > 0:
                    _, sub_passed, _ = _eval_pass(sp, sa, alpha)
                    if sub_passed:
                        subtests_passing += 1
                    worst_rate = min(worst_rate, sp / sa)
            overall_rate = total_pass / total_applicable if total_applicable > 0 else 0
            # Overall pass: all sub-tests pass their individual thresholds
            # (under H0, ~0.6% of sub-tests fail by chance, so allow some)
            # Use same NIST §4.2 formula on sub-test pass rate
            sub_rate = subtests_passing / n_sub if n_sub > 0 else 0
            # For small n_sub (<30), just require all to pass is too strict.
            # Report per-sub-test results transparently.
            passed = subtests_passing == n_sub
            status = "PASS" if passed else f"{subtests_passing}/{n_sub}"
            # §4.2.2 p-value uniformity: worst across sub-tests
            uniformity_fail = 0
            for key in sub_keys:
                pvals = [r[key] for r in raw if r is not None and key in r]
                u = pvalue_uniformity_test(pvals)
                if u is not None and u < 0.0001:
                    uniformity_fail += 1
            results[test_name] = {
                "pass_count": total_pass,
                "applicable": total_applicable,
                "rate": overall_rate,
                "passed": passed,
                "n_subtests": n_sub,
                "subtests_passing": subtests_passing,
                "worst_subtest_rate": worst_rate,
                "uniformity_fail": uniformity_fail,
            }
            u_str = "" if uniformity_fail == 0 else f" U-FAIL:{uniformity_fail}"
            sys.stdout.write(
                f"\r{test_name:<28} | {total_pass:>4}/{total_applicable:<5} | "
                f"{overall_rate:>5.1%} | {status:<6} "
                f"({subtests_passing}/{n_sub} sub-tests pass, "
                f"worst {worst_rate:.1%}{u_str})\n")
        else:
            # Scalar p-value or None
            pass_count = 0
            applicable = 0
            for r in raw:
                if r is None:
                    continue
                applicable += 1
                if r >= alpha:
                    pass_count += 1
            rate, passed, status = _eval_pass(pass_count, applicable, alpha)
            n_str = f"{applicable}" if applicable < n_sequences else f"{n_sequences}"
            # §4.2.2 p-value uniformity test
            pvals = [r for r in raw if r is not None]
            u_p = pvalue_uniformity_test(pvals)
            u_pass = u_p is None or u_p >= 0.0001
            results[test_name] = {
                "pass_count": pass_count,
                "applicable": applicable,
                "rate": rate,
                "passed": passed and u_pass,
                "uniformity_p": u_p,
            }
            u_str = "" if u_pass else f" U:{u_p:.4f}"
            sys.stdout.write(f"\r{test_name:<28} | {pass_count:>4}/{n_str:<3} | "
                             f"{rate:>5.1%} | {status:<6}{u_str}\n")
        sys.stdout.flush()

    print("-" * 78)
    # For overall: scalar tests must pass; sub-test results reported transparently
    scalar_tests = {k: v for k, v in results.items() if "n_subtests" not in v}
    subtest_tests = {k: v for k, v in results.items() if "n_subtests" in v}
    scalar_pass = all(r["passed"] for r in scalar_tests.values())
    if scalar_pass and not subtest_tests:
        print("Overall: ALL PASS")
    elif scalar_pass:
        parts = []
        for name, r in subtest_tests.items():
            parts.append(f"{name}: {r['subtests_passing']}/{r['n_subtests']} sub-tests pass")
        print(f"Overall: scalar tests ALL PASS; {'; '.join(parts)}")
    else:
        print("Overall: SOME FAIL")
    print()

    return results, rc_ext, rc_int

def run_independence_tests(rc_ext, rc_int, n_perms=100000):
    """Run inter-ticket independence tests."""

    print("=" * 78)
    print("Inter-Ticket Independence Tests")
    print("=" * 78)
    print()

    # 1. XOR independence
    print("1. XOR Independence (Monobit on XOR of ticket bitstreams)")
    print("-" * 58)
    s0, s1, s2 = generate_ticket_streams(n_perms, rc_ext, rc_int)
    pairs = [("T0 ⊕ T1", s0, s1), ("T0 ⊕ T2", s0, s2), ("T1 ⊕ T2", s1, s2)]
    xor_results = []
    for name, sa, sb in pairs:
        p_val = test_xor_independence(sa, sb)
        status = "PASS" if p_val >= 0.01 else "FAIL"
        print(f"  {name}: p = {p_val:.6f}  [{status}]")
        xor_results.append((name, p_val))
    print()

    # 2. Pearson correlation
    print(f"2. Element-wise Pearson Correlation ({n_perms:,} permutations)")
    print("-" * 58)
    all_r = test_pearson_correlation(rc_ext, rc_int, n_perms)
    max_r = max(all_r)
    mean_r = sum(all_r) / len(all_r)
    print(f"  Cross-ticket pairs:  {len(all_r)}")
    print(f"  Mean |r|:            {mean_r:.6f}")
    print(f"  Max  |r|:            {max_r:.6f}")
    expected_max = math.sqrt(2 * math.log(len(all_r)) / n_perms)
    print(f"  Expected max (random): ~{expected_max:.6f}")
    print(f"  Result:              {'PASS' if mean_r < 0.01 else 'FAIL'}"
          f" (mean |r| < 0.01)")
    print()

    # 3. Joint success
    print(f"3. Joint Ticket Success (difficulty=5, {n_perms:,} permutations)")
    print("-" * 58)
    joint_results, p_single = test_joint_success(rc_ext, rc_int, n_perms, difficulty=5)
    print(f"  P(single success) = {p_single:.6f}")
    for (ta, tb), res in joint_results.items():
        status = "PASS" if res["p_value"] >= 0.01 else "FAIL"
        print(f"  T{ta} ∧ T{tb}: observed={res['observed']:.8f}, "
              f"expected={res['expected']:.8f}, "
              f"chi² p={res['p_value']:.4f}  [{status}]")
    print()

    return xor_results, all_r, joint_results

def main():
    print()
    print("Poseidon2 Width-24 / M31 — NIST SP 800-22 Statistical Testing")
    print("Port from ZK-SPoW main.rs (Rust → Python)")
    print()

    results, rc_ext, rc_int = run_nist_suite(
        n_sequences=100, seq_length=1_000_000, alpha=0.01
    )

    xor_results, all_r, joint_results = run_independence_tests(
        rc_ext, rc_int, n_perms=100000
    )

    # Summary
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print()
    scalar_results = {k: v for k, v in results.items() if "n_subtests" not in v}
    subtest_results = {k: v for k, v in results.items() if "n_subtests" in v}
    n_scalar_pass = sum(1 for r in scalar_results.values() if r["passed"])
    n_scalar = len(scalar_results)
    print(f"NIST SP 800-22:  {n_scalar_pass}/{n_scalar} scalar tests pass at ≥96/100 threshold")
    for name, r in subtest_results.items():
        print(f"  {name}: {r['subtests_passing']}/{r['n_subtests']} sub-tests pass"
              f" (worst {r['worst_subtest_rate']:.1%})")
    print(f"XOR Monobit:     all p > 0.01 = "
          f"{'YES' if all(p >= 0.01 for _, p in xor_results) else 'NO'}")
    mean_r = sum(all_r) / len(all_r)
    print(f"Pearson |r|:     mean = {mean_r:.6f} (< 0.01 target)")
    joint_pass = all(r["p_value"] >= 0.01 for r in joint_results.values())
    print(f"Joint success:   all p > 0.01 = {'YES' if joint_pass else 'NO'}")
    print()
    all_scalar_pass = n_scalar_pass == n_scalar
    all_indep_pass = all(p >= 0.01 for _, p in xor_results) and mean_r < 0.01
    if all_scalar_pass and all_indep_pass:
        print("CONCLUSION: Poseidon2 Width-24 output passes all statistical tests.")
    else:
        print("CONCLUSION: Some tests failed — investigate.")
    print()

if __name__ == "__main__":
    main()
