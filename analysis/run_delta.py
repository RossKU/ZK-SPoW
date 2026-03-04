#!/usr/bin/env python3
"""Run only the changed NIST SP 800-22 tests (delta from compliance fixes).

Usage: python3 run_delta.py <group>
  group 1: DFT (Spectral) — Bluestein exact n-point
  group 2: Non-overlapping Template — 148 aperiodic templates
  group 3: Serial + CuSum + Random Excursions + Random Excursions Variant
"""
import sys, os, math, time, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nist_sp800_22_poseidon2 import (
    W, P, gen_rc, generate_bitstream, pvalue_uniformity_test,
    test_spectral, test_nonoverlapping_template,
    test_serial, test_cumulative_sums,
    test_random_excursions, test_random_excursions_variant,
)

def _eval_pass(pass_count, applicable, alpha=0.01):
    if applicable == 0:
        return 0.0, False
    rate = pass_count / applicable
    min_pass = math.ceil(
        (1 - alpha - 3 * math.sqrt(alpha * (1 - alpha) / applicable))
        * applicable)
    return rate, pass_count >= min_pass

def generate_all_bitstreams(n_sequences=100, seq_length=1_000_000):
    rc_ext, rc_int = gen_rc()
    perms_per_seq = math.ceil(seq_length / (W * 31))
    t0 = time.time()
    all_bits = []
    for i in range(n_sequences):
        bits, _ = generate_bitstream(seq_length, rc_ext, rc_int,
                                     start_counter=i * perms_per_seq)
        all_bits.append(bits)
        if (i + 1) % 10 == 0:
            print(f"  bitstreams {i+1}/{n_sequences}", flush=True)
    print(f"  generated in {time.time()-t0:.1f}s", flush=True)
    return all_bits

def run_test(name, test_fn, all_bits, alpha=0.01):
    n = len(all_bits)
    t0 = time.time()
    raw = [test_fn(all_bits[i]) for i in range(n)]
    elapsed = time.time() - t0

    if raw and isinstance(raw[0], dict):
        sub_keys = sorted(raw[0].keys())
        n_sub = len(sub_keys)
        subtests_passing = 0
        worst_rate = 1.0
        total_pass = 0
        total_applicable = 0
        sub_details = {}
        for key in sub_keys:
            sp, sa = 0, 0
            for r in raw:
                if r is None:
                    continue
                if key in r:
                    sa += 1
                    if r[key] >= alpha:
                        sp += 1
            total_pass += sp
            total_applicable += sa
            rate, passed = _eval_pass(sp, sa, alpha)
            if passed:
                subtests_passing += 1
            if sa > 0:
                worst_rate = min(worst_rate, sp / sa)
            sub_details[str(key)] = {"pass": sp, "applicable": sa, "rate": rate, "passed": passed}
        # Uniformity
        u_fail = 0
        for key in sub_keys:
            pvals = [r[key] for r in raw if r is not None and key in r]
            u = pvalue_uniformity_test(pvals)
            if u is not None and u < 0.0001:
                u_fail += 1
        result = {
            "test": name, "type": "dict", "elapsed": elapsed,
            "n_subtests": n_sub, "subtests_passing": subtests_passing,
            "total_pass": total_pass, "total_applicable": total_applicable,
            "worst_rate": worst_rate, "uniformity_fail": u_fail,
            "sub_details": sub_details,
        }
    else:
        pc, ap = 0, 0
        for r in raw:
            if r is None:
                continue
            ap += 1
            if r >= alpha:
                pc += 1
        rate, passed = _eval_pass(pc, ap, alpha)
        pvals = [r for r in raw if r is not None]
        u_p = pvalue_uniformity_test(pvals)
        u_pass = u_p is None or u_p >= 0.0001
        result = {
            "test": name, "type": "scalar", "elapsed": elapsed,
            "pass_count": pc, "applicable": ap,
            "rate": rate, "passed": passed and u_pass,
            "uniformity_p": u_p,
        }

    return result

def main():
    group = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if group == 1:
        tests = [("DFT (Spectral)", test_spectral)]
    elif group == 2:
        tests = [("Non-overlapping Template", test_nonoverlapping_template)]
    elif group == 3:
        tests = [
            ("Serial", test_serial),
            ("Cumulative Sums", test_cumulative_sums),
            ("Random Excursions", test_random_excursions),
            ("Random Excursions Var", test_random_excursions_variant),
        ]
    else:
        print("Usage: python3 run_delta.py <1|2|3>")
        sys.exit(1)

    print(f"=== Group {group}: {', '.join(t[0] for t in tests)} ===", flush=True)
    all_bits = generate_all_bitstreams()

    results = []
    for name, fn in tests:
        print(f"  Running {name}...", flush=True)
        r = run_test(name, fn, all_bits)
        results.append(r)
        # Print summary
        if r["type"] == "dict":
            print(f"  {name}: {r['subtests_passing']}/{r['n_subtests']} sub-tests pass, "
                  f"worst {r['worst_rate']:.1%}, {r['elapsed']:.1f}s", flush=True)
        else:
            print(f"  {name}: {r['pass_count']}/{r['applicable']} "
                  f"({r['rate']:.1%}), passed={r['passed']}, {r['elapsed']:.1f}s", flush=True)

    # Write JSON results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"delta_results_g{group}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}", flush=True)

if __name__ == "__main__":
    main()
