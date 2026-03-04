#!/usr/bin/env python3
"""Compute P-value_T (§4.2.2 uniformity) for all NIST tests."""
import sys, os, math, time, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nist_sp800_22_poseidon2 import (
    W, gen_rc, generate_bitstream, pvalue_uniformity_test, NIST_TESTS,
)

def main():
    rc_ext, rc_int = gen_rc()
    n_sequences = 100
    seq_length = 1_000_000
    alpha = 0.01
    perms_per_seq = math.ceil(seq_length / (W * 31))

    print("Generating bitstreams...", flush=True)
    t0 = time.time()
    all_bits = []
    for i in range(n_sequences):
        bits, _ = generate_bitstream(seq_length, rc_ext, rc_int,
                                     start_counter=i * perms_per_seq)
        all_bits.append(bits)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_sequences}", flush=True)
    print(f"Generated in {time.time()-t0:.1f}s", flush=True)

    results = {}
    for test_name, test_fn in NIST_TESTS:
        t0 = time.time()
        print(f"Running {test_name}...", end="", flush=True)
        raw = [test_fn(all_bits[i]) for i in range(n_sequences)]
        elapsed = time.time() - t0

        if raw and isinstance(raw[0], dict):
            # Dict test: compute per-subtest uniformity, report worst
            sub_keys = sorted(raw[0].keys())
            worst_u = 1.0
            n_fail = 0
            for key in sub_keys:
                pvals = [r[key] for r in raw if r is not None and key in r]
                u = pvalue_uniformity_test(pvals)
                if u is not None:
                    worst_u = min(worst_u, u)
                    if u < 0.0001:
                        n_fail += 1
            # Also compute pass counts per subtest
            pass_counts = {}
            for key in sub_keys:
                pc = sum(1 for r in raw if r is not None and key in r and r[key] >= alpha)
                ap = sum(1 for r in raw if r is not None and key in r)
                pass_counts[str(key)] = {"pass": pc, "applicable": ap}
            results[test_name] = {
                "type": "dict", "worst_uniformity_p": worst_u,
                "uniformity_fail": n_fail, "n_subtests": len(sub_keys),
                "pass_counts": pass_counts, "elapsed": elapsed,
            }
            print(f" worst_u={worst_u:.4f}, fail={n_fail}/{len(sub_keys)}, {elapsed:.1f}s", flush=True)
        else:
            # Scalar test
            pvals = [r for r in raw if r is not None]
            applicable = len(pvals)
            pass_count = sum(1 for p in pvals if p >= alpha)
            u_p = pvalue_uniformity_test(pvals)
            results[test_name] = {
                "type": "scalar", "uniformity_p": u_p,
                "pass_count": pass_count, "applicable": applicable,
                "elapsed": elapsed,
            }
            print(f" {pass_count}/{applicable}, u_p={u_p:.4f}, {elapsed:.1f}s", flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "uniformity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out_path}", flush=True)

if __name__ == "__main__":
    main()
