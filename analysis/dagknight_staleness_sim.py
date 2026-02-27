#!/usr/bin/env python3
"""
DAGKnight anticone simulation under varying header staleness.

Measures the marginal impact of ZK-SPoW header staleness on DAG
anticone sizes relative to network propagation delay.

No external dependencies (stdlib only).
"""

import math
import random
from bisect import bisect_left, bisect_right
from statistics import mean, median, stdev


def lognormal_params(target_mean, target_std):
    """Convert desired mean/std to log-normal mu/sigma."""
    variance = target_std ** 2
    mu = math.log(target_mean ** 2 / math.sqrt(variance + target_mean ** 2))
    sigma = math.sqrt(math.log(1 + variance / target_mean ** 2))
    return mu, sigma


def simulate(block_rate, duration, delay_mean, delay_std, staleness,
             n_runs=10, seed=42):
    """
    Simulate Poisson block arrivals with network delay + header staleness.

    Model: block B is in the anticone of block A if neither was visible
    to the other's miner at creation time.  Simplified as:
        |t_A - t_B| < (delay_A + delay_B) / 2 + staleness

    Using per-block log-normal propagation delay to model heterogeneous
    network conditions.
    """
    rng = random.Random(seed)
    mu, sigma = lognormal_params(delay_mean, delay_std)

    all_anticones = []
    total_blocks = 0

    for run in range(n_runs):
        # Generate Poisson arrivals
        times = []
        t = 0.0
        while t < duration:
            dt = rng.expovariate(block_rate)
            t += dt
            if t < duration:
                times.append(t)

        n = len(times)
        total_blocks += n

        # Per-block propagation delay (log-normal)
        delays = [rng.lognormvariate(mu, sigma) for _ in range(n)]

        # Effective half-window for each block
        # Two blocks are concurrent if |t_i - t_j| < (d_i + d_j)/2 + staleness
        # Approximation for efficiency: use per-block window = delay_i + staleness
        windows = [d + staleness for d in delays]

        # Compute anticone sizes via sorted scan
        for i in range(n):
            w = windows[i]
            lo = bisect_left(times, times[i] - w)
            hi = bisect_right(times, times[i] + w)
            anticone = hi - lo - 1  # exclude self
            all_anticones.append(anticone)

    all_anticones.sort()
    n_total = len(all_anticones)
    p95_idx = int(n_total * 0.95)
    p99_idx = int(n_total * 0.99)

    return {
        "mean": mean(all_anticones),
        "median": median(all_anticones),
        "std": stdev(all_anticones) if n_total > 1 else 0.0,
        "p95": all_anticones[p95_idx] if p95_idx < n_total else all_anticones[-1],
        "p99": all_anticones[p99_idx] if p99_idx < n_total else all_anticones[-1],
        "max": all_anticones[-1],
        "n_blocks": n_total,
    }


def main():
    block_rate = 100   # 100 BPS
    duration = 60      # seconds per run
    n_runs = 10        # runs per scenario

    delay_scenarios = [
        ("LAN  (5ms)",    0.005, 0.002),
        ("Regional(20ms)",0.020, 0.010),
        ("Global-L(50ms)",0.050, 0.025),
        ("Global-M(100ms)",0.100,0.050),
        ("Global-H(200ms)",0.200,0.100),
    ]

    staleness_scenarios = [
        ("None",      0.0),
        ("ASIC 0.05ms", 0.00005),
        ("GPU  3.4ms",  0.0034),
    ]

    print(f"DAGKnight Anticone Simulation")
    print(f"Block rate: {block_rate} BPS | Duration: {duration}s x {n_runs} runs")
    print(f"{'=' * 100}")
    hdr = (f"{'Network Delay':<18} | {'Staleness':<14} | "
           f"{'Mean':>7} | {'Median':>6} | {'P95':>5} | "
           f"{'P99':>5} | {'Max':>5} | {'delta Mean':>10}")
    print(hdr)
    print(f"{'-' * 100}")

    for dname, dmean, dstd in delay_scenarios:
        baseline_mean = None
        for sname, stale in staleness_scenarios:
            r = simulate(block_rate, duration, dmean, dstd, stale,
                         n_runs=n_runs)
            if baseline_mean is None:
                baseline_mean = r["mean"]
                delta = ""
            else:
                pct = (r["mean"] - baseline_mean) / baseline_mean * 100
                delta = f"+{pct:.2f}%"
            print(f"{dname:<18} | {sname:<14} | "
                  f"{r['mean']:7.2f} | {r['median']:6.1f} | "
                  f"{r['p95']:5.0f} | {r['p99']:5.0f} | "
                  f"{r['max']:5.0f} | {delta:>10}")
        print(f"{'-' * 100}")

    # Analytical comparison
    print()
    print("Analytical expected anticone: E[anticone] ≈ 2 * λ * (delay + staleness)")
    print(f"{'Network Delay':<18} | {'No stale':>10} | {'ASIC':>10} | {'GPU':>10} | {'GPU delta':>10}")
    print(f"{'-' * 70}")
    for dname, dmean, _ in delay_scenarios:
        a0 = 2 * block_rate * dmean
        a_asic = 2 * block_rate * (dmean + 0.00005)
        a_gpu = 2 * block_rate * (dmean + 0.0034)
        delta = (a_gpu - a0) / a0 * 100
        print(f"{dname:<18} | {a0:10.2f} | {a_asic:10.2f} | {a_gpu:10.2f} | +{delta:8.2f}%")


if __name__ == "__main__":
    main()
