from __future__ import annotations

from fractions import Fraction

from .tau_jitter import tau_modulation_from_latencies_ns


def _median_fraction(values):
    vals=tuple(int(v) for v in values)
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("latencies must be positive integer nanoseconds")
    s=sorted(vals); n=len(s); m=n//2
    return Fraction(s[m],1) if n%2 else Fraction(s[m-1]+s[m],2)


def capture_benchmark_metrics(*, count:int, baseline_latencies_ns, capture_latencies_ns, stale_budget_s:float) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    if stale_budget_s <= 0:
        raise ValueError("stale_budget_s must be positive")
    b=tuple(int(x) for x in baseline_latencies_ns)
    c=tuple(int(x) for x in capture_latencies_ns)
    mb=_median_fraction(b); mc=_median_fraction(c)
    tb=tau_modulation_from_latencies_ns(b, stale_budget_s).tau_scalar
    tc=tau_modulation_from_latencies_ns(c, stale_budget_s).tau_scalar
    bhps=Fraction(count*1_000_000_000,1)/mb
    chps=Fraction(count*1_000_000_000,1)/mc
    raw=chps/bhps
    tau_ratio=tc/tb
    adjusted=raw*tau_ratio
    return {
        "schema": "BITMYNER_NATIVE_CAPTURE_BENCH_METRICS_V0_1",
        "count": count,
        "baseline_median_hps_num": bhps.numerator,
        "baseline_median_hps_den": bhps.denominator,
        "capture_median_hps_num": chps.numerator,
        "capture_median_hps_den": chps.denominator,
        "raw_throughput_ratio_num": raw.numerator,
        "raw_throughput_ratio_den": raw.denominator,
        "baseline_tau_num": tb.numerator,
        "baseline_tau_den": tb.denominator,
        "capture_tau_num": tc.numerator,
        "capture_tau_den": tc.denominator,
        "tau_ratio_num": tau_ratio.numerator,
        "tau_ratio_den": tau_ratio.denominator,
        "tau_adjusted_throughput_ratio_num": adjusted.numerator,
        "tau_adjusted_throughput_ratio_den": adjusted.denominator,
    }
