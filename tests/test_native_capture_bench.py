from fractions import Fraction

import pytest

from bitmyner.native_capture_bench import capture_benchmark_metrics


def test_capture_metrics_exact_stable_ratio():
    r = capture_benchmark_metrics(
        count=1000,
        baseline_latencies_ns=(100, 100, 100),
        capture_latencies_ns=(110, 110, 110),
        stale_budget_s=1.0,
    )
    assert r['baseline_median_hps_num'] == 10_000_000_000
    assert r['baseline_median_hps_den'] == 1
    assert r['capture_median_hps_num'] == 100_000_000_000
    assert r['capture_median_hps_den'] == 11
    assert Fraction(r['raw_throughput_ratio_num'], r['raw_throughput_ratio_den']) == Fraction(10, 11)
    assert Fraction(r['tau_adjusted_throughput_ratio_num'], r['tau_adjusted_throughput_ratio_den']) == Fraction(10, 11)


def test_capture_metrics_include_exact_tau_ratio():
    r = capture_benchmark_metrics(
        count=1000,
        baseline_latencies_ns=(100, 100, 200),
        capture_latencies_ns=(120, 120, 120),
        stale_budget_s=1.0,
    )
    raw = Fraction(r['raw_throughput_ratio_num'], r['raw_throughput_ratio_den'])
    tau = Fraction(r['tau_ratio_num'], r['tau_ratio_den'])
    adjusted = Fraction(r['tau_adjusted_throughput_ratio_num'], r['tau_adjusted_throughput_ratio_den'])
    assert adjusted == raw * tau
    assert tau > 1


def test_capture_metrics_fail_closed_on_bad_measurements():
    with pytest.raises(ValueError):
        capture_benchmark_metrics(count=0, baseline_latencies_ns=(1,), capture_latencies_ns=(1,), stale_budget_s=1.0)
    with pytest.raises(ValueError):
        capture_benchmark_metrics(count=1, baseline_latencies_ns=(), capture_latencies_ns=(1,), stale_budget_s=1.0)
    with pytest.raises(ValueError):
        capture_benchmark_metrics(count=1, baseline_latencies_ns=(1,), capture_latencies_ns=(0,), stale_budget_s=1.0)
