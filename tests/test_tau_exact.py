from __future__ import annotations

from fractions import Fraction

from bitmyner.tau_exact import exact_tau_score_hps


def test_exact_tau_score_zero_jitter():
    score, evidence = exact_tau_score_hps(
        1_000_000,
        (10_000_000, 10_000_000, 10_000_000),
        0.200,
    )
    assert score == Fraction(100_000_000, 1)
    assert evidence["effective_latency_ns_num"] == 10_000_000
    assert evidence["effective_latency_ns_den"] == 1


def test_exact_tau_score_penalizes_jitter_in_time_domain():
    # median = 100 ms; MAD about median = 100/3 ms.
    score, evidence = exact_tau_score_hps(
        1_000_000,
        (100_000_000, 100_000_000, 200_000_000),
        0.200,
    )
    assert score == Fraction(7_500_000, 1)
    assert evidence["score_hps_num"] == 7_500_000
    assert evidence["score_hps_den"] == 1


def test_exact_tau_score_includes_stale_latency_transform():
    # No jitter, but median latency 300 ms exceeds a 200 ms budget.
    # Raw service rate is 10 MH/s and s_tau_latency=2/3.
    score, _ = exact_tau_score_hps(
        3_000_000,
        (300_000_000, 300_000_000, 300_000_000),
        0.200,
    )
    assert score == Fraction(20_000_000, 3)
