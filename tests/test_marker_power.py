import pytest

from bitmyner.marker_power import (
    expected_threshold_events,
    samples_for_expected_events,
    plan_threshold_power,
)


def test_expected_threshold_events_matches_sha256_baseline():
    assert expected_threshold_events(2048, 8) == pytest.approx(8.0)
    assert expected_threshold_events(12800, 4) == pytest.approx(800.0)


def test_samples_for_expected_events_is_exact_integer_planner():
    assert samples_for_expected_events(8, 100) == 25_600
    assert samples_for_expected_events(8, 200) == 51_200
    assert samples_for_expected_events(12, 100) == 409_600


def test_plan_threshold_power_reports_bottleneck_sample_count():
    p = plan_threshold_power((4, 6, 8), desired_events=200)
    assert p['per_threshold']['4']['samples'] == 3_200
    assert p['per_threshold']['6']['samples'] == 12_800
    assert p['per_threshold']['8']['samples'] == 51_200
    assert p['minimum_common_samples'] == 51_200


def test_power_planner_fails_closed_on_invalid_inputs():
    with pytest.raises(ValueError):
        expected_threshold_events(0, 8)
    with pytest.raises(ValueError):
        samples_for_expected_events(0, 100)
    with pytest.raises(ValueError):
        samples_for_expected_events(257, 100)
    with pytest.raises(ValueError):
        plan_threshold_power((), desired_events=100)
