from bitmyner.marker_effect_power_v15 import (
    expected_gamma_lcb_v15,
    plan_effect_power_v15,
)


def test_expected_gamma_lcb_design_is_below_point_estimate():
    r = expected_gamma_lcb_v15(
        262144,
        event_probability=1 / 256,
        marker_rate=0.25,
        priority_work_fraction=0.40,
        target_gamma=1.25,
        alpha=0.05 / 24,
    )
    assert r['implied_lift'] == 2.0
    assert 0.0 < r['expected_gamma_lcb'] < 1.25


def test_effect_power_planner_is_label_free_and_meets_requested_lcb():
    p = plan_effect_power_v15(
        hardest_k=8,
        marker_rate=0.25,
        priority_work_fraction=0.40,
        target_gamma=1.25,
        desired_gamma_lcb=1.0,
        familywise_alpha=0.05,
        candidate_count=8,
        threshold_count=3,
        max_samples=1 << 24,
    )
    assert p['label_access'] is False
    assert p['effect_is_assumed_for_power_design_not_observed'] is True
    assert p['status'] == 'POWER_TARGET_MET'
    assert p['design_at_required_samples']['expected_gamma_lcb'] >= 1.0
    assert p['required_samples'] > 16384


def test_stronger_target_effect_needs_no_more_samples_than_weaker_effect():
    weak = plan_effect_power_v15(
        hardest_k=8,
        marker_rate=0.25,
        priority_work_fraction=0.40,
        target_gamma=1.15,
        max_samples=1 << 26,
    )
    strong = plan_effect_power_v15(
        hardest_k=8,
        marker_rate=0.25,
        priority_work_fraction=0.40,
        target_gamma=1.50,
        max_samples=1 << 26,
    )
    assert strong['required_samples'] <= weak['required_samples']


def test_stricter_multiple_testing_alpha_does_not_reduce_required_samples():
    fewer_tests = plan_effect_power_v15(
        hardest_k=8,
        marker_rate=0.25,
        priority_work_fraction=0.40,
        target_gamma=1.25,
        candidate_count=2,
        threshold_count=3,
        max_samples=1 << 24,
    )
    more_tests = plan_effect_power_v15(
        hardest_k=8,
        marker_rate=0.25,
        priority_work_fraction=0.40,
        target_gamma=1.25,
        candidate_count=16,
        threshold_count=3,
        max_samples=1 << 24,
    )
    assert more_tests['required_samples'] >= fewer_tests['required_samples']
