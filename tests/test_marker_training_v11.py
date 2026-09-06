from bitmyner.marker_training_v11 import (
    adaptive_cohort_count_v11,
    run_nested_crossfit_training_v11,
)


def test_v11_adaptive_cohort_count_uses_expected_power_only():
    small = adaptive_cohort_count_v11(
        header_count=40,
        nonces_per_header=32,
        hardest_k=8,
        train_discovery_fraction=0.72,
        split_seed='v11-small',
        header_seed=101,
        requested_max_cohorts=4,
        min_expected_events_per_cohort=8.0,
    )
    large = adaptive_cohort_count_v11(
        header_count=200,
        nonces_per_header=128,
        hardest_k=8,
        train_discovery_fraction=0.72,
        split_seed='v11-large',
        header_seed=101,
        requested_max_cohorts=4,
        min_expected_events_per_cohort=8.0,
    )
    assert 2 <= small['cohort_count'] <= 4
    assert 2 <= large['cohort_count'] <= 4
    assert large['expected_events_total'] > small['expected_events_total']
    assert large['cohort_count'] >= small['cohort_count']
    assert small['label_access'] is False
    assert large['label_access'] is False


def test_v11_nested_crossfit_uses_lcb_cohort_robustness_and_frozen_confirmation():
    r = run_nested_crossfit_training_v11(
        iterations=4,
        seeds=(4040, 4141, 4242),
        header_count=36,
        nonces_per_header=24,
        thresholds=(2, 3),
        population_size=12,
        offspring_per_iteration=2,
        max_predicates=2,
        train_discovery_fraction=0.70,
        requested_max_cohorts=4,
        min_expected_events_per_cohort=2.0,
    )
    assert r['status'] == 'PASS_NESTED_CROSSFIT_V11_COMPLETE'
    assert r['total_iterations'] == 12
    assert len(r['runs']) == 3
    assert r['claims']['cohort_robustness_uses_lower_confidence_bounds'] is True
    assert r['claims']['adaptive_cohort_count_is_label_free'] is True
    assert r['claims']['validation_and_holdout_are_not_training_surfaces'] is True
    for run, policy in zip(r['runs'], r['cohort_policies']):
        assert run['training_policy']['cohort_robust_mode'] == 'LCB_Q25'
        assert run['training_policy']['selection_surface'] == 'TRAIN_DISCOVERY_ONLY'
        assert run['claims']['nested_train_confirmation_enabled'] is True
        assert 2 <= policy['cohort_count'] <= 4
