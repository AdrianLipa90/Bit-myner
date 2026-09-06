from bitmyner.marker_priority_v14 import annotate_priority_phase
from bitmyner.marker_training_v14 import (
    plan_confirmation_power_v14,
    run_power_balanced_augmented_crossfit_v14,
)


def test_priority_annotation_is_no_skip_and_preserves_gamma_identity():
    metric = {
        'n': 100,
        'marker_n': 25,
        'event_n': 10,
        'marker_event_n': 4,
        'lift': 1.6,
        'lift_lcb': 1.1,
        'gamma': 1.35,
        'gamma_lcb': 1.02,
        'compute_fraction': 0.41,
    }
    p = annotate_priority_phase(metric)
    assert p['priority_candidate_fraction'] == 0.25
    assert p['priority_event_recall'] == 0.4
    assert p['priority_event_density_gain'] == 1.35
    assert p['priority_event_density_gain_lcb'] == 1.02
    assert p['deferred_candidates_discarded'] is False
    assert p['hard_rejection_authorized'] is False
    assert p['exact_sha256d_coverage_preserved_if_deferred_completed'] is True


def test_power_plan_is_label_free_and_hits_target_when_capacity_exists():
    p = plan_confirmation_power_v14(
        header_count=320,
        nonces_per_header=256,
        hardest_k=8,
        split_seed='v14-power-test',
        header_seed=6060,
        desired_confirmation_events=64.0,
        min_discovery_headers=32,
    )
    assert p['label_access'] is False
    assert p['power_target_met'] is True
    assert p['confirmation_samples'] >= 64 * 256
    assert p['discovery_headers'] >= 32
    assert 0.0 < p['discovery_fraction'] < 1.0


def test_v14_small_run_keeps_validation_frozen_unless_confirmation_passes():
    r = run_power_balanced_augmented_crossfit_v14(
        iterations=8,
        seeds=(11, 22, 33),
        header_count=30,
        nonces_per_header=24,
        thresholds=(2, 3),
        population_size=10,
        offspring_per_iteration=2,
        max_predicates=2,
        desired_confirmation_events=1.0,
        min_discovery_headers=8,
        requested_max_cohorts=2,
        min_expected_events_per_cohort=1.0,
    )
    assert r['status'] == 'PASS_POWER_BALANCED_AUGMENTED_CROSSFIT_V14_COMPLETE'
    assert r['claims']['production_policy_modified'] is False
    assert r['claims']['gamma_does_not_authorize_hard_rejection'] is True
    assert len(r['confirmation_power_plans']) == 3
    for run in r['runs']:
        for row in run['final_top8']:
            if row.get('ladder'):
                assert row['confirmation_pass'] is True
            for level in row.get('confirmation_ladder', []):
                assert level['priority_phase']['deferred_candidates_discarded'] is False
