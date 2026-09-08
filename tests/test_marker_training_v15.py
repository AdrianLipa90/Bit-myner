from bitmyner.marker_training_v15 import (
    economical_structural_population_v15,
    run_economical_crossfit_v15,
)
from bitmyner.structural_marker import augmented_structural_population


def test_economical_population_is_strict_early_subset():
    full = augmented_structural_population()
    early = economical_structural_population_v15(max_checkpoint_cost=32)
    assert 0 < len(early) < len(full)
    assert all(p.checkpoint_cost <= 32 for p in early)
    assert all(p.stage == 1 for p in early)
    assert {p.op for p in early} >= {'direct', 'xor', 'carry', 'xor_prev'}


def test_smoke_crossfit_v15_keeps_no_skip_and_nested_gates():
    r = run_economical_crossfit_v15(
        iterations=8,
        seeds=(101, 202, 303),
        header_count=34,
        nonces_per_header=24,
        thresholds=(2, 3),
        population_size=10,
        offspring_per_iteration=2,
        max_predicates=2,
        max_checkpoint_cost=16,
        desired_confirmation_events=1.0,
        min_discovery_headers=8,
        requested_max_cohorts=2,
        min_expected_events_per_cohort=1.0,
    )
    assert r['status'] == 'PASS_ECONOMICAL_EARLY_CHECKPOINT_CROSSFIT_V15_COMPLETE'
    assert r['claims']['feature_restriction_is_checkpoint_cost_only_not_event_label_based'] is True
    assert r['claims']['production_policy_modified'] is False
    assert r['priority_semantics']['deferred_candidates_discarded'] is False
    assert r['priority_semantics']['hard_rejection_authorized'] is False
    assert r['feature_space']['max_checkpoint_cost'] == 16
    for run in r['runs']:
        for row in run['final_top8']:
            if row.get('ladder'):
                assert row['confirmation_pass'] is True
