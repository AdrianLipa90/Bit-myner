from bitmyner.marker_training_v10 import run_nested_crossfit_training_v10


def test_v10_runs_nested_islands_and_filters_meta_to_confirmation_passes():
    r = run_nested_crossfit_training_v10(
        iterations=3,
        seeds=(1010, 1111, 1212),
        header_count=30,
        nonces_per_header=20,
        thresholds=(2,3),
        population_size=12,
        offspring_per_iteration=2,
        max_predicates=2,
        train_discovery_fraction=0.70,
    )
    assert r['status'] == 'PASS_NESTED_CROSSFIT_TRAINING_COMPLETE'
    assert r['total_iterations'] == 9
    assert len(r['runs']) == 3
    assert all(x['training_policy']['selection_surface'] == 'TRAIN_DISCOVERY_ONLY' for x in r['runs'])
    assert all(x['claims']['nested_train_confirmation_enabled'] is True for x in r['runs'])
    assert r['claims']['meta_uses_confirmation_passed_candidates_only'] is True
    assert isinstance(r['external_evaluation_authorized_candidates'], list)
