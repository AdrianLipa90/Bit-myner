from bitmyner.marker_training import run_training_1000


def test_training_is_holdout_blind_and_bounded():
    r = run_training_1000(
        iterations=20,
        header_count=18,
        nonces_per_header=24,
        thresholds=(2, 3),
        population_size=12,
        offspring_per_iteration=2,
        max_predicates=2,
        seed=616,
        checkpoint_every=10,
    )
    assert r['status'] == 'PASS_TRAINING_COMPLETE'
    assert r['iterations'] == 20
    assert r['training_policy']['selection_surface'] == 'TRAIN_ONLY'
    assert r['training_policy']['holdout_role'] == 'FROZEN_UNTIL_ITERATION_1000_COMPLETE'
    assert r['claims']['holdout_used_during_training'] is False
    assert r['claims']['production_policy_changed'] is False
    assert r['search']['unique_genomes_evaluated'] > 0
