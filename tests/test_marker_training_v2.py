import pytest

from bitmyner.marker_training_v2 import (
    NegativeMemory,
    _genome_distance,
    _novelty_score,
    run_training_1000_v2,
)


def test_genome_distance_and_novelty_are_bounded():
    assert _genome_distance((1, 2), (1, 2)) == 0.0
    assert _genome_distance((1,), (2,)) == 1.0
    assert _genome_distance((1, 2), (2, 3)) == pytest.approx(2 / 3)
    n = _novelty_score((1, 2), [(1, 2), (2, 3), (8, 9)], k=2)
    assert 0.0 <= n <= 1.0


def test_negative_memory_penalty_increases_for_repeated_failed_family():
    mem = NegativeMemory()
    assert mem.penalty((7, 9)) == 0.0
    mem.observe((7, 9), severity=1.0)
    first = mem.penalty((7, 9))
    mem.observe((7, 9), severity=1.0)
    second = mem.penalty((7, 9))
    assert first > 0.0
    assert second >= first
    assert mem.penalty((123,)) <= second


def test_training_v2_is_train_only_during_evolution_and_uses_rolling_cohorts():
    r = run_training_1000_v2(
        iterations=24,
        header_count=20,
        nonces_per_header=24,
        thresholds=(2, 3),
        population_size=12,
        offspring_per_iteration=2,
        max_predicates=2,
        seed=616,
        checkpoint_every=12,
        cohort_count=3,
        cohort_refresh_every=4,
    )
    assert r['status'] == 'PASS_TRAINING_COMPLETE'
    assert r['iterations'] == 24
    p = r['training_policy']
    assert p['selection_surface'] == 'TRAIN_COHORTS_ONLY'
    assert p['validation_role'] == 'FROZEN_UNTIL_TRAINING_COMPLETE_NO_SELECTION_AUTHORITY'
    assert p['holdout_role'] == 'FROZEN_UNTIL_TRAINING_COMPLETE'
    assert p['novelty_search'] is True
    assert p['negative_memory'] is True
    assert p['genealogy_penalty'] is True
    assert p['rolling_header_cohorts'] is True
    assert r['claims']['validation_used_during_training'] is False
    assert r['claims']['holdout_used_during_training'] is False
    assert r['search']['cohort_refreshes'] >= 1
    assert r['search']['unique_genomes_evaluated'] > 0
    assert r['search']['operator_stats']['restart']['proposed'] >= 0


def test_training_v2_is_deterministic_for_same_seed():
    kwargs = dict(
        iterations=16,
        header_count=18,
        nonces_per_header=20,
        thresholds=(2, 3),
        population_size=10,
        offspring_per_iteration=2,
        max_predicates=2,
        seed=777,
        checkpoint_every=8,
        cohort_count=3,
        cohort_refresh_every=4,
    )
    a = run_training_1000_v2(**kwargs)
    b = run_training_1000_v2(**kwargs)
    assert a['receipt_blake2b256'] == b['receipt_blake2b256']
    assert a['search'] == b['search']
    assert a['final_top8'] == b['final_top8']
