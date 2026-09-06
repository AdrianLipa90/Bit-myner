from __future__ import annotations

from bitmyner.gremlin_marker import deterministic_headers, collect_samples
from bitmyner.marker_training_v4 import run_training_1000_v4
from bitmyner.marker_null_v12 import (
    confirmation_partition_v12,
    permute_confirmation_event_masks_v12,
    calibrate_confirmation_run_v12,
)
from bitmyner.structural_marker import (
    StructuralCompiledCorpus,
    nonce_variation_filter,
    structural_population,
)


def _bit_count_in(mask: int, subset: int) -> int:
    return (mask & subset).bit_count()


def test_within_header_permutation_preserves_counts_and_threshold_nesting():
    split_seed = 'marker-null-v12-test'
    headers = deterministic_headers(24, seed=4040)
    samples = collect_samples(
        headers,
        nonces_per_header=24,
        event_k=2,
        nonce_seed=4040 ^ 0x5EED,
    )
    raw = structural_population(rounds=(0, 3), bits=(0, 7))
    predicates, _ = nonce_variation_filter(samples, raw, split_seed=split_seed)
    corpus = StructuralCompiledCorpus(samples, predicates, (2, 3), split_seed)
    confirmation_headers, confirmation_mask = confirmation_partition_v12(
        corpus,
        split_seed=split_seed,
        discovery_fraction=0.70,
    )
    permuted = permute_confirmation_event_masks_v12(
        corpus,
        confirmation_headers=confirmation_headers,
        thresholds=(2, 3),
        seed=12345,
    )

    assert confirmation_mask != 0
    assert permuted[3] & ~permuted[2] == 0
    for header_id in confirmation_headers:
        hmask = corpus.header_masks['train'][header_id]
        for k in (2, 3):
            assert _bit_count_in(permuted[k], hmask) == _bit_count_in(
                corpus.event_masks['train'][k], hmask
            )


def test_confirmation_null_calibration_is_deterministic_and_read_only():
    run = run_training_1000_v4(
        iterations=8,
        header_count=28,
        nonces_per_header=24,
        thresholds=(2, 3),
        population_size=8,
        offspring_per_iteration=2,
        max_predicates=2,
        seed=4242,
        split_seed='marker-null-v12-small-run',
        checkpoint_every=4,
        cohort_count=2,
        cohort_refresh_every=4,
        final_evaluate=True,
        train_discovery_fraction=0.70,
        cohort_robust_mode='LCB_Q25',
    )

    a = calibrate_confirmation_run_v12(run, permutations=16, calibration_seed=9001)
    b = calibrate_confirmation_run_v12(run, permutations=16, calibration_seed=9001)

    assert a == b
    assert a['permutations'] == 16
    assert 0.0 <= a['familywise_false_pass_rate'] <= 1.0
    assert 0.0 < a['empirical_p_max_gamma_lcb'] <= 1.0
    assert a['label_permutation_scope'] == 'WITHIN_CONFIRMATION_HEADER'
    assert a['threshold_nesting_preserved'] is True
    assert a['per_header_event_counts_preserved'] is True
    assert a['production_policy_modified'] is False
