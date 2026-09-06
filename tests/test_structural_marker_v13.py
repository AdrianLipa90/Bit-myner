from __future__ import annotations

from bitmyner.gremlin_marker import deterministic_headers, collect_samples
from bitmyner.marker_training_v13 import run_augmented_crossfit_training_v13
from bitmyner.structural_marker import (
    StructuralPredicate,
    augmented_structural_population,
    nonce_variation_filter,
    structural_population,
)


def _carry_mask_reference(x: int, y: int) -> int:
    carry = 0
    out = 0
    for bit in range(32):
        xb = (x >> bit) & 1
        yb = (y >> bit) & 1
        carry_out = (xb & yb) | (xb & carry) | (yb & carry)
        if carry_out:
            out |= 1 << bit
        carry = carry_out
    return out


def test_augmented_population_adds_temporal_and_carry_without_changing_base_default():
    base = structural_population()
    augmented = augmented_structural_population()
    assert all(p.op in {'direct', 'xor'} for p in base)
    assert any(p.op == 'xor_prev' for p in augmented)
    assert any(p.op == 'carry' for p in augmented)
    assert len(augmented) > len(base)
    base_ids = {p.predicate_id for p in base}
    assert base_ids.issubset({p.predicate_id for p in augmented})


def test_temporal_xor_and_carry_match_explicit_round_arithmetic():
    header = deterministic_headers(1, seed=1313)[0]
    sample = collect_samples((header,), nonces_per_header=1, event_k=2, nonce_seed=77)[0]

    temporal = StructuralPredicate(1, 7, 'xor_prev', 'a', 0, 0, 'a')
    got_temporal = temporal.expression_value(sample)
    expected_temporal = sample.trace.record(1, 7).a ^ sample.trace.record(1, 6).a
    assert got_temporal == expected_temporal

    carry = StructuralPredicate(2, 15, 'carry', 't1', 0, 0, 't2')
    r = sample.trace.record(2, 15)
    assert carry.expression_value(sample) == _carry_mask_reference(r.t1, r.t2)


def test_augmented_predicates_pass_through_unlabelled_nonce_variation_filter():
    split_seed = 'marker-v13-filter-test'
    headers = deterministic_headers(20, seed=1313)
    samples = collect_samples(headers, nonces_per_header=24, event_k=2, nonce_seed=991)
    augmented = augmented_structural_population(
        rounds=(3, 7),
        bits=(0, 7),
    )
    kept, receipt = nonce_variation_filter(samples, augmented, split_seed=split_seed)
    assert receipt['label_access'] is False
    assert receipt['kept_predicates'] == len(kept)
    assert any(p.op == 'xor_prev' for p in kept)
    assert any(p.op == 'carry' for p in kept)


def test_v13_tiny_crossfit_completes_without_production_promotion_authority():
    result = run_augmented_crossfit_training_v13(
        iterations=12,
        seeds=(1313, 1414, 1515),
        header_count=24,
        nonces_per_header=24,
        thresholds=(2, 3),
        population_size=10,
        offspring_per_iteration=2,
        max_predicates=2,
        train_discovery_fraction=0.70,
        requested_max_cohorts=2,
        min_expected_events_per_cohort=2.0,
    )
    assert result['status'] == 'PASS_AUGMENTED_CROSSFIT_V13_COMPLETE'
    assert result['feature_space']['temporal_xor_enabled'] is True
    assert result['feature_space']['carry_enabled'] is True
    assert result['claims']['production_policy_modified'] is False
    assert result['claims']['cryptographic_advantage_assumed'] is False
    assert result['total_iterations'] == 36
