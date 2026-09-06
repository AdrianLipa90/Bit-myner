from __future__ import annotations

import hashlib
import json
from typing import Sequence

from . import marker_training_v4 as _v4
from .marker_training_v11 import run_nested_crossfit_training_v11
from .structural_marker import augmented_structural_population, structural_population

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_13_AUGMENTED_STRUCTURAL_CROSSFIT'


def run_augmented_crossfit_training_v13(
    *,
    iterations: int = 1000,
    seeds: Sequence[int] = (5050, 5151, 5252),
    header_count: int = 160,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4, 6, 8),
    population_size: int = 64,
    offspring_per_iteration: int = 8,
    max_predicates: int = 3,
    train_discovery_fraction: float = 0.72,
    minimum_seed_support: int = 2,
    requested_max_cohorts: int = 4,
    min_expected_events_per_cohort: float = 8.0,
) -> dict:
    """Run v0.11 crossfit unchanged, but with a frozen augmented predicate provider.

    The provider override is local to this synchronous research call and restored in
    `finally`; no production path or older default feature space is changed.
    """
    augmented = augmented_structural_population()
    original_provider = _v4.structural_population
    _v4.structural_population = lambda: augmented
    try:
        base = run_nested_crossfit_training_v11(
            iterations=iterations,
            seeds=seeds,
            header_count=header_count,
            nonces_per_header=nonces_per_header,
            thresholds=thresholds,
            population_size=population_size,
            offspring_per_iteration=offspring_per_iteration,
            max_predicates=max_predicates,
            train_discovery_fraction=train_discovery_fraction,
            minimum_seed_support=minimum_seed_support,
            requested_max_cohorts=requested_max_cohorts,
            min_expected_events_per_cohort=min_expected_events_per_cohort,
        )
    finally:
        _v4.structural_population = original_provider

    op_counts: dict[str, int] = {}
    for predicate in augmented:
        op_counts[predicate.op] = op_counts.get(predicate.op, 0) + 1

    payload = dict(base)
    payload['schema'] = SCHEMA
    payload['status'] = 'PASS_AUGMENTED_CROSSFIT_V13_COMPLETE'
    payload['base_engine_schema'] = base['schema']
    payload['base_engine_receipt_blake2b256'] = base['receipt_blake2b256']
    payload['feature_space'] = {
        'base_predicate_count': len(structural_population()),
        'augmented_predicate_count': len(augmented),
        'operator_counts': op_counts,
        'temporal_xor_enabled': op_counts.get('xor_prev', 0) > 0,
        'carry_enabled': op_counts.get('carry', 0) > 0,
        'old_base_population_modified': False,
        'unlabelled_nonce_variation_filter_retained': True,
        'provider_override_scope': 'SYNCHRONOUS_RESEARCH_CALL_ONLY_RESTORED_IN_FINALLY',
    }
    claims = dict(payload.get('claims', {}))
    claims.update({
        'cryptographic_advantage_assumed': False,
        'production_policy_modified': False,
        'augmented_features_receive_no_special_fitness_bonus': True,
    })
    payload['claims'] = claims
    payload.pop('receipt_blake2b256', None)
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
