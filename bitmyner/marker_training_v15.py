from __future__ import annotations

import hashlib
import json
import statistics
from typing import Sequence

from . import marker_training_v4 as _v4
from .marker_meta_v5 import summarize_multi_seed
from .marker_training_v11 import adaptive_cohort_count_v11
from .marker_training_v14 import (
    _annotate_run_priority,
    _best_confirmation_lcb,
    _confirmation_filtered_run,
    plan_confirmation_power_v14,
)
from .structural_marker import augmented_structural_population

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_15_ECONOMICAL_EARLY_CHECKPOINT_CROSSFIT'


def economical_structural_population_v15(*, max_checkpoint_cost: int = 32):
    cap = int(max_checkpoint_cost)
    if not 1 <= cap < 128:
        raise ValueError('max_checkpoint_cost must be in [1,127]')
    out = tuple(p for p in augmented_structural_population() if p.checkpoint_cost <= cap)
    if not out:
        raise RuntimeError('economical predicate space is empty')
    return out


def run_economical_crossfit_v15(
    *,
    iterations: int = 1000,
    seeds: Sequence[int] = (7070, 7171, 7272),
    header_count: int = 320,
    nonces_per_header: int = 256,
    thresholds: Sequence[int] = (4, 6, 8),
    population_size: int = 64,
    offspring_per_iteration: int = 8,
    max_predicates: int = 3,
    max_checkpoint_cost: int = 32,
    desired_confirmation_events: float = 64.0,
    min_discovery_headers: int = 32,
    minimum_seed_support: int = 2,
    requested_max_cohorts: int = 4,
    min_expected_events_per_cohort: float = 8.0,
) -> dict:
    seeds = tuple(int(s) for s in seeds)
    thresholds = tuple(int(k) for k in thresholds)
    if len(seeds) < 3:
        raise ValueError('at least three island seeds are required')
    if not thresholds:
        raise ValueError('at least one threshold is required')

    economical = economical_structural_population_v15(max_checkpoint_cost=max_checkpoint_cost)
    original_provider = _v4.structural_population
    runs: list[dict] = []
    power_plans: list[dict] = []
    cohort_policies: list[dict] = []
    confirmation_counts: list[int] = []
    validation_open_counts: list[int] = []

    _v4.structural_population = lambda: economical
    try:
        for i, seed in enumerate(seeds):
            split_seed = f'bitmyner-marker-v0.15-economical-island-{i}'
            plan = plan_confirmation_power_v14(
                header_count=header_count,
                nonces_per_header=nonces_per_header,
                hardest_k=max(thresholds),
                split_seed=split_seed,
                header_seed=seed,
                desired_confirmation_events=desired_confirmation_events,
                min_discovery_headers=min_discovery_headers,
            )
            cohort = adaptive_cohort_count_v11(
                header_count=header_count,
                nonces_per_header=nonces_per_header,
                hardest_k=max(thresholds),
                train_discovery_fraction=float(plan['discovery_fraction']),
                split_seed=split_seed,
                header_seed=seed,
                requested_max_cohorts=requested_max_cohorts,
                min_expected_events_per_cohort=min_expected_events_per_cohort,
            )
            run = _v4.run_training_1000_v4(
                iterations=iterations,
                header_count=header_count,
                nonces_per_header=nonces_per_header,
                thresholds=thresholds,
                population_size=population_size,
                offspring_per_iteration=offspring_per_iteration,
                max_predicates=max_predicates,
                seed=seed,
                split_seed=split_seed,
                checkpoint_every=max(1, iterations // 10),
                cohort_count=int(cohort['cohort_count']),
                cohort_refresh_every=max(5, min(25, iterations // 8 if iterations >= 8 else 5)),
                family_prior=None,
                family_prior_strength=0.0,
                final_evaluate=True,
                train_discovery_fraction=float(plan['discovery_fraction']),
                cohort_robust_mode='LCB_Q25',
            )
            if int(run['training_policy']['train_confirmation_headers']) != int(plan['confirmation_headers']):
                raise RuntimeError('power-plan confirmation mismatch')
            run = _annotate_run_priority(run)
            runs.append(run)
            power_plans.append(plan)
            cohort_policies.append(cohort)
            confirmation_counts.append(sum(bool(row.get('confirmation_pass')) for row in run.get('final_top8', [])))
            validation_open_counts.append(sum(bool(row.get('ladder')) for row in run.get('final_top8', [])))
    finally:
        _v4.structural_population = original_provider

    filtered = [_confirmation_filtered_run(run) for run in runs]
    cross = summarize_multi_seed(filtered, minimum_seed_support=minimum_seed_support)
    duplicates = [float(r['search']['duplicate_proposal_fraction']) for r in runs]
    robust = [float(r['checkpoints'][-1]['robust_core']) for r in runs]
    promoted = [int(r.get('promoted_count', 0)) for r in runs]
    best_confirmation = [_best_confirmation_lcb(r) for r in runs]

    op_counts: dict[str, int] = {}
    checkpoint_counts: dict[str, int] = {}
    for predicate in economical:
        op_counts[predicate.op] = op_counts.get(predicate.op, 0) + 1
        checkpoint_counts[str(predicate.checkpoint_cost)] = checkpoint_counts.get(str(predicate.checkpoint_cost), 0) + 1

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_ECONOMICAL_EARLY_CHECKPOINT_CROSSFIT_V15_COMPLETE',
        'iterations_per_island': int(iterations),
        'total_iterations': int(iterations) * len(runs),
        'seeds': list(seeds),
        'fixture': {
            'header_count_per_island': int(header_count),
            'nonces_per_header': int(nonces_per_header),
            'thresholds': list(thresholds),
        },
        'feature_space': {
            'predicate_count': len(economical),
            'max_checkpoint_cost': int(max_checkpoint_cost),
            'operator_counts': op_counts,
            'checkpoint_cost_counts': checkpoint_counts,
            'unlabelled_nonce_variation_filter_retained': True,
            'late_predicates_excluded_for_computational_utility': True,
        },
        'confirmation_power_plans': power_plans,
        'cohort_policies': cohort_policies,
        'runs': runs,
        'confirmation_pass_counts': confirmation_counts,
        'validation_holdout_open_counts': validation_open_counts,
        'best_confirmation_gamma_lcb_by_island': best_confirmation,
        'mean_best_confirmation_gamma_lcb': float(statistics.mean(best_confirmation)),
        'mean_duplicate_fraction': float(statistics.mean(duplicates)),
        'mean_final_robust_core': float(statistics.mean(robust)),
        'promoted_counts': promoted,
        'crossfit_meta': cross,
        'external_evaluation_authorized_candidates': cross.get('consensus_exact_genomes', [])[:8],
        'priority_semantics': {
            'deferred_candidates_discarded': False,
            'hard_rejection_authorized': False,
            'exact_sha256d_coverage_after_deferred_completion': True,
            'production_filter_claim': False,
        },
        'claims': {
            'feature_restriction_is_checkpoint_cost_only_not_event_label_based': True,
            'confirmation_allocation_is_label_free': True,
            'validation_holdout_open_only_after_nested_train_confirmation': True,
            'cryptographic_advantage_assumed': False,
            'production_policy_modified': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
