from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Sequence

from . import marker_training_v4 as _v4
from .gremlin_marker import deterministic_headers
from .marker_eval import split_name
from .marker_meta_v5 import summarize_multi_seed
from .marker_priority_v14 import annotate_priority_phase
from .marker_training_v11 import adaptive_cohort_count_v11
from .structural_marker import augmented_structural_population, structural_population

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_14_POWER_BALANCED_AUGMENTED_CROSSFIT'


def _header_id(header76: bytes) -> str:
    return hashlib.blake2b(header76, digest_size=16).hexdigest()


def plan_confirmation_power_v14(
    *,
    header_count: int,
    nonces_per_header: int,
    hardest_k: int,
    split_seed: str,
    header_seed: int,
    desired_confirmation_events: float = 64.0,
    min_discovery_headers: int = 32,
    min_confirmation_headers: int = 4,
) -> dict:
    """Choose the nested TRAIN confirmation allocation without reading event labels."""
    if header_count < 1 or nonces_per_header < 1:
        raise ValueError('header_count and nonces_per_header must be positive')
    if not 1 <= int(hardest_k) <= 32:
        raise ValueError('hardest_k must be in [1,32]')
    if desired_confirmation_events <= 0:
        raise ValueError('desired_confirmation_events must be positive')
    if min_discovery_headers < 6:
        raise ValueError('min_discovery_headers must be >= 6')
    if min_confirmation_headers < 4:
        raise ValueError('min_confirmation_headers must be >= 4')

    headers = deterministic_headers(int(header_count), seed=int(header_seed))
    train_ids = sorted(
        _header_id(h) for h in headers
        if split_name(_header_id(h), seed=split_seed) == 'train'
    )
    if len(train_ids) < min_discovery_headers + min_confirmation_headers:
        raise RuntimeError('insufficient outer TRAIN headers for power-balanced nested split')

    required_samples = int(math.ceil(float(desired_confirmation_events) * float(1 << int(hardest_k))))
    required_headers = int(math.ceil(required_samples / int(nonces_per_header)))
    max_confirmation_headers = len(train_ids) - int(min_discovery_headers)
    confirmation_headers = min(
        max_confirmation_headers,
        max(int(min_confirmation_headers), required_headers),
    )
    discovery_headers = len(train_ids) - confirmation_headers
    discovery_fraction = discovery_headers / len(train_ids)
    confirmation_samples = confirmation_headers * int(nonces_per_header)
    expected_events = confirmation_samples / float(1 << int(hardest_k))

    return {
        'schema': 'BITMYNER_MARKER_V0_14_LABEL_FREE_CONFIRMATION_POWER_PLAN',
        'header_count': int(header_count),
        'nonces_per_header': int(nonces_per_header),
        'outer_train_headers': len(train_ids),
        'discovery_headers': discovery_headers,
        'confirmation_headers': confirmation_headers,
        'discovery_fraction': discovery_fraction,
        'hardest_k': int(hardest_k),
        'desired_confirmation_events': float(desired_confirmation_events),
        'required_confirmation_samples': required_samples,
        'required_confirmation_headers': required_headers,
        'confirmation_samples': confirmation_samples,
        'expected_confirmation_events': expected_events,
        'power_target_met': bool(expected_events >= float(desired_confirmation_events)),
        'allocation_capped_by_min_discovery_headers': bool(required_headers > max_confirmation_headers),
        'label_access': False,
    }


def _confirmation_filtered_run(run: dict) -> dict:
    out = dict(run)
    out['final_top8'] = [row for row in run.get('final_top8', []) if row.get('confirmation_pass', False)]
    return out


def _best_confirmation_lcb(run: dict) -> float:
    best = 0.0
    for row in run.get('final_top8', []):
        for level in row.get('confirmation_ladder', []):
            metric = level.get('confirmation') or {}
            best = max(best, float(metric.get('gamma_lcb', 0.0)))
    return best


def _annotate_run_priority(run: dict) -> dict:
    out = dict(run)
    rows = []
    for row in run.get('final_top8', []):
        rr = dict(row)
        cl = []
        for level in row.get('confirmation_ladder', []):
            ll = dict(level)
            metric = level.get('confirmation')
            if metric:
                ll['priority_phase'] = annotate_priority_phase(metric)
            cl.append(ll)
        rr['confirmation_ladder'] = cl
        ladder = []
        for level in row.get('ladder', []):
            ll = dict(level)
            if level.get('validation'):
                ll['validation_priority_phase'] = annotate_priority_phase(level['validation'])
            if level.get('holdout'):
                ll['holdout_priority_phase'] = annotate_priority_phase(level['holdout'])
            ladder.append(ll)
        rr['ladder'] = ladder
        rows.append(rr)
    out['final_top8'] = rows
    return out


def run_power_balanced_augmented_crossfit_v14(
    *,
    iterations: int = 1000,
    seeds: Sequence[int] = (6060, 6161, 6262),
    header_count: int = 320,
    nonces_per_header: int = 256,
    thresholds: Sequence[int] = (4, 6, 8),
    population_size: int = 64,
    offspring_per_iteration: int = 8,
    max_predicates: int = 3,
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

    augmented = augmented_structural_population()
    original_provider = _v4.structural_population
    runs: list[dict] = []
    power_plans: list[dict] = []
    cohort_policies: list[dict] = []
    confirmation_counts: list[int] = []
    validation_open_counts: list[int] = []

    _v4.structural_population = lambda: augmented
    try:
        for i, seed in enumerate(seeds):
            split_seed = f'bitmyner-marker-v0.14-island-{i}'
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
            actual_confirmation = int(run['training_policy']['train_confirmation_headers'])
            if actual_confirmation != int(plan['confirmation_headers']):
                raise RuntimeError(
                    f'power-plan confirmation mismatch: planned={plan["confirmation_headers"]} actual={actual_confirmation}'
                )
            run = _annotate_run_priority(run)
            runs.append(run)
            power_plans.append(plan)
            cohort_policies.append(cohort)
            passed = [row for row in run.get('final_top8', []) if row.get('confirmation_pass', False)]
            opened = [row for row in run.get('final_top8', []) if row.get('ladder')]
            confirmation_counts.append(len(passed))
            validation_open_counts.append(len(opened))
    finally:
        _v4.structural_population = original_provider

    filtered = [_confirmation_filtered_run(run) for run in runs]
    cross = summarize_multi_seed(filtered, minimum_seed_support=minimum_seed_support)
    duplicates = [float(r['search']['duplicate_proposal_fraction']) for r in runs]
    robust = [float(r['checkpoints'][-1]['robust_core']) for r in runs]
    promoted = [int(r.get('promoted_count', 0)) for r in runs]
    best_confirmation = [_best_confirmation_lcb(r) for r in runs]

    op_counts: dict[str, int] = {}
    for predicate in augmented:
        op_counts[predicate.op] = op_counts.get(predicate.op, 0) + 1

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_POWER_BALANCED_AUGMENTED_CROSSFIT_V14_COMPLETE',
        'iterations_per_island': int(iterations),
        'total_iterations': int(iterations) * len(runs),
        'seeds': list(seeds),
        'fixture': {
            'header_count_per_island': int(header_count),
            'nonces_per_header': int(nonces_per_header),
            'thresholds': list(thresholds),
        },
        'feature_space': {
            'base_predicate_count': len(structural_population()),
            'augmented_predicate_count': len(augmented),
            'operator_counts': op_counts,
            'unlabelled_nonce_variation_filter_retained': True,
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
            'gamma_interpreted_as': 'MARKER_POSITIVE_PRIORITY_PHASE_EVENT_DENSITY_PER_SHA_ROUND',
            'deferred_candidates_discarded': False,
            'exhaustive_sha256d_coverage_after_deferred_completion': True,
            'hard_rejection_authorized': False,
            'production_filter_claim': False,
        },
        'claims': {
            'confirmation_allocation_is_label_free': True,
            'confirmation_power_target_uses_expected_2^-k_not_observed_events': True,
            'cohort_count_is_label_free': True,
            'cohort_robustness_uses_lower_confidence_bounds': True,
            'validation_holdout_open_only_after_nested_train_confirmation': True,
            'gamma_does_not_authorize_hard_rejection': True,
            'cryptographic_advantage_assumed': False,
            'production_policy_modified': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
