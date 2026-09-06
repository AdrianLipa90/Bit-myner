from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Sequence

from .gremlin_marker import deterministic_headers
from .marker_eval import split_name
from .marker_meta_v5 import summarize_multi_seed
from .marker_training_v4 import run_training_1000_v4

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_11_POWER_AWARE_LCB_CROSSFIT'


def _header_id(header76: bytes) -> str:
    return hashlib.blake2b(header76, digest_size=16).hexdigest()


def adaptive_cohort_count_v11(
    *,
    header_count: int,
    nonces_per_header: int,
    hardest_k: int,
    train_discovery_fraction: float,
    split_seed: str,
    header_seed: int,
    requested_max_cohorts: int = 4,
    min_expected_events_per_cohort: float = 8.0,
) -> dict:
    """Choose discovery cohort count from expected surrogate-event power only.

    This function never reads observed low-hash labels. It reproduces the deterministic
    header partition and nested TRAIN discovery split, then uses p=2^-k to budget folds.
    """
    if header_count < 1 or nonces_per_header < 1:
        raise ValueError('header_count and nonces_per_header must be positive')
    if hardest_k < 1 or hardest_k > 32:
        raise ValueError('hardest_k must be in [1,32]')
    if not 0.0 < float(train_discovery_fraction) < 1.0:
        raise ValueError('v0.11 requires a strict discovery/confirmation split')
    if requested_max_cohorts < 2:
        raise ValueError('requested_max_cohorts must be >= 2')
    if min_expected_events_per_cohort <= 0.0:
        raise ValueError('min_expected_events_per_cohort must be positive')

    headers = deterministic_headers(int(header_count), seed=int(header_seed))
    train_ids = sorted(
        _header_id(h) for h in headers
        if split_name(_header_id(h), seed=split_seed) == 'train'
    )
    if len(train_ids) < 10:
        discovery_headers = train_ids
        confirmation_headers: list[str] = []
    else:
        keyed = sorted(
            (hashlib.blake2b(f'{split_seed}|outer-train|{h}'.encode(), digest_size=8).digest(), h)
            for h in train_ids
        )
        ordered = [h for _, h in keyed]
        discovery_count = max(
            6,
            min(len(ordered) - 4, int(round(len(ordered) * float(train_discovery_fraction))))
        )
        discovery_headers = ordered[:discovery_count]
        confirmation_headers = ordered[discovery_count:]

    discovery_samples = len(discovery_headers) * int(nonces_per_header)
    expected_events_total = discovery_samples / float(1 << int(hardest_k))
    max_by_power = int(math.floor(expected_events_total / float(min_expected_events_per_cohort)))
    if len(discovery_headers) >= 2:
        cohort_count = min(
            int(requested_max_cohorts),
            len(discovery_headers),
            max(2, max_by_power),
        )
    else:
        cohort_count = 2

    return {
        'schema': 'BITMYNER_MARKER_V0_11_LABEL_FREE_COHORT_POWER_PLAN',
        'header_count': int(header_count),
        'nonces_per_header': int(nonces_per_header),
        'train_headers': len(train_ids),
        'discovery_headers': len(discovery_headers),
        'confirmation_headers': len(confirmation_headers),
        'discovery_samples': discovery_samples,
        'hardest_k': int(hardest_k),
        'event_probability_model': f'2^-{int(hardest_k)}',
        'expected_events_total': expected_events_total,
        'min_expected_events_per_cohort': float(min_expected_events_per_cohort),
        'requested_max_cohorts': int(requested_max_cohorts),
        'max_cohorts_by_expected_power': max_by_power,
        'cohort_count': int(cohort_count),
        'expected_events_per_cohort': expected_events_total / max(1, int(cohort_count)),
        'power_target_met': bool(max_by_power >= 2),
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


def run_nested_crossfit_training_v11(
    *,
    iterations: int = 1000,
    seeds: Sequence[int] = (4040, 4141, 4242),
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
    seeds = tuple(int(s) for s in seeds)
    thresholds = tuple(int(k) for k in thresholds)
    if len(seeds) < 3:
        raise ValueError('at least three island seeds are required')
    if not thresholds:
        raise ValueError('at least one threshold is required')
    if not 0.0 < float(train_discovery_fraction) < 1.0:
        raise ValueError('v0.11 requires a strict discovery/confirmation split')

    runs: list[dict] = []
    cohort_policies: list[dict] = []
    confirmation_counts: list[int] = []
    validation_open_counts: list[int] = []

    for i, seed in enumerate(seeds):
        split_seed = f'bitmyner-marker-v0.11-island-{i}'
        power = adaptive_cohort_count_v11(
            header_count=header_count,
            nonces_per_header=nonces_per_header,
            hardest_k=max(thresholds),
            train_discovery_fraction=train_discovery_fraction,
            split_seed=split_seed,
            header_seed=seed,
            requested_max_cohorts=requested_max_cohorts,
            min_expected_events_per_cohort=min_expected_events_per_cohort,
        )
        run = run_training_1000_v4(
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
            cohort_count=int(power['cohort_count']),
            cohort_refresh_every=max(5, min(25, iterations // 8 if iterations >= 8 else 5)),
            family_prior=None,
            family_prior_strength=0.0,
            final_evaluate=True,
            train_discovery_fraction=train_discovery_fraction,
            cohort_robust_mode='LCB_Q25',
        )
        runs.append(run)
        cohort_policies.append(power)
        passed = [row for row in run.get('final_top8', []) if row.get('confirmation_pass', False)]
        opened = [row for row in run.get('final_top8', []) if row.get('ladder')]
        confirmation_counts.append(len(passed))
        validation_open_counts.append(len(opened))

    filtered = [_confirmation_filtered_run(run) for run in runs]
    cross = summarize_multi_seed(filtered, minimum_seed_support=minimum_seed_support)
    duplicates = [float(r['search']['duplicate_proposal_fraction']) for r in runs]
    robust = [float(r['checkpoints'][-1]['robust_core']) for r in runs]
    promoted = [int(r.get('promoted_count', 0)) for r in runs]
    best_confirmation = [_best_confirmation_lcb(r) for r in runs]

    confirmation_power = []
    for run in runs:
        n_headers = int(run['training_policy']['train_confirmation_headers'])
        n_samples = n_headers * int(nonces_per_header)
        confirmation_power.append({
            'confirmation_headers': n_headers,
            'confirmation_samples': n_samples,
            'expected_events': {str(k): n_samples / float(1 << int(k)) for k in thresholds},
            'label_access': False,
        })

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_NESTED_CROSSFIT_V11_COMPLETE',
        'iterations_per_island': int(iterations),
        'total_iterations': int(iterations) * len(runs),
        'seeds': list(seeds),
        'fixture': {
            'header_count_per_island': int(header_count),
            'nonces_per_header': int(nonces_per_header),
            'thresholds': list(thresholds),
            'train_discovery_fraction': float(train_discovery_fraction),
        },
        'cohort_policies': cohort_policies,
        'confirmation_power': confirmation_power,
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
        'claims': {
            'cohort_robustness_uses_lower_confidence_bounds': True,
            'adaptive_cohort_count_is_label_free': True,
            'cohort_count_uses_expected_2^-k_power_not_observed_events': True,
            'meta_uses_confirmation_passed_candidates_only': True,
            'validation_and_holdout_are_not_training_surfaces': True,
            'validation_holdout_open_only_after_nested_train_confirmation': True,
            'cryptographic_advantage_assumed': False,
            'production_policy_modified': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
