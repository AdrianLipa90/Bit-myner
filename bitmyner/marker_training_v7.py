from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .gremlin_marker import deterministic_headers
from .marker_meta_v5 import summarize_multi_seed
from .marker_training_v4 import run_training_1000_v4

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_7_ADAPTIVE_PRIOR_ISLANDS'
CALIBRATION_SCHEMA = 'BITMYNER_MARKER_PRIOR_CALIBRATION_V0_7'


def _recurrent_prior(meta_summary: dict, minimum_seed_support: int) -> dict[str, float]:
    if meta_summary.get('status') != 'PASS_META_COMPLETE':
        raise ValueError('meta summary must be complete')
    rows = [
        r for r in meta_summary.get('consensus_component_families', [])
        if int(r.get('seed_support', 0)) >= int(minimum_seed_support)
    ]
    prior = {str(r['family']): float(r['seed_support']) for r in rows}
    if not prior:
        raise RuntimeError('no recurrent training-only component families')
    return prior


def calibrate_family_prior_v7(
    meta_summary: dict,
    *,
    strengths: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    pilot_iterations: int = 120,
    seed: int = 1717,
    header_count: int = 60,
    nonces_per_header: int = 96,
    thresholds: Sequence[int] = (4, 6, 8),
    minimum_seed_support: int = 2,
) -> dict:
    prior = _recurrent_prior(meta_summary, minimum_seed_support)
    strengths = tuple(sorted(set(max(0.0, float(s)) for s in strengths)))
    if not strengths:
        raise ValueError('at least one prior strength is required')
    rows = []
    for strength in strengths:
        run = run_training_1000_v4(
            iterations=pilot_iterations,
            header_count=header_count,
            nonces_per_header=nonces_per_header,
            thresholds=thresholds,
            population_size=48,
            offspring_per_iteration=6,
            max_predicates=3,
            seed=seed,
            split_seed='bitmyner-marker-v0.7-prior-calibration',
            checkpoint_every=max(1, pilot_iterations // 4),
            cohort_count=4,
            cohort_refresh_every=max(5, pilot_iterations // 8),
            family_prior=prior,
            family_prior_strength=strength,
            final_evaluate=False,
        )
        c = run['checkpoints'][-1]
        dup = float(run['search']['duplicate_proposal_fraction'])
        robust = float(c['robust_core'])
        instability = float(c['instability'])
        negative = float(c['negative_penalty'])
        objective = robust - 0.30 * dup - 0.05 * instability - 0.02 * negative
        rows.append({
            'strength': strength,
            'objective': objective,
            'robust_core': robust,
            'duplicate_fraction': dup,
            'instability': instability,
            'negative_penalty': negative,
            'receipt_blake2b256': run['receipt_blake2b256'],
            'post_training_evaluation_performed': run['claims']['post_training_evaluation_performed'],
        })
    chosen = max(rows, key=lambda r: (r['objective'], -r['duplicate_fraction'], -r['strength']))
    payload = {
        'schema': CALIBRATION_SCHEMA,
        'status': 'PASS_PRIOR_CALIBRATION_COMPLETE',
        'training_only_prior': prior,
        'minimum_seed_support': int(minimum_seed_support),
        'pilot_iterations': int(pilot_iterations),
        'fixture': {
            'seed': int(seed),
            'header_count': int(header_count),
            'nonces_per_header': int(nonces_per_header),
            'thresholds': [int(k) for k in thresholds],
        },
        'candidates': rows,
        'chosen_strength': float(chosen['strength']),
        'chosen_objective': float(chosen['objective']),
        'claims': {
            'validation_opened_during_calibration': False,
            'holdout_opened_during_calibration': False,
            'calibration_uses_training_only_threshold_labels': True,
            'production_policy_modified': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload


def run_adaptive_island_training_v7(
    meta_summary: dict,
    *,
    iterations: int = 1000,
    seeds: Sequence[int] = (1919, 2020, 2121),
    header_count: int = 100,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4, 6, 8),
    minimum_seed_support: int = 2,
    strengths: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    pilot_iterations: int = 120,
    calibration_seed: int = 1717,
) -> dict:
    if len(seeds) < 3:
        raise ValueError('at least three full-training island seeds required')
    prior = _recurrent_prior(meta_summary, minimum_seed_support)
    calibration = calibrate_family_prior_v7(
        meta_summary,
        strengths=strengths,
        pilot_iterations=pilot_iterations,
        seed=calibration_seed,
        header_count=min(60, header_count),
        nonces_per_header=min(96, nonces_per_header),
        thresholds=thresholds,
        minimum_seed_support=minimum_seed_support,
    )
    chosen_strength = float(calibration['chosen_strength'])

    calibration_headers = set(deterministic_headers(min(60, header_count), seed=calibration_seed))
    full_headers = set()
    for seed in seeds:
        full_headers.update(deterministic_headers(header_count, seed=int(seed)))
    overlap = len(calibration_headers & full_headers)
    if overlap:
        raise RuntimeError('calibration/full fixture overlap detected')

    runs = []
    policies = []
    for i, seed in enumerate(seeds):
        focused = i == 0
        run = run_training_1000_v4(
            iterations=iterations,
            header_count=header_count,
            nonces_per_header=nonces_per_header,
            thresholds=thresholds,
            population_size=64,
            offspring_per_iteration=8,
            max_predicates=3,
            seed=int(seed),
            split_seed=f'bitmyner-marker-v0.7-island-{i}',
            checkpoint_every=max(1, iterations // 10),
            cohort_count=4,
            cohort_refresh_every=25,
            family_prior=(prior if focused else None),
            family_prior_strength=(chosen_strength if focused else 0.0),
            final_evaluate=True,
        )
        runs.append(run)
        policies.append({
            'island': i,
            'seed': int(seed),
            'mode': 'CALIBRATED_SOFT_PRIOR' if focused else 'UNRESTRICTED_ESCAPE',
            'family_prior_strength': chosen_strength if focused else 0.0,
            'family_prior': prior if focused else {},
        })

    cross = summarize_multi_seed(runs, minimum_seed_support=minimum_seed_support)
    dup = [float(r['search']['duplicate_proposal_fraction']) for r in runs]
    robust = [float(r['checkpoints'][-1]['robust_core']) for r in runs]
    payload = {
        'schema': SCHEMA,
        'status': 'PASS_ADAPTIVE_ISLAND_TRAINING_COMPLETE',
        'iterations_per_island': int(iterations),
        'total_full_iterations': int(iterations) * len(runs),
        'pilot_iterations_per_strength': int(pilot_iterations),
        'calibration': calibration,
        'calibration_full_header_overlap': overlap,
        'policies': policies,
        'runs': runs,
        'cross_island_meta': cross,
        'mean_duplicate_fraction': sum(dup) / len(dup),
        'mean_final_robust_core': sum(robust) / len(robust),
        'external_evaluation_authorized_candidates': cross.get('consensus_exact_genomes', [])[:8],
        'claims': {
            'prior_strength_selected_without_validation_or_holdout': True,
            'full_islands_use_disjoint_fixture_from_calibration': True,
            'meta_consensus_is_not_cryptographic_proof': True,
            'production_policy_modified': False,
            'cryptographic_advantage_assumed': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
