from __future__ import annotations

import hashlib
import json
import statistics
from typing import Sequence

from .gremlin_marker import deterministic_headers
from .marker_meta_v5 import summarize_multi_seed
from .marker_training_v4 import run_training_1000_v4
from .marker_training_v7 import _recurrent_prior

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_8_MULTI_PILOT_ADAPTIVE_PRIOR'
CALIBRATION_SCHEMA = 'BITMYNER_MARKER_PRIOR_CALIBRATION_V0_8_MULTI_PILOT'


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    return statistics.median(abs(float(x) - med) for x in values)


def calibrate_family_prior_v8(
    meta_summary: dict,
    *,
    strengths: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    pilot_iterations: int = 80,
    pilot_seeds: Sequence[int] = (1317, 1417, 1517),
    header_count: int = 60,
    nonces_per_header: int = 96,
    thresholds: Sequence[int] = (4, 6, 8),
    minimum_seed_support: int = 2,
) -> dict:
    prior = _recurrent_prior(meta_summary, minimum_seed_support)
    strengths = tuple(sorted(set(max(0.0, float(s)) for s in strengths)))
    pilot_seeds = tuple(int(s) for s in pilot_seeds)
    if not strengths:
        raise ValueError('at least one prior strength is required')
    if len(pilot_seeds) < 3:
        raise ValueError('at least three pilot seeds are required')

    candidates = []
    for strength in strengths:
        runs = []
        for seed in pilot_seeds:
            run = run_training_1000_v4(
                iterations=pilot_iterations,
                header_count=header_count,
                nonces_per_header=nonces_per_header,
                thresholds=thresholds,
                population_size=48,
                offspring_per_iteration=6,
                max_predicates=3,
                seed=seed,
                split_seed=f'bitmyner-marker-v0.8-pilot-{seed}',
                checkpoint_every=max(1, pilot_iterations // 4),
                cohort_count=4,
                cohort_refresh_every=max(5, pilot_iterations // 8),
                family_prior=prior,
                family_prior_strength=strength,
                final_evaluate=False,
            )
            c = run['checkpoints'][-1]
            runs.append({
                'seed': seed,
                'robust_core': float(c['robust_core']),
                'duplicate_fraction': float(run['search']['duplicate_proposal_fraction']),
                'instability': float(c['instability']),
                'negative_penalty': float(c['negative_penalty']),
                'receipt_blake2b256': run['receipt_blake2b256'],
                'post_training_evaluation_performed': run['claims']['post_training_evaluation_performed'],
            })
        robust = [x['robust_core'] for x in runs]
        duplicate = [x['duplicate_fraction'] for x in runs]
        instability = [x['instability'] for x in runs]
        negative = [x['negative_penalty'] for x in runs]
        median_robust = float(statistics.median(robust))
        robust_mad = float(_mad(robust))
        mean_duplicate = float(statistics.mean(duplicate))
        mean_instability = float(statistics.mean(instability))
        mean_negative = float(statistics.mean(negative))
        objective = (
            median_robust
            - 0.35 * mean_duplicate
            - 0.10 * robust_mad
            - 0.03 * mean_instability
            - 0.01 * mean_negative
        )
        candidates.append({
            'strength': strength,
            'objective': objective,
            'median_robust_core': median_robust,
            'robust_core_mad': robust_mad,
            'mean_duplicate_fraction': mean_duplicate,
            'mean_instability': mean_instability,
            'mean_negative_penalty': mean_negative,
            'pilot_runs': runs,
        })

    chosen = max(
        candidates,
        key=lambda r: (
            r['objective'],
            r['median_robust_core'],
            -r['mean_duplicate_fraction'],
            -r['strength'],
        ),
    )
    payload = {
        'schema': CALIBRATION_SCHEMA,
        'status': 'PASS_MULTI_PILOT_CALIBRATION_COMPLETE',
        'training_only_prior': prior,
        'minimum_seed_support': int(minimum_seed_support),
        'pilot_iterations': int(pilot_iterations),
        'pilot_seeds': list(pilot_seeds),
        'fixture': {
            'header_count_per_seed': int(header_count),
            'nonces_per_header': int(nonces_per_header),
            'thresholds': [int(k) for k in thresholds],
        },
        'objective': 'MEDIAN_ROBUST_MINUS_DUPLICATE_MINUS_ROBUST_MAD_MINUS_INSTABILITY_MINUS_NEGATIVE_MEMORY',
        'candidates': candidates,
        'chosen_strength': float(chosen['strength']),
        'chosen_objective': float(chosen['objective']),
        'claims': {
            'validation_opened_during_calibration': False,
            'holdout_opened_during_calibration': False,
            'three_or_more_independent_pilot_fixtures': True,
            'production_policy_modified': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload


def run_multi_pilot_island_training_v8(
    meta_summary: dict,
    *,
    iterations: int = 1000,
    full_seeds: Sequence[int] = (2222, 2323, 2424),
    header_count: int = 100,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4, 6, 8),
    minimum_seed_support: int = 2,
    strengths: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    pilot_iterations: int = 80,
    pilot_seeds: Sequence[int] = (1317, 1417, 1517),
) -> dict:
    full_seeds = tuple(int(s) for s in full_seeds)
    if len(full_seeds) < 3:
        raise ValueError('at least three full-training island seeds required')
    prior = _recurrent_prior(meta_summary, minimum_seed_support)
    calibration = calibrate_family_prior_v8(
        meta_summary,
        strengths=strengths,
        pilot_iterations=pilot_iterations,
        pilot_seeds=pilot_seeds,
        header_count=min(60, header_count),
        nonces_per_header=min(96, nonces_per_header),
        thresholds=thresholds,
        minimum_seed_support=minimum_seed_support,
    )
    chosen_strength = float(calibration['chosen_strength'])

    pilot_headers = set()
    for seed in pilot_seeds:
        pilot_headers.update(deterministic_headers(min(60, header_count), seed=int(seed)))
    full_headers = set()
    for seed in full_seeds:
        full_headers.update(deterministic_headers(header_count, seed=int(seed)))
    overlap = len(pilot_headers & full_headers)
    if overlap:
        raise RuntimeError('pilot/full fixture overlap detected')

    runs = []
    policies = []
    for i, seed in enumerate(full_seeds):
        focused = i == 0 and chosen_strength > 0.0
        run = run_training_1000_v4(
            iterations=iterations,
            header_count=header_count,
            nonces_per_header=nonces_per_header,
            thresholds=thresholds,
            population_size=64,
            offspring_per_iteration=8,
            max_predicates=3,
            seed=seed,
            split_seed=f'bitmyner-marker-v0.8-full-{i}',
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
            'seed': seed,
            'mode': 'MULTI_PILOT_CALIBRATED_SOFT_PRIOR' if focused else 'UNRESTRICTED_ESCAPE',
            'family_prior_strength': chosen_strength if focused else 0.0,
            'family_prior': prior if focused else {},
        })

    cross = summarize_multi_seed(runs, minimum_seed_support=minimum_seed_support)
    dup = [float(r['search']['duplicate_proposal_fraction']) for r in runs]
    robust = [float(r['checkpoints'][-1]['robust_core']) for r in runs]
    payload = {
        'schema': SCHEMA,
        'status': 'PASS_MULTI_PILOT_ISLAND_TRAINING_COMPLETE',
        'iterations_per_island': int(iterations),
        'total_full_iterations': int(iterations) * len(runs),
        'calibration': calibration,
        'pilot_full_header_overlap': overlap,
        'policies': policies,
        'runs': runs,
        'cross_island_meta': cross,
        'mean_duplicate_fraction': float(statistics.mean(dup)),
        'mean_final_robust_core': float(statistics.mean(robust)),
        'external_evaluation_authorized_candidates': cross.get('consensus_exact_genomes', [])[:8],
        'claims': {
            'prior_strength_selected_from_training_only_multi_seed_pilots': True,
            'pilot_and_full_fixtures_disjoint': True,
            'two_escape_islands_preserve_global_exploration': True,
            'meta_consensus_is_not_cryptographic_proof': True,
            'production_policy_modified': False,
            'cryptographic_advantage_assumed': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
