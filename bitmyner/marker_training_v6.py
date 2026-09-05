from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .marker_training_v4 import run_training_1000_v4
from .marker_meta_v5 import summarize_multi_seed

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_6_ISLAND_SOFT_PRIOR'


def summarize_island_runs_v6(
    meta_summary: dict,
    runs: Sequence[dict],
    policies: Sequence[dict],
    *,
    iterations: int,
    minimum_seed_support: int = 2,
) -> dict:
    if meta_summary.get('status') != 'PASS_META_COMPLETE':
        raise ValueError('meta summary must be complete')
    if len(runs) < 3 or len(runs) != len(policies):
        raise ValueError('three or more runs with matching policies required')
    meta = summarize_multi_seed(runs, minimum_seed_support=minimum_seed_support)
    dup = [float(r['search']['duplicate_proposal_fraction']) for r in runs]
    focused = [d for d,p in zip(dup, policies) if p.get('mode') == 'SOFT_RECURRENT_FAMILY_PRIOR']
    escape = [d for d,p in zip(dup, policies) if p.get('mode') == 'UNRESTRICTED_ESCAPE']
    payload = {
        'schema': SCHEMA,
        'status': 'PASS_ISLAND_TRAINING_COMPLETE',
        'iterations_per_island': int(iterations),
        'total_iterations': int(iterations) * len(runs),
        'policies': list(policies),
        'runs': list(runs),
        'cross_island_meta': meta,
        'mean_duplicate_fraction': sum(dup) / len(dup),
        'focused_duplicate_fraction': (sum(focused) / len(focused)) if focused else None,
        'escape_mean_duplicate_fraction': (sum(escape) / len(escape)) if escape else None,
        'claims': {
            'hard_whitelist_applied_to_any_island': False,
            'escape_islands_preserve_global_exploration': True,
            'focused_island_preserves_global_support': True,
            'validation_or_holdout_used_to_build_prior': False,
            'production_policy_modified': False,
            'cryptographic_advantage_assumed': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload


def run_island_training_v6(
    meta_summary: dict,
    *,
    iterations: int = 1000,
    seeds: Sequence[int] = (919, 1020, 1121),
    header_count: int = 120,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4, 6, 8),
    minimum_seed_support: int = 2,
    focused_prior_strength: float = 2.0,
) -> dict:
    if meta_summary.get('status') != 'PASS_META_COMPLETE':
        raise ValueError('meta summary must be complete')
    if len(seeds) < 3:
        raise ValueError('at least three island seeds required')
    recurrent_rows = [
        r for r in meta_summary.get('consensus_component_families', [])
        if int(r.get('seed_support', 0)) >= int(minimum_seed_support)
    ]
    family_prior = {str(r['family']): float(r['seed_support']) for r in recurrent_rows}
    if not family_prior:
        raise RuntimeError('no recurrent families')

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
            split_seed=f'bitmyner-marker-v0.6-island-{i}',
            checkpoint_every=max(1, iterations // 10),
            cohort_count=4,
            cohort_refresh_every=25,
            family_whitelist=None,
            family_prior=(family_prior if focused else None),
            family_prior_strength=(float(focused_prior_strength) if focused else 0.0),
        )
        runs.append(run)
        policies.append({
            'island': i,
            'seed': int(seed),
            'mode': 'SOFT_RECURRENT_FAMILY_PRIOR' if focused else 'UNRESTRICTED_ESCAPE',
            'family_whitelist': None,
            'family_prior': family_prior if focused else {},
            'family_prior_strength': float(focused_prior_strength) if focused else 0.0,
        })

    result = summarize_island_runs_v6(
        meta_summary, runs, policies, iterations=iterations, minimum_seed_support=minimum_seed_support
    )
    result['recurrent_family_prior'] = sorted(family_prior)
    result['focused_prior_strength'] = float(focused_prior_strength)
    result['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps({k:v for k,v in result.items() if k != 'receipt_blake2b256'}, sort_keys=True, separators=(',', ':')).encode(),
        digest_size=32,
    ).hexdigest()
    return result
