from __future__ import annotations

import hashlib
import json
import statistics
from typing import Sequence

from .marker_meta_v5 import summarize_multi_seed
from .marker_training_v4 import run_training_1000_v4

SCHEMA = 'BITMYNER_MARKER_TRAINING_V0_10_NESTED_CROSSFIT_ISLANDS'


def _confirmation_filtered_run(run: dict) -> dict:
    out = dict(run)
    out['final_top8'] = [row for row in run.get('final_top8', []) if row.get('confirmation_pass', False)]
    return out


def run_nested_crossfit_training_v10(
    *,
    iterations: int = 1000,
    seeds: Sequence[int] = (3030, 3131, 3232),
    header_count: int = 120,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4, 6, 8),
    population_size: int = 64,
    offspring_per_iteration: int = 8,
    max_predicates: int = 3,
    train_discovery_fraction: float = 0.72,
    minimum_seed_support: int = 2,
) -> dict:
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) < 3:
        raise ValueError('at least three island seeds are required')
    if not 0.0 < float(train_discovery_fraction) < 1.0:
        raise ValueError('v0.10 requires a strict discovery/confirmation split')

    runs: list[dict] = []
    confirmation_counts: list[int] = []
    validation_open_counts: list[int] = []
    for i, seed in enumerate(seeds):
        run = run_training_1000_v4(
            iterations=iterations,
            header_count=header_count,
            nonces_per_header=nonces_per_header,
            thresholds=thresholds,
            population_size=population_size,
            offspring_per_iteration=offspring_per_iteration,
            max_predicates=max_predicates,
            seed=seed,
            split_seed=f'bitmyner-marker-v0.10-island-{i}',
            checkpoint_every=max(1, iterations // 10),
            cohort_count=4,
            cohort_refresh_every=max(5, min(25, iterations // 8 if iterations >= 8 else 5)),
            family_prior=None,
            family_prior_strength=0.0,
            final_evaluate=True,
            train_discovery_fraction=train_discovery_fraction,
        )
        runs.append(run)
        passed = [row for row in run.get('final_top8', []) if row.get('confirmation_pass', False)]
        opened = [row for row in run.get('final_top8', []) if row.get('ladder')]
        confirmation_counts.append(len(passed))
        validation_open_counts.append(len(opened))

    filtered = [_confirmation_filtered_run(run) for run in runs]
    cross = summarize_multi_seed(filtered, minimum_seed_support=minimum_seed_support)
    duplicate = [float(r['search']['duplicate_proposal_fraction']) for r in runs]
    robust = [float(r['checkpoints'][-1]['robust_core']) for r in runs]
    promoted = [int(r.get('promoted_count', 0)) for r in runs]

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_NESTED_CROSSFIT_TRAINING_COMPLETE',
        'iterations_per_island': int(iterations),
        'total_iterations': int(iterations) * len(runs),
        'seeds': list(seeds),
        'fixture': {
            'header_count_per_island': int(header_count),
            'nonces_per_header': int(nonces_per_header),
            'thresholds': [int(k) for k in thresholds],
            'train_discovery_fraction': float(train_discovery_fraction),
        },
        'runs': runs,
        'confirmation_pass_counts': confirmation_counts,
        'validation_holdout_open_counts': validation_open_counts,
        'mean_duplicate_fraction': float(statistics.mean(duplicate)),
        'mean_final_robust_core': float(statistics.mean(robust)),
        'promoted_counts': promoted,
        'crossfit_meta': cross,
        'external_evaluation_authorized_candidates': cross.get('consensus_exact_genomes', [])[:8],
        'claims': {
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
