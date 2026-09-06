from __future__ import annotations

import hashlib
import json
import random
import statistics
from typing import Sequence

from .gremlin_marker import deterministic_headers, collect_samples
from .marker_training_v11 import run_nested_crossfit_training_v11
from .structural_marker import StructuralCompiledCorpus, StructuralPredicate

SCHEMA = 'BITMYNER_MARKER_NULL_CALIBRATION_V0_12'


def _bit_indices(mask: int) -> list[int]:
    out: list[int] = []
    m = int(mask)
    while m:
        lsb = m & -m
        out.append(lsb.bit_length() - 1)
        m ^= lsb
    return out


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError('q must be in [0,1]')
    s = sorted(float(x) for x in values)
    if len(s) == 1:
        return s[0]
    pos = float(q) * (len(s) - 1)
    lo = int(pos)
    hi = min(len(s) - 1, lo + 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _longest_true(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def confirmation_partition_v12(
    corpus: StructuralCompiledCorpus,
    *,
    split_seed: str,
    discovery_fraction: float,
) -> tuple[tuple[str, ...], int]:
    """Reproduce v0.4/v0.11's nested TRAIN discovery/confirmation split."""
    if not 0.0 < float(discovery_fraction) < 1.0:
        raise ValueError('v0.12 requires a strict discovery/confirmation split')
    train_headers = sorted(corpus.header_masks['train'])
    if len(train_headers) < 10:
        raise RuntimeError('insufficient train headers for null calibration')
    keyed = sorted(
        (
            hashlib.blake2b(
                f'{split_seed}|outer-train|{header_id}'.encode(), digest_size=8
            ).digest(),
            header_id,
        )
        for header_id in train_headers
    )
    ordered = [header_id for _, header_id in keyed]
    discovery_count = max(
        6,
        min(
            len(ordered) - 4,
            int(round(len(ordered) * float(discovery_fraction))),
        ),
    )
    confirmation_headers = tuple(ordered[discovery_count:])
    confirmation_mask = 0
    for header_id in confirmation_headers:
        confirmation_mask |= corpus.header_masks['train'][header_id]
    if not confirmation_headers or confirmation_mask == 0:
        raise RuntimeError('empty confirmation surface')
    return confirmation_headers, confirmation_mask


def permute_confirmation_event_masks_v12(
    corpus: StructuralCompiledCorpus,
    *,
    confirmation_headers: Sequence[str],
    thresholds: Sequence[int],
    seed: int,
) -> dict[int, int]:
    """Permute observed low-hash labels within each confirmation header.

    Exact leading-zero-bit labels are shuffled as a unit, so every header keeps its
    event count at every threshold and threshold nesting remains exact. Predicate
    features and nonce/header identities are untouched.
    """
    thresholds = tuple(sorted(set(int(k) for k in thresholds)))
    if not thresholds:
        raise ValueError('at least one threshold is required')
    for k in thresholds:
        if k not in corpus.event_masks['train']:
            raise ValueError(f'threshold {k} absent from corpus')

    rows = corpus.samples_by_split['train']
    rng = random.Random(int(seed))
    out = {k: 0 for k in thresholds}

    for header_id in sorted(set(str(h) for h in confirmation_headers)):
        if header_id not in corpus.header_masks['train']:
            raise KeyError(header_id)
        indices = _bit_indices(corpus.header_masks['train'][header_id])
        labels = [int(rows[i].trace.leading_zero_bits) for i in indices]
        rng.shuffle(labels)
        for i, lz in zip(indices, labels):
            bit = 1 << i
            for k in thresholds:
                if lz >= k:
                    out[k] |= bit

    # Fail closed if a programming change ever breaks nesting or per-header counts.
    for lo, hi in zip(thresholds, thresholds[1:]):
        if out[hi] & ~out[lo]:
            raise RuntimeError('permuted threshold nesting broken')
    for header_id in sorted(set(str(h) for h in confirmation_headers)):
        hmask = corpus.header_masks['train'][header_id]
        for k in thresholds:
            expected = (corpus.event_masks['train'][k] & hmask).bit_count()
            observed = (out[k] & hmask).bit_count()
            if observed != expected:
                raise RuntimeError('per-header event count changed during permutation')
    return out


def _catalog_to_predicates(run: dict) -> tuple[StructuralPredicate, ...]:
    predicates: list[StructuralPredicate] = []
    for row in run.get('predicate_catalog', []):
        predicates.append(
            StructuralPredicate(
                stage=int(row['stage']),
                round=int(row['round']),
                op=str(row['op']),
                left=str(row['left']),
                bit=int(row['bit']),
                value=int(row['value']),
                right=None if row.get('right') is None else str(row['right']),
            )
        )
    if not predicates:
        raise RuntimeError('run does not contain a predicate catalog')
    return tuple(predicates)


def _rebuild_run_corpus_v12(run: dict) -> tuple[StructuralCompiledCorpus, tuple[str, ...], int]:
    fixture = run['fixture']
    policy = run['training_policy']
    seed = int(fixture['seed'])
    split_seed = str(fixture['split_seed'])
    thresholds = tuple(int(k) for k in fixture['thresholds'])
    headers = deterministic_headers(int(fixture['header_count']), seed=seed)
    samples = collect_samples(
        headers,
        nonces_per_header=int(fixture['nonces_per_header']),
        event_k=min(thresholds),
        nonce_seed=seed ^ 0x5EED,
    )
    corpus = StructuralCompiledCorpus(
        samples,
        _catalog_to_predicates(run),
        thresholds,
        split_seed,
    )
    confirmation_headers, confirmation_mask = confirmation_partition_v12(
        corpus,
        split_seed=split_seed,
        discovery_fraction=float(policy['train_discovery_fraction']),
    )
    expected_headers = int(policy['train_confirmation_headers'])
    if len(confirmation_headers) != expected_headers:
        raise RuntimeError(
            f'confirmation partition mismatch: rebuilt={len(confirmation_headers)} expected={expected_headers}'
        )
    return corpus, confirmation_headers, confirmation_mask


def _evaluate_confirmation_surface(
    corpus: StructuralCompiledCorpus,
    *,
    candidates: Sequence[dict],
    confirmation_mask: int,
    thresholds: Sequence[int],
    familywise_alpha: float,
) -> dict:
    thresholds = tuple(int(k) for k in thresholds)
    alpha = float(familywise_alpha) / max(1, len(candidates) * len(thresholds))
    rows: list[dict] = []
    max_gamma_lcb = 0.0
    any_pass = False
    max_longest = 0

    for candidate in candidates:
        genome = tuple(int(x) for x in candidate['genome'])
        flags: list[bool] = []
        ladder: list[dict] = []
        for k in thresholds:
            metric = corpus.metric(
                genome,
                'train',
                int(k),
                sample_mask=confirmation_mask,
                alpha=alpha,
            )
            qualified = bool(
                metric['marker_n'] >= 32
                and metric['event_n'] > 0
                and metric['gamma_lcb'] > 1.0
            )
            flags.append(qualified)
            max_gamma_lcb = max(max_gamma_lcb, float(metric['gamma_lcb']))
            ladder.append({'k': int(k), 'metric': metric, 'qualified': qualified})
        longest = _longest_true(flags)
        passed = longest >= min(2, len(thresholds))
        max_longest = max(max_longest, longest)
        any_pass = any_pass or passed
        rows.append({'genome': list(genome), 'ladder': ladder, 'longest_qualified_run': longest, 'pass': passed})

    return {
        'alpha_per_candidate_threshold': alpha,
        'candidates': rows,
        'max_gamma_lcb': max_gamma_lcb,
        'max_longest_qualified_run': max_longest,
        'any_pass': any_pass,
    }


def calibrate_confirmation_run_v12(
    run: dict,
    *,
    permutations: int = 1000,
    calibration_seed: int = 12012,
    familywise_alpha: float = 0.05,
) -> dict:
    """Null-calibrate the already-frozen nested confirmation gate.

    Candidate genomes remain frozen. Only confirmation low-hash labels are permuted,
    within header, after discovery is complete. This measures the conditional
    familywise false-pass behaviour of the confirmation gate without using validation
    or holdout and without changing any mining policy.
    """
    if permutations < 1:
        raise ValueError('permutations must be positive')
    candidates = tuple(run.get('final_top8', []))
    if not candidates:
        raise RuntimeError('run has no frozen final candidates')
    thresholds = tuple(int(k) for k in run['fixture']['thresholds'])
    corpus, confirmation_headers, confirmation_mask = _rebuild_run_corpus_v12(run)

    observed = _evaluate_confirmation_surface(
        corpus,
        candidates=candidates,
        confirmation_mask=confirmation_mask,
        thresholds=thresholds,
        familywise_alpha=familywise_alpha,
    )
    original_masks = corpus.event_masks['train']
    null_max_gamma_lcb: list[float] = []
    null_any_pass: list[bool] = []
    null_max_longest: list[int] = []

    try:
        for permutation_index in range(int(permutations)):
            seed_material = (
                f'{int(calibration_seed)}|{run["fixture"]["seed"]}|{permutation_index}'
            ).encode()
            trial_seed = int.from_bytes(
                hashlib.blake2b(seed_material, digest_size=8).digest(), 'little'
            )
            corpus.event_masks['train'] = permute_confirmation_event_masks_v12(
                corpus,
                confirmation_headers=confirmation_headers,
                thresholds=thresholds,
                seed=trial_seed,
            )
            trial = _evaluate_confirmation_surface(
                corpus,
                candidates=candidates,
                confirmation_mask=confirmation_mask,
                thresholds=thresholds,
                familywise_alpha=familywise_alpha,
            )
            null_max_gamma_lcb.append(float(trial['max_gamma_lcb']))
            null_any_pass.append(bool(trial['any_pass']))
            null_max_longest.append(int(trial['max_longest_qualified_run']))
    finally:
        corpus.event_masks['train'] = original_masks

    observed_max = float(observed['max_gamma_lcb'])
    exceed = sum(x >= observed_max for x in null_max_gamma_lcb)
    empirical_p = (exceed + 1.0) / (len(null_max_gamma_lcb) + 1.0)
    false_pass_rate = sum(null_any_pass) / len(null_any_pass)
    q95 = _quantile(null_max_gamma_lcb, 0.95)

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_CONFIRMATION_NULL_CALIBRATION_COMPLETE',
        'run_seed': int(run['fixture']['seed']),
        'thresholds': list(thresholds),
        'candidate_count': len(candidates),
        'confirmation_headers': len(confirmation_headers),
        'confirmation_samples': confirmation_mask.bit_count(),
        'permutations': int(permutations),
        'calibration_seed': int(calibration_seed),
        'familywise_alpha': float(familywise_alpha),
        'label_permutation_scope': 'WITHIN_CONFIRMATION_HEADER',
        'threshold_nesting_preserved': True,
        'per_header_event_counts_preserved': True,
        'observed': {
            'max_gamma_lcb': observed_max,
            'max_longest_qualified_run': int(observed['max_longest_qualified_run']),
            'any_pass': bool(observed['any_pass']),
        },
        'null': {
            'mean_max_gamma_lcb': float(statistics.mean(null_max_gamma_lcb)),
            'median_max_gamma_lcb': float(statistics.median(null_max_gamma_lcb)),
            'q90_max_gamma_lcb': _quantile(null_max_gamma_lcb, 0.90),
            'q95_max_gamma_lcb': q95,
            'q99_max_gamma_lcb': _quantile(null_max_gamma_lcb, 0.99),
            'max_max_gamma_lcb': max(null_max_gamma_lcb),
            'mean_max_longest_qualified_run': float(statistics.mean(null_max_longest)),
            'familywise_false_pass_count': sum(null_any_pass),
        },
        'familywise_false_pass_rate': false_pass_rate,
        'empirical_p_max_gamma_lcb': empirical_p,
        'calibrated_gamma_lcb_threshold_95': max(1.0, q95),
        'production_policy_modified': False,
        'cryptographic_advantage_assumed': False,
        'validation_or_holdout_used': False,
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload


def run_null_calibrated_crossfit_v12(
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
    requested_max_cohorts: int = 4,
    min_expected_events_per_cohort: float = 8.0,
    permutations_per_island: int = 1000,
    calibration_seed: int = 12012,
) -> dict:
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
        requested_max_cohorts=requested_max_cohorts,
        min_expected_events_per_cohort=min_expected_events_per_cohort,
    )
    calibrations = [
        calibrate_confirmation_run_v12(
            run,
            permutations=permutations_per_island,
            calibration_seed=int(calibration_seed) + i * 100003,
        )
        for i, run in enumerate(base['runs'])
    ]
    false_rates = [float(c['familywise_false_pass_rate']) for c in calibrations]
    p_values = [float(c['empirical_p_max_gamma_lcb']) for c in calibrations]
    thresholds95 = [float(c['calibrated_gamma_lcb_threshold_95']) for c in calibrations]

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_NULL_CALIBRATED_CROSSFIT_V12_COMPLETE',
        'base_v11_receipt_blake2b256': base['receipt_blake2b256'],
        'iterations_per_island': int(iterations),
        'total_training_iterations': int(iterations) * len(base['runs']),
        'seeds': [int(x) for x in seeds],
        'permutations_per_island': int(permutations_per_island),
        'total_null_permutations': int(permutations_per_island) * len(calibrations),
        'base_confirmation_pass_counts': list(base['confirmation_pass_counts']),
        'base_external_evaluation_authorized_candidates': list(base['external_evaluation_authorized_candidates']),
        'calibrations': calibrations,
        'mean_familywise_false_pass_rate': float(statistics.mean(false_rates)),
        'max_familywise_false_pass_rate': max(false_rates),
        'empirical_p_max_gamma_lcb_by_island': p_values,
        'recommended_gamma_lcb_gate': max(1.0, max(thresholds95)),
        'gate_action': 'KEEP_OR_RAISE_CONFIRMATION_GATE_ONLY',
        'claims': {
            'candidate_genomes_frozen_before_null_permutation': True,
            'null_labels_permuted_within_confirmation_header_only': True,
            'per_header_event_counts_preserved': True,
            'threshold_nesting_preserved': True,
            'validation_and_holdout_unopened_for_null_calibration': True,
            'cryptographic_advantage_assumed': False,
            'production_policy_modified': False,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
