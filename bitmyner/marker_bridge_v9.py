from __future__ import annotations

import hashlib
import itertools
import json
from typing import Iterable, Sequence

from .gremlin_marker import collect_samples, deterministic_headers
from .marker_training import _canonical
from .marker_training_v4 import _consistent_genome
from .structural_marker import StructuralCompiledCorpus, StructuralPredicate, nonce_variation_filter, structural_population

SCHEMA = 'BITMYNER_MARKER_BRIDGE_V0_9_CROSS_SEED_RECOMBINATION'


def _score_train(corpus: StructuralCompiledCorpus, genome: tuple[int, ...], thresholds: Sequence[int]) -> tuple[float, list[dict]]:
    rows: list[dict] = []
    qualified: list[float] = []
    for k in thresholds:
        r = corpus.metric(genome, 'train', int(k), alpha=0.05)
        rows.append({'k': int(k), **r})
        if r['event_n'] >= 8 and r['marker_n'] >= 32 and r['marker_event_n'] >= 2:
            qualified.append(float(r['gamma_lcb']))
    return (min(qualified) if qualified else 0.0), rows


def _predicate_row(p: StructuralPredicate) -> dict:
    return {
        'predicate_id': p.predicate_id,
        'family': p.family,
        'stage': p.stage,
        'round': p.round,
        'op': p.op,
        'left': p.left,
        'right': p.right,
        'bit': p.bit,
        'value': p.value,
        'checkpoint_cost': p.checkpoint_cost,
    }


def _family_support(meta_summary: dict, minimum_seed_support: int) -> dict[str, int]:
    if meta_summary.get('status') != 'PASS_META_COMPLETE':
        raise ValueError('meta summary must be PASS_META_COMPLETE')
    out: dict[str, int] = {}
    for row in meta_summary.get('consensus_component_families', []):
        fam = str(row.get('family', ''))
        support = int(row.get('seed_support', 0))
        if fam and support >= minimum_seed_support:
            out[fam] = max(out.get(fam, 0), support)
    return out


def run_bridge_recombination_v9(
    meta_summary: dict,
    *,
    bridge_seed: int = 2626,
    excluded_seeds: Sequence[int] = (1317, 1417, 1517, 2222, 2323, 2424),
    excluded_header_count: int = 100,
    header_count: int = 160,
    nonces_per_header: int = 128,
    thresholds: Sequence[int] = (4, 6, 8),
    top_singles_per_family: int = 4,
    pair_limit: int = 64,
    final_top_k: int = 8,
    minimum_seed_support: int = 2,
    split_seed: str = 'bitmyner-marker-bridge-v0.9',
    familywise_alpha: float = 0.05,
) -> dict:
    if header_count < 12 or nonces_per_header < 8:
        raise ValueError('bridge fixture too small')
    if top_singles_per_family < 1 or pair_limit < 0 or final_top_k < 1:
        raise ValueError('invalid candidate limits')
    thresholds = tuple(sorted(set(int(k) for k in thresholds)))
    if not thresholds:
        raise ValueError('at least one threshold required')

    recurrent = _family_support(meta_summary, minimum_seed_support)
    if not recurrent:
        raise RuntimeError('no recurrent component families')

    bridge_headers = deterministic_headers(header_count, seed=int(bridge_seed))
    excluded_headers: set[bytes] = set()
    for seed in excluded_seeds:
        excluded_headers.update(deterministic_headers(excluded_header_count, seed=int(seed)))
    overlap = len(set(bridge_headers) & excluded_headers)
    if overlap:
        raise RuntimeError('bridge/excluded fixture overlap detected')

    samples = collect_samples(
        bridge_headers,
        nonces_per_header=nonces_per_header,
        event_k=min(thresholds),
        nonce_seed=int(bridge_seed) ^ 0xB9D63E,
    )
    raw_predicates = structural_population()
    filtered, filter_receipt = nonce_variation_filter(samples, raw_predicates, split_seed=split_seed)
    predicates = tuple(p for p in filtered if p.family in recurrent)
    if not predicates:
        raise RuntimeError('no nonce-varying predicates from recurrent families')

    corpus = StructuralCompiledCorpus(samples, predicates, thresholds, split_seed)
    family_to_indices: dict[str, list[int]] = {}
    for i, p in enumerate(predicates):
        family_to_indices.setdefault(p.family, []).append(i)

    singles: list[dict] = []
    selected_single_indices: list[int] = []
    for family in sorted(family_to_indices):
        ranked: list[tuple[float, int, list[dict]]] = []
        for idx in family_to_indices[family]:
            score, rows = _score_train(corpus, (idx,), thresholds)
            ranked.append((score, idx, rows))
        ranked.sort(key=lambda x: (x[0], -predicates[x[1]].checkpoint_cost, predicates[x[1]].predicate_id), reverse=True)
        for score, idx, rows in ranked[:top_singles_per_family]:
            selected_single_indices.append(idx)
            singles.append({
                'predicate_index': idx,
                'predicate': _predicate_row(predicates[idx]),
                'train_score': score,
                'train_levels': rows,
                'family_seed_support': recurrent.get(family, 0),
            })

    pair_rows: list[dict] = []
    unique_pairs: set[tuple[int, ...]] = set()
    for a, b in itertools.combinations(selected_single_indices, 2):
        if predicates[a].family == predicates[b].family:
            continue
        g = _canonical((a, b), 2)
        if g in unique_pairs or not _consistent_genome(g, predicates):
            continue
        unique_pairs.add(g)
        score, rows = _score_train(corpus, g, thresholds)
        pair_rows.append({
            'genome': list(g),
            'predicate_ids': [predicates[i].predicate_id for i in g],
            'family_signature': [predicates[i].family for i in g],
            'train_score': score,
            'train_levels': rows,
        })
    pair_rows.sort(key=lambda x: (x['train_score'], x['predicate_ids']), reverse=True)
    if pair_limit:
        pair_rows = pair_rows[:pair_limit]
    else:
        pair_rows = []

    single_candidates = [
        {
            'genome': [r['predicate_index']],
            'predicate_ids': [r['predicate']['predicate_id']],
            'family_signature': [r['predicate']['family']],
            'train_score': r['train_score'],
            'train_levels': r['train_levels'],
        }
        for r in singles
    ]
    candidate_pool = single_candidates + pair_rows
    candidate_pool.sort(key=lambda x: (x['train_score'], x['predicate_ids']), reverse=True)
    selected = candidate_pool[:final_top_k]

    alpha_per_test = familywise_alpha / max(1, len(selected) * len(thresholds) * 2)
    evaluated: list[dict] = []
    authorized: list[dict] = []
    for row in selected:
        genome = tuple(int(i) for i in row['genome'])
        ladder = []
        flags: list[bool] = []
        for k in thresholds:
            vr = corpus.metric(genome, 'validation', int(k), alpha=alpha_per_test)
            hr = corpus.metric(genome, 'holdout', int(k), alpha=alpha_per_test)
            qualified = bool(
                vr['event_n'] > 0 and hr['event_n'] > 0
                and vr['marker_n'] >= 32 and hr['marker_n'] >= 32
                and vr['gamma_lcb'] > 1.0 and hr['gamma_lcb'] > 1.0
            )
            flags.append(qualified)
            ladder.append({'k': int(k), 'validation': vr, 'holdout': hr, 'qualified': qualified})
        longest = cur = 0
        for flag in flags:
            cur = cur + 1 if flag else 0
            longest = max(longest, cur)
        bridge_pass = longest >= min(2, len(thresholds))
        result = {
            **row,
            'ladder': ladder,
            'longest_qualified_run': longest,
            'bridge_pass': bridge_pass,
        }
        evaluated.append(result)
        if bridge_pass:
            authorized.append({
                'predicate_ids': list(row['predicate_ids']),
                'family_signature': list(row['family_signature']),
                'bridge_train_score': float(row['train_score']),
                'longest_qualified_run': longest,
            })

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_BRIDGE_RECOMBINATION_COMPLETE',
        'fixture': {
            'bridge_seed': int(bridge_seed),
            'excluded_seeds': [int(x) for x in excluded_seeds],
            'excluded_header_count': int(excluded_header_count),
            'header_count': int(header_count),
            'nonces_per_header': int(nonces_per_header),
            'sample_count': len(samples),
            'split_counts': {k: len(v) for k, v in corpus.samples_by_split.items()},
            'thresholds': list(thresholds),
            'header_overlap_with_excluded_fixtures': overlap,
        },
        'causal_filter': filter_receipt,
        'recurrent_families': [
            {'family': family, 'seed_support': support}
            for family, support in sorted(recurrent.items(), key=lambda x: (-x[1], x[0]))
        ],
        'candidate_generation': {
            'recurrent_family_count': len(recurrent),
            'filtered_predicate_count': len(predicates),
            'train_selected_single_count': len(singles),
            'pair_candidate_count': len(pair_rows),
            'candidate_pool_count': len(candidate_pool),
            'final_top_k': len(selected),
        },
        'train_selected_singles': singles,
        'selected_top_k': evaluated,
        'multiple_testing': {
            'familywise_alpha': float(familywise_alpha),
            'per_test_alpha': float(alpha_per_test),
            'correction': 'BONFERRONI_SELECTED_CANDIDATES_X_THRESHOLDS_X_VALIDATION_HOLDOUT',
        },
        'external_evaluation_authorized_candidates': authorized,
        'policy': {
            'ranking_surface': 'BRIDGE_TRAIN_ONLY',
            'validation_used_for_ranking': False,
            'holdout_used_for_ranking': False,
            'bridge_pass_is_not_production_promotion': True,
            'external_preregistered_evaluation_required_after_bridge_pass': True,
        },
        'claims': {
            'cryptographic_advantage_assumed': False,
            'production_policy_modified': False,
            'meta_recurrence_used_only_to_define_family_search_space': True,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(), digest_size=32
    ).hexdigest()
    return payload
