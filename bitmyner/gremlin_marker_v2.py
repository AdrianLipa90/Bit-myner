from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .dncrypt_marker import MarkerGenome, raven_single_bit_population
from .dncrypt_marker_v2 import raven_pair_population
from .gremlin_marker import deterministic_headers, collect_samples
from .marker_cost import gamma_for_genome, gamma_lcb_from_lift
from .marker_eval import TraceSample, raven_rank_train, evaluate_genome, samples_for_split
from .marker_graph import HashEvidenceGraph, append_trace_evidence
from .mole_ring import MoleEventRing
from .sha256_trace import sha256d_trace

SCHEMA = 'BITMYNER_MARKER_GRAPH_EXPERIMENT_V0_2'


def _unique_genomes(genomes: Sequence[MarkerGenome]) -> tuple[MarkerGenome, ...]:
    out: dict[str, MarkerGenome] = {}
    for genome in genomes:
        out[genome.genome_id] = genome
    return tuple(out[key] for key in sorted(out))


def _samples_at_threshold(samples: Sequence[TraceSample], k: int) -> list[TraceSample]:
    return [TraceSample(s.header_id, s.trace, k) for s in samples]


def _longest_true_run(flags: Sequence[bool]) -> int:
    best = cur = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _cost_payload(genome: MarkerGenome, samples: Sequence[TraceSample], report: dict | None) -> dict | None:
    if not samples or report is None:
        return None
    gamma = gamma_for_genome(genome, samples)
    payload = gamma.to_dict()
    payload['gamma_lcb_from_lift'] = gamma_lcb_from_lift(gamma, lift_lcb=float(report['lift_lcb']))
    payload['cost_model'] = 'EXACT_SHA256_ROUND_UNITS_ONLY'
    return payload


def _build_replay_graph(ring: MoleEventRing, *, event_k: int, max_replays: int = 24) -> tuple[HashEvidenceGraph, int]:
    graph = HashEvidenceGraph()
    replayed = 0
    for row in ring.replay_rows()[:max_replays]:
        header = bytes.fromhex(row.header76_hex)
        trace = sha256d_trace(header, row.nonce)
        if trace.digest.hex() != row.digest_hex:
            raise RuntimeError('MANTIS_REPLAY_DIGEST_MISMATCH')
        append_trace_evidence(
            graph,
            header_id=row.header_id,
            header76=header,
            nonce=row.nonce,
            digest=trace.digest,
            leading_zero_bits=trace.leading_zero_bits,
            first_digest=trace.first_digest,
            round_records=trace.stage1 + trace.stage2,
            event_k=event_k,
        )
        replayed += 1
    return graph, replayed


def run_experiment_v2(
    *,
    header_count: int = 21,
    nonces_per_header: int = 512,
    thresholds: Sequence[int] = (6, 8, 10, 12),
    seed_top_k: int = 16,
    pair_limit: int = 64,
    final_top_k: int = 8,
    seed: int = 616,
    split_seed: str = 'bitmyner-marker-v0.2',
    min_marker_n: int = 16,
    delta: float = 0.0,
) -> dict:
    levels = tuple(sorted(set(int(k) for k in thresholds)))
    if not levels or any(k < 1 or k > 32 for k in levels):
        raise ValueError('v0.2 bounded thresholds must be unique integers in [1,32]')
    if seed_top_k < 2 or pair_limit < 1 or final_top_k < 1:
        raise ValueError('candidate limits must be positive; seed_top_k must be >=2')

    headers = deterministic_headers(header_count, seed=seed)
    base_k = levels[0]
    base_samples = collect_samples(
        headers,
        nonces_per_header=nonces_per_header,
        event_k=base_k,
    )

    # MOLE sees only completed digest tuples. It does not invoke the tracer in
    # the online capture surface. This bounded benchmark already owns traces for
    # falsification, so their digests are fed into the same capture API.
    ring = MoleEventRing(
        thresholds=levels,
        event_capacity_per_level=64,
        control_capacity=64,
        control_modulus=257,
    )
    for sample in base_samples:
        ring.observe(sample.trace.header76, sample.trace.nonce, sample.trace.digest)
    ring_snapshot = ring.snapshot()

    single_candidates = raven_single_bit_population(
        stages=(1, 2),
        rounds=(0, 7, 15, 31, 47, 63),
        sources=('a', 'e', 't1', 'w'),
        bits=(0, 7, 15, 23, 31),
    )
    seed_ranked = raven_rank_train(
        single_candidates,
        base_samples,
        seed=split_seed,
        min_marker_n=min_marker_n,
        top_k=seed_top_k,
    )
    seed_genomes = tuple(genome for genome, _ in seed_ranked)
    composite_candidates = raven_pair_population(seed_genomes, max_pairs=pair_limit, mode='all')
    candidate_pool = _unique_genomes(seed_genomes + composite_candidates)
    selected = raven_rank_train(
        candidate_pool,
        base_samples,
        seed=split_seed,
        min_marker_n=min_marker_n,
        top_k=final_top_k,
    )

    alpha_family = 0.05
    test_count = max(1, len(selected) * len(levels) * 2)  # validation + holdout
    alpha_per_test = alpha_family / test_count
    required_persistent_levels = min(2, len(levels))

    selected_payload = []
    promoted = []
    for genome, training_rank_report in selected:
        ladder = []
        qualified_flags: list[bool] = []
        for k in levels:
            samples_k = _samples_at_threshold(base_samples, k)
            evaluation = evaluate_genome(
                genome,
                samples_k,
                seed=split_seed,
                alpha=alpha_per_test,
            )
            reports = evaluation['reports']
            cost_by_split = {}
            for split in ('train', 'validation', 'holdout'):
                split_samples = samples_for_split(samples_k, split, seed=split_seed)
                cost_by_split[split] = _cost_payload(genome, split_samples, reports.get(split))

            validation = reports.get('validation')
            holdout = reports.get('holdout')
            validation_cost = cost_by_split.get('validation')
            holdout_cost = cost_by_split.get('holdout')
            qualified = bool(
                validation
                and holdout
                and validation_cost
                and holdout_cost
                and validation['marker_n'] >= min_marker_n
                and holdout['marker_n'] >= min_marker_n
                and validation['event_n'] > 0
                and holdout['event_n'] > 0
                and validation_cost['gamma_lcb_from_lift'] > 1.0 + delta
                and holdout_cost['gamma_lcb_from_lift'] > 1.0 + delta
            )
            qualified_flags.append(qualified)
            ladder.append({
                'k': k,
                'evaluation': evaluation,
                'cost_gain': cost_by_split,
                'qualified_for_persistence': qualified,
            })

        longest_run = _longest_true_run(qualified_flags)
        verdict = {
            'promote': longest_run >= required_persistent_levels,
            'required_consecutive_levels': required_persistent_levels,
            'longest_qualified_run': longest_run,
            'rule': 'VALIDATION_AND_HOLDOUT_GAMMA_LCB>1+delta_ON_CONSECUTIVE_LEVELS',
            'delta': delta,
        }
        entry = {
            'genome': genome.to_dict(),
            'training_rank_report_base_threshold': training_rank_report,
            'ladder': ladder,
            'verdict': verdict,
        }
        selected_payload.append(entry)
        if verdict['promote']:
            promoted.append(entry)

    graph, replayed = _build_replay_graph(ring, event_k=base_k)
    compact_ring = {
        'schema': ring_snapshot['schema'],
        'thresholds': ring_snapshot['thresholds'],
        'observed': ring_snapshot['observed'],
        'hot_path_round_tracing': ring_snapshot['hot_path_round_tracing'],
        'event_counts': ring_snapshot['event_counts'],
        'control_count': ring_snapshot['control_count'],
        'replay_count': ring_snapshot['replay_count'],
        'mantis_replayed_into_graph': replayed,
    }

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_EXPERIMENT_COMPLETE',
        'thresholds': list(levels),
        'fixture': {
            'synthetic_header_fixture': True,
            'header_count': header_count,
            'nonces_per_header': nonces_per_header,
            'sample_count': len(base_samples),
            'seed': seed,
        },
        'mole_ring': compact_ring,
        'candidate_generation': {
            'single_candidates': len(single_candidates),
            'training_seed_candidates': len(seed_genomes),
            'composite_candidates': len(composite_candidates),
            'final_candidate_pool': len(candidate_pool),
        },
        'selection': {
            'final_top_k_requested': final_top_k,
            'selected_count': len(selected_payload),
            'ranking_split': 'TRAIN_ONLY_AT_BASE_THRESHOLD',
            'holdout_used_for_ranking': False,
            'validation_used_for_ranking': False,
            'split_seed': split_seed,
        },
        'multiple_testing': {
            'familywise_alpha': alpha_family,
            'test_count_validation_plus_holdout': test_count,
            'per_test_alpha': alpha_per_test,
            'correction': 'BONFERRONI_SELECTED_CANDIDATES_X_LEVELS_X_2',
        },
        'selected': selected_payload,
        'promoted_count': len(promoted),
        'promotion_semantics': {
            'cryptographic_advantage_assumed': False,
            'production_policy_changed': False,
            'required_consecutive_levels': required_persistent_levels,
            'promotion_requires_cost_adjusted_validation_and_holdout': True,
        },
        'tau_cost': {
            'round_cost_model': 'EXACT_SHA256D_128_ROUND_UNITS',
            'runtime_tau_measurement_available': False,
            'tau_adjustment': 'IDENTITY_UNTIL_MEASURED_RUNTIME_LATENCY_RECEIPT',
            'laboratory_time_modified': False,
            'physical_time_claim': False,
        },
        'graph': {
            'schema': graph.schema,
            'graph_id': graph.graph_id,
            'node_count': len(graph.nodes),
            'edge_count': len(graph.edges),
            'source': 'MOLE_RING_TO_MANTIS_REPLAY',
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'),
        digest_size=32,
    ).hexdigest()
    return payload
