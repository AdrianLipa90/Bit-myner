from __future__ import annotations

import hashlib
import json
import random
from typing import Sequence

from .dncrypt_marker import raven_single_bit_population
from .marker_eval import TraceSample, raven_rank, belzebub_promotion_verdict
from .marker_graph import HashEvidenceGraph, append_trace_evidence, append_marker_association
from .sha256_trace import sha256d_trace

SCHEMA = 'BITMYNER_MARKER_GRAPH_EXPERIMENT_V0_1'
ROLES = {
    'MOLE': 'HASH_AND_EVENT_CAPTURE',
    'MANTIS': 'DETERMINISTIC_SHA256D_ROUND_DECOMPOSITION',
    'RAVEN': 'MOTIF_CANDIDATE_SEARCH',
    'DNCrypt': 'CONTENT_ADDRESSED_MARKER_GENOMES',
    'HOUND': 'HEADER_LEVEL_VALIDATION_AND_HOLDOUT_FALSIFICATION',
    'OWL': 'LIFT_AND_CONFIDENCE_TELEMETRY',
    'BELZEBUB': 'PROMOTION_GATE_ONLY',
}


def _header_id(header76: bytes) -> str:
    return hashlib.blake2b(header76, digest_size=16).hexdigest()


def deterministic_headers(count: int, *, seed: int = 616) -> tuple[bytes, ...]:
    if count < 1:
        raise ValueError('count must be positive')
    rng = random.Random(seed)
    out = []
    for i in range(count):
        raw = bytearray(rng.getrandbits(8) for _ in range(76))
        raw[:4] = (2 + (i % 3)).to_bytes(4, 'little')
        out.append(bytes(raw))
    return tuple(out)


def collect_samples(
    headers: Sequence[bytes],
    *,
    nonces_per_header: int,
    event_k: int,
    nonce_seed: int = 0,
) -> list[TraceSample]:
    if nonces_per_header < 1:
        raise ValueError('nonces_per_header must be positive')
    if not 1 <= event_k <= 32:
        raise ValueError('event_k for v0.1 benchmark must be in [1,32]')
    samples: list[TraceSample] = []
    for hi, header in enumerate(headers):
        hid = _header_id(header)
        start = (nonce_seed + hi * 0x9E3779B1) & 0xFFFFFFFF
        for j in range(nonces_per_header):
            nonce = (start + j) & 0xFFFFFFFF
            samples.append(TraceSample(hid, sha256d_trace(header, nonce), event_k))
    return samples


def _graph_for_selected_samples(samples: Sequence[TraceSample], *, event_k: int, max_events: int = 8, max_controls: int = 8) -> HashEvidenceGraph:
    graph = HashEvidenceGraph()
    events = [s for s in samples if s.event][:max_events]
    controls = [s for s in samples if not s.event][:max_controls]
    for sample in events + controls:
        append_trace_evidence(
            graph,
            header_id=sample.header_id,
            header76=sample.trace.header76,
            nonce=sample.trace.nonce,
            digest=sample.trace.digest,
            leading_zero_bits=sample.trace.leading_zero_bits,
            round_records=sample.trace.stage1 + sample.trace.stage2,
            event_k=event_k,
        )
    return graph


def run_experiment(
    *,
    header_count: int = 15,
    nonces_per_header: int = 512,
    event_k: int = 8,
    top_k: int = 8,
    seed: int = 616,
    split_seed: str = 'bitmyner-marker-v0.1',
) -> dict:
    headers = deterministic_headers(header_count, seed=seed)
    samples = collect_samples(headers, nonces_per_header=nonces_per_header, event_k=event_k)
    candidates = raven_single_bit_population(
        stages=(1, 2),
        rounds=(0, 7, 15, 31, 47, 63),
        sources=('a', 'e', 't1', 'w'),
        bits=(0, 7, 15, 23, 31),
    )
    ranked = raven_rank(candidates, samples, seed=split_seed, min_marker_n=32, top_k=top_k)

    ranked_payload = []
    promoted = []
    for genome, evaluation in ranked:
        verdict = belzebub_promotion_verdict(evaluation, min_holdout_marker_n=32)
        entry = {
            'genome': genome.to_dict(),
            'evaluation': evaluation,
            'verdict': verdict,
        }
        ranked_payload.append(entry)
        if verdict['promote']:
            promoted.append(entry)

    graph = _graph_for_selected_samples(samples, event_k=event_k)
    if ranked_payload:
        top = ranked_payload[0]
        event_nodes = [n.node_id for n in graph.nodes if n.node_type == 'THRESHOLD_EVENT']
        if event_nodes:
            train = top['evaluation']['reports']['train'] or {}
            append_marker_association(
                graph,
                genome=top['genome'],
                event_node=event_nodes[0],
                metrics={
                    'scope': 'TRAINING_ASSOCIATION_ONLY',
                    'lift': train.get('lift'),
                    'lift_lcb': train.get('lift_lcb'),
                },
            )

    payload = {
        'schema': SCHEMA,
        'status': 'PASS_EXPERIMENT_COMPLETE',
        'roles': ROLES,
        'fixture': {
            'synthetic_header_fixture': True,
            'header_count': header_count,
            'nonces_per_header': nonces_per_header,
            'sample_count': len(samples),
            'event_k': event_k,
            'event_count': sum(s.event for s in samples),
            'seed': seed,
        },
        'candidate_count': len(candidates),
        'top_k': ranked_payload,
        'promoted_count': len(promoted),
        'promotion_semantics': {
            'cryptographic_advantage_assumed': False,
            'header_level_holdout_required': True,
            'association_edge_is_not_computational_edge': True,
            'production_policy_changed': False,
        },
        'graph': {
            'graph_id': graph.graph_id,
            'node_count': len(graph.nodes),
            'edge_count': len(graph.edges),
            'schema': graph.schema,
        },
    }
    payload['receipt_blake2b256'] = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'),
        digest_size=32,
    ).hexdigest()
    return payload
