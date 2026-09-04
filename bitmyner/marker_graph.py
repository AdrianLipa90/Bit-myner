from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

NODE_TYPES = {
    'JOB','HEADER','NONCE','ROUND_STATE','MESSAGE_WORD','FEATURE','MARKER','THRESHOLD_EVENT','SHARE','BLOCK','TAU_COST','GENOME','HASH'
}
EDGE_TYPES = {'COMPUTATIONAL_EDGE','ASSOCIATION_EDGE','EVIDENCE_EDGE','DERIVATION_EDGE'}


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def _id(prefix: str, payload: Any) -> str:
    return f"{prefix}:" + hashlib.blake2b(_canonical(payload), digest_size=24).hexdigest()


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_type: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GraphEdge:
    edge_id: str
    edge_type: str
    source: str
    target: str
    payload: dict[str, Any]


class HashEvidenceGraph:
    schema = 'BITMYNER_HASH_EVIDENCE_GRAPH_V0_1'

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[GraphEdge, ...]:
        return tuple(self._edges.values())

    def add_node(self, node_type: str, payload: dict[str, Any]) -> str:
        if node_type not in NODE_TYPES:
            raise ValueError(f'unsupported node_type: {node_type}')
        body = {'node_type': node_type, 'payload': payload}
        node_id = _id('n', body)
        node = GraphNode(node_id, node_type, dict(payload))
        existing = self._nodes.get(node_id)
        if existing is not None and existing != node:
            raise RuntimeError('append-only node collision')
        self._nodes.setdefault(node_id, node)
        return node_id

    def add_edge(self, edge_type: str, source: str, target: str, payload: dict[str, Any] | None = None) -> str:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f'unsupported edge_type: {edge_type}')
        if source not in self._nodes or target not in self._nodes:
            raise KeyError('both edge endpoints must exist before edge creation')
        payload = {} if payload is None else dict(payload)
        body = {'edge_type': edge_type, 'source': source, 'target': target, 'payload': payload}
        edge_id = _id('e', body)
        edge = GraphEdge(edge_id, edge_type, source, target, payload)
        existing = self._edges.get(edge_id)
        if existing is not None and existing != edge:
            raise RuntimeError('append-only edge collision')
        self._edges.setdefault(edge_id, edge)
        return edge_id

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema': self.schema,
            'nodes': [
                {'id': n.node_id, 'type': n.node_type, 'payload': n.payload}
                for n in sorted(self._nodes.values(), key=lambda x: x.node_id)
            ],
            'edges': [
                {'id': e.edge_id, 'type': e.edge_type, 'source': e.source, 'target': e.target, 'payload': e.payload}
                for e in sorted(self._edges.values(), key=lambda x: x.edge_id)
            ],
        }

    @property
    def graph_id(self) -> str:
        return hashlib.blake2b(_canonical(self.to_dict()), digest_size=32).hexdigest()

    def validate(self) -> None:
        for e in self._edges.values():
            if e.source not in self._nodes or e.target not in self._nodes:
                raise RuntimeError('dangling graph edge')
            if e.edge_type == 'COMPUTATIONAL_EDGE' and self._nodes[e.source].node_type == 'MARKER':
                raise RuntimeError('marker associations may not be mislabeled as computational edges')
            if e.edge_type == 'ASSOCIATION_EDGE' and self._nodes[e.source].node_type not in {'MARKER','FEATURE','GENOME'}:
                raise RuntimeError('association edge must originate at MARKER, FEATURE, or GENOME')


def append_trace_evidence(
    graph: HashEvidenceGraph,
    *,
    header_id: str,
    header76: bytes,
    nonce: int,
    digest: bytes,
    leading_zero_bits: int,
    round_records: Iterable[Any],
    event_k: int | None = None,
) -> dict[str, str]:
    header_node = graph.add_node('HEADER', {'header_id': header_id, 'header76_hex': header76.hex()})
    nonce_node = graph.add_node('NONCE', {'header_id': header_id, 'nonce': nonce})
    graph.add_edge('COMPUTATIONAL_EDGE', header_node, nonce_node, {'operator': 'append_nonce_le32'})

    previous = nonce_node
    first_state = None
    last_state = None
    for r in round_records:
        state_payload = {
            'stage': int(r.stage), 'round': int(r.round),
            'a': int(r.a), 'b': int(r.b), 'c': int(r.c), 'd': int(r.d),
            'e': int(r.e), 'f': int(r.f), 'g': int(r.g), 'h': int(r.h),
            't1': int(r.t1), 't2': int(r.t2), 'w': int(r.w),
        }
        state_node = graph.add_node('ROUND_STATE', state_payload)
        graph.add_edge('COMPUTATIONAL_EDGE', previous, state_node, {'operator': 'sha256_round'})
        previous = state_node
        if first_state is None:
            first_state = state_node
        last_state = state_node

    hash_node = graph.add_node('HASH', {'digest_hex': digest.hex(), 'leading_zero_bits': leading_zero_bits})
    graph.add_edge('COMPUTATIONAL_EDGE', previous, hash_node, {'operator': 'sha256d_finalize'})
    event_node = ''
    if event_k is not None:
        event_node = graph.add_node('THRESHOLD_EVENT', {
            'k': int(event_k),
            'hit': bool(leading_zero_bits >= event_k),
            'rule': 'H < 2^(256-k)',
        })
        graph.add_edge('EVIDENCE_EDGE', hash_node, event_node, {'relation': 'threshold_classification'})

    graph.validate()
    return {
        'header': header_node,
        'nonce': nonce_node,
        'first_state': first_state or '',
        'last_state': last_state or '',
        'hash': hash_node,
        'event': event_node,
    }


def append_marker_association(
    graph: HashEvidenceGraph,
    *,
    genome: dict[str, Any],
    event_node: str,
    metrics: dict[str, Any],
) -> dict[str, str]:
    genome_node = graph.add_node('GENOME', genome)
    marker_node = graph.add_node('MARKER', {'genome_id': genome.get('genome_id'), 'policy': genome.get('policy')})
    graph.add_edge('DERIVATION_EDGE', genome_node, marker_node, {'relation': 'genome_instantiates_marker'})
    graph.add_edge('ASSOCIATION_EDGE', marker_node, event_node, metrics)
    graph.validate()
    return {'genome': genome_node, 'marker': marker_node}
