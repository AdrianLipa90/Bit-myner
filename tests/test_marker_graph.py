import pytest

from bitmyner.marker_graph import HashEvidenceGraph, append_trace_evidence, append_marker_association
from bitmyner.dncrypt_marker import single_bit_genome
from bitmyner.sha256_trace import sha256d_trace


def test_append_only_trace_graph_and_edge_classes():
    h = bytes(range(76))
    t = sha256d_trace(h, 7)
    g = HashEvidenceGraph()
    refs = append_trace_evidence(
        g,
        header_id='h1',
        header76=h,
        nonce=7,
        digest=t.digest,
        leading_zero_bits=t.leading_zero_bits,
        first_digest=t.first_digest,
        round_records=t.stage1 + t.stage2,
        event_k=8,
    )
    assert len(g.nodes) == 133
    assert refs['event']
    assert refs['first_hash']
    assert any(e.edge_type == 'COMPUTATIONAL_EDGE' for e in g.edges)
    assert any(e.edge_type == 'EVIDENCE_EDGE' for e in g.edges)
    gid = g.graph_id
    append_trace_evidence(
        g,
        header_id='h1', header76=h, nonce=7, digest=t.digest,
        leading_zero_bits=t.leading_zero_bits, first_digest=t.first_digest,
        round_records=t.stage1 + t.stage2, event_k=8,
    )
    assert g.graph_id == gid


def test_marker_association_cannot_be_computational_edge():
    h = bytes(range(76))
    t = sha256d_trace(h, 8)
    g = HashEvidenceGraph()
    refs = append_trace_evidence(g, header_id='h2', header76=h, nonce=8, digest=t.digest,
                                 leading_zero_bits=t.leading_zero_bits, first_digest=t.first_digest,
                                 round_records=t.stage1 + t.stage2, event_k=8)
    genome = single_bit_genome(1, 0, 'a', 0, 1).to_dict()
    assoc = append_marker_association(g, genome=genome, event_node=refs['event'], metrics={'lift': 1.1})
    assert any(e.edge_type == 'ASSOCIATION_EDGE' for e in g.edges)
    with pytest.raises(RuntimeError):
        g.add_edge('COMPUTATIONAL_EDGE', assoc['marker'], refs['event'])
        g.validate()
