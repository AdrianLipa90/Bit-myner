from bitmyner.gremlin_marker import deterministic_headers, collect_samples
from bitmyner.structural_marker import StructuralPredicate, structural_population, nonce_variation_filter


def test_structural_xor_matches_manual_trace_value():
    s=collect_samples(deterministic_headers(1,seed=9),nonces_per_header=2,event_k=2)[0]
    p=StructuralPredicate(1,3,'xor','e',0,1,'t1')
    r=s.trace.record(1,3)
    expected=((r.e ^ r.t1)&1)==1
    assert p.matches(s) is expected


def test_unlabelled_variation_filter_removes_known_header_only_stage1_round0_w():
    samples=collect_samples(deterministic_headers(12,seed=11),nonces_per_header=16,event_k=2)
    preds=(StructuralPredicate(1,0,'direct','w',0,0), StructuralPredicate(1,3,'direct','w',24,0))
    kept, receipt=nonce_variation_filter(samples,preds,split_seed='v4-test',min_variable_header_fraction=0.2)
    assert preds[0] not in kept
    assert preds[1] in kept
    assert receipt['label_access'] is False


def test_structural_population_is_deterministic_and_contains_xor():
    a=structural_population(); b=structural_population()
    assert a==b
    assert any(p.op=='xor' for p in a)
    assert len(a)>1000
